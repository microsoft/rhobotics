"""Invert the flow sampler: recover the noise that would have produced a given action chunk.

Online adaptation needs the inverse of the usual direction. The sampler integrates an ODE
from noise ``w`` to actions ``a``; to learn a noise policy from demonstrated actions we need
``a -> w``, so a corrected action chunk becomes a supervision target in noise space.

``perstep_fp`` inverts the *exact discrete Euler step* the sampler takes, rather than
integrating a continuous reverse ODE. Forward (denoising, t: 1 -> 0) each step is

    x_{t+dt_fwd} = x_t + dt_fwd * v(x_t, t)

so going backwards (t: 0 -> 1, ``dt_rev = -dt_fwd = 1/N``) needs the ``x_next`` satisfying

    x_next = x_prev + dt_rev * v(x_next, t_next)

which is implicit in ``x_next``. A short fixed-point iteration solves it:

    x_next^(0)   = x_prev + dt_rev * v(x_prev, t_prev)
    x_next^(j+1) = x_prev + dt_rev * v(x_next^(j), t_next)

Judge the result by ``error`` -- the round-trip -- and not by whether the recovered noise
matches the noise a sample happened to start from. **Preimages are not unique.** Measured on
rho-base, the forward map contracts noise differences by ~0.44 and is near-insensitive along
some directions, so many different noises decode to nearly the same action chunk. Inverting a
chunk therefore returns *a* preimage, not *the* one. That is exactly what online adaptation
needs (noise that reproduces the expert's actions) and ``perstep_fp`` is deterministic, so the
targets stay self-consistent -- but an equality check against a known input will fail and does
not indicate a bug.

The fixed point is only contracting while ``dt * ||dv/dx|| < 1``. Where it is not, extra
iterations make things worse rather than better; that shows up as the round-trip error rising
with ``fp_per_step``. Raising ``num_steps`` (smaller dt) is the fix, not more iterations.
``error`` is returned so unreliable inversions can be dropped instead of poisoning the target.

The model must provide ``velocity_eval`` and ``sample_actions_from_precomputed``
(see ``rho.policies.rho.rho_model``).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def perstep_fp_noise_map(
    flow_model,
    state: torch.Tensor,
    a_target: torch.Tensor,
    precomputed_hidden_state,
    fp_per_step: int = 5,
    num_steps: int | None = None,
    roundtrip_steps: int | None = None,
):
    """Recover the noise that denoises to ``a_target``.

    Args:
        flow_model: model exposing ``velocity_eval`` / ``sample_actions_from_precomputed``.
        state: (B, n_obs_steps, state_dim) padded observation state.
        a_target: (B, chunk, action_dim) NORMALIZED target actions, as the policy emits them.
        precomputed_hidden_state: ``(embed, mask)`` from ``get_image_text_hidden_state``.
            Reused across every step, so the VLM runs once per inversion.
        fp_per_step: fixed-point iterations per denoise step.
        num_steps: denoise steps; defaults to the model's own, which is what the sampler
            will use at serving time. Mismatching it silently desyncs the inversion.
        roundtrip_steps: score the round-trip on only the first N chunk rows -- the rows
            that carried real supervision when the base policy was trained.

    Returns:
        (noise, error) -- noise (B, chunk, max_action_dim) float32, and per-sample
        round-trip MSE (B,).
    """
    device, dtype = state.device, flow_model.dtype
    n_steps = num_steps or flow_model.config.num_steps
    dt_rev = 1.0 / n_steps
    max_action_dim = flow_model.config.max_action_dim

    # Accumulate in fp32. The forward sampler takes its Euler step in fp32 and only casts
    # back to the model dtype to evaluate the velocity; inverting in bf16 instead loses far
    # more precision than the fixed point converges to, and the round-trip does not close.
    x_prev = a_target.to(device=device, dtype=torch.float32)
    if x_prev.shape[-1] < max_action_dim:
        x_prev = F.pad(x_prev, (0, max_action_dim - x_prev.shape[-1]))

    for step in range(n_steps):
        t_prev = step * dt_rev
        t_next = t_prev + dt_rev
        v_init = flow_model.velocity_eval(
            state, x_prev.to(dtype), t_prev, precomputed_hidden_state
        ).float()
        x_next = x_prev + dt_rev * v_init
        for _ in range(fp_per_step):
            v_est = flow_model.velocity_eval(
                state, x_next.to(dtype), t_next, precomputed_hidden_state
            ).float()
            x_next = x_prev + dt_rev * v_est
        x_prev = x_next

    noise = x_prev

    # Round-trip: decode the recovered noise and compare against what it came from.
    target = a_target.to(device=device, dtype=torch.float32)
    recon = flow_model.sample_actions_from_precomputed(
        state, precomputed_hidden_state, noise=noise.to(dtype), num_steps=n_steps
    ).float()[..., : target.shape[-1]]
    if roundtrip_steps is not None:
        recon, target = recon[:, :roundtrip_steps], target[:, :roundtrip_steps]
    error = ((recon - target) ** 2).mean(dim=tuple(range(1, target.ndim)))
    return noise, error
