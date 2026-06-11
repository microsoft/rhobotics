"""
Inverse noise mapping for flow matching policy models.

Given a state s and target action a_target, finds w* such that
model.sample_actions_from_precomputed(s, precomputed, noise=w*) ≈ a_target.

Supports three inversion methods:
- "adam": Gradient-based optimization from random init (original behavior)
- "euler_reverse": Deterministic ODE reverse (t=0→1) — fast, no gradients
- "hybrid": Euler reverse for init, then Adam refinement
- "perstep_fp": Per-step fixed-point inversion (the default; favoured in practice)

Flow-model contract (the "FlowModelInverterProtocol")
-----------------------------------------------------
This module is base-policy-agnostic. Any flow-matching model can be inverted
as long as it conforms to:

  precomputed = flow_model.get_image_text_hidden_state(image, prompt)
      Run the VLM prefix forward once. The returned object is opaque to this
      module and is threaded back into velocity_eval / sample_actions_from_precomputed.
      It encodes the prefix conditioning that does NOT depend on x_t / time,
      so it can be cached across denoise steps.

  v_t = flow_model.velocity_eval(state, x_t, time_scalar, precomputed)
      Single velocity-field evaluation v(x_t, t). Returns a tensor shaped
      like x_t in x_t's dtype.

  recon = flow_model.sample_actions_from_precomputed(state, precomputed, noise, num_steps=None)
      Full denoise loop reusing the cached precomputed prefix. Used for the
      round-trip MSE check after inversion.

Attributes required on flow_model:
  - config.chunk_size, config.max_action_dim, config.num_steps
  - dtype  (the module's parameter dtype)

Concretely:
  - rhoalpha's RhoAlphaFlowMatchingModel: precomputed = (image_text_embed,
    image_text_mask). velocity_eval glues embed_state/embed_time_for_cond/
    action_expert.forward/action_head.
  - pi0's _PI0Pytorch: precomputed = (prefix_pad_masks, past_key_values).
    velocity_eval wraps _denoise_step.
Both encode the same prefix conditioning; the only difference is which layer
of the transformer caches it. This module never inspects the precomputed
object, so the architectural difference is invisible here.
"""

import logging

import torch

logger = logging.getLogger(__name__)


@torch.no_grad()
def euler_reverse_noise_map(
    flow_model,
    image,
    prompt,
    state,
    a_target,
    num_steps=None,
    precomputed_hidden_state=None,
    verify_roundtrip=True,
):
    """Deterministic ODE reverse: run the flow forward (t=0→1) to map clean actions to noise.

    This mirrors the 'reverse' direction from hil_dsrl_pi0's _build_euler_fn.
    Starting from clean target actions at t=0, we integrate forward with dt=+1/N
    to recover the corresponding noise at t=1.

    Args:
        flow_model: Flow matching model instance (weights should be frozen).
        image: Image input — List[List[Tensor]], length = batch_size.
        prompt: Text prompt — List[str], length = batch_size.
        state: Robot state tensor (batch_size, n_obs_steps, state_dim).
        a_target: Target actions (batch_size, chunk_size, action_dim).
        num_steps: Number of Euler steps for the reverse ODE. None = model default.
        precomputed_hidden_state: Optional (embed, mask) tuple to skip VLM forward.
        verify_roundtrip: If True, denoise recovered noise back and report MSE.

    Returns:
        w: (batch_size, chunk_size, max_action_dim) — recovered noise vectors.
        roundtrip_mse: (batch_size,) — reconstruction MSE (0 if verify_roundtrip=False).
        metadata: dict with diagnostics.
    """
    device = state.device
    dtype = flow_model.dtype
    B = state.shape[0]

    N = num_steps or flow_model.config.num_steps
    dt = 1.0 / N

    # Precompute VLM features
    if precomputed_hidden_state is not None:
        img_text_embed, img_text_mask = precomputed_hidden_state
    else:
        img_text_embed, img_text_mask = flow_model.get_image_text_hidden_state(image, prompt)
    precomputed = (img_text_embed, img_text_mask)

    # Start from clean target actions at t=0
    max_action_dim = flow_model.config.max_action_dim
    x_t = a_target.to(device=device, dtype=dtype)

    # Pad to max_action_dim if needed
    if x_t.shape[-1] < max_action_dim:
        x_t = torch.nn.functional.pad(x_t, (0, max_action_dim - x_t.shape[-1]))

    # Euler forward integration: t=0 → t=1 with dt=+1/N
    for step in range(N):
        time = step * dt
        # Get velocity from the flow model's action expert
        v_t = (
            flow_model.sample_actions(
                image,
                prompt,
                state,
                noise=x_t,
                precomputed_hidden_state=precomputed,
                num_steps=1,
                _euler_override_time=time,
            )
            if hasattr(flow_model, "_supports_euler_override")
            else _euler_step_via_velocity(flow_model, image, prompt, state, x_t, time, precomputed)
        )
        x_t = x_t + dt * v_t

    w = x_t.float()  # recovered noise at t=1

    # Round-trip verification
    roundtrip_mse = torch.zeros(B, device=device)
    if verify_roundtrip:
        a_recovered = flow_model.sample_actions_from_precomputed(
            state,
            precomputed,
            noise=w.to(dtype),
            num_steps=None,  # full steps
        )
        roundtrip_mse = (
            (a_recovered.float() - a_target.to(device=device, dtype=torch.float32)).pow(2).mean(dim=(1, 2))
        )

    metadata = {
        "method": "euler_reverse",
        "num_steps": N,
        "w_norm_mean": w.abs().mean().item(),
        "w_norm_max": w.abs().max().item(),
        "roundtrip_mse_mean": roundtrip_mse.mean().item(),
        "roundtrip_mse_max": roundtrip_mse.max().item(),
    }

    logger.info(
        f"Euler reverse: B={B}, N={N} | "
        f"roundtrip MSE: {roundtrip_mse.mean().item():.6f} (mean) | "
        f"||w||_mean={w.abs().mean().item():.4f}"
    )

    return w, roundtrip_mse, metadata


@torch.no_grad()
def _velocity_eval(flow_model, state, x_t, time_scalar, precomputed):
    """Evaluate the flow model's velocity field v(x_t, t).

    Thin proxy to ``flow_model.velocity_eval(...)`` -- the model owns the
    architecture-specific glue (rhoalpha unrolls embed_state / action_expert /
    action_head; pi0 wraps _denoise_step). See the module docstring for the
    full FlowModelInverterProtocol contract.

    Returns a tensor in x_t.dtype shaped (B, chunk_size, max_action_dim).
    """
    return flow_model.velocity_eval(state, x_t, time_scalar, precomputed)


# Backward-compat alias: the original euler_reverse path called this name.
def _euler_step_via_velocity(flow_model, image, prompt, state, x_t, time, precomputed):
    return _velocity_eval(flow_model, state, x_t, time, precomputed)


@torch.no_grad()
def fixed_point_noise_map(
    flow_model,
    image,
    prompt,
    state,
    a_target,
    refine_steps=3,
    num_steps=None,
    precomputed_hidden_state=None,
):
    """Residual-correction fixed-point inversion (Mike: "FP").

    Algorithm (lifted from hil_dsrl_pi0/flow_matching_inverter.py:899):
        noise <- euler_reverse(target)
        for _ in refine_steps:
            recon  = denoise(noise)
            res    = target - recon
            target'= target + res            # overshoot by the gap
            noise <- euler_reverse(target')
        return noise

    Each refinement only costs one denoise + one reverse pass; no implicit
    inner loop. Works well when the discrete Euler reverse is close but not
    exact -- residual feedback compensates for the discretization drift.
    """
    device = state.device
    dtype = flow_model.dtype

    if precomputed_hidden_state is None:
        precomputed = flow_model.get_image_text_hidden_state(image, prompt)
    else:
        precomputed = precomputed_hidden_state

    a_target_f = a_target.to(device=device, dtype=torch.float32)

    # Initial euler_reverse from raw target.
    noise, _, _ = euler_reverse_noise_map(
        flow_model,
        image,
        prompt,
        state,
        a_target,
        num_steps=num_steps,
        precomputed_hidden_state=precomputed,
        verify_roundtrip=False,
    )

    corrected_target = a_target_f
    for _ in range(refine_steps):
        recon = flow_model.sample_actions_from_precomputed(
            state,
            precomputed,
            noise=noise.to(dtype),
        )
        residual = a_target_f - recon.float()
        corrected_target = a_target_f + residual
        noise, _, _ = euler_reverse_noise_map(
            flow_model,
            image,
            prompt,
            state,
            corrected_target,
            num_steps=num_steps,
            precomputed_hidden_state=precomputed,
            verify_roundtrip=False,
        )

    # Final round-trip error at full denoise steps.
    recon = flow_model.sample_actions_from_precomputed(
        state,
        precomputed,
        noise=noise.to(dtype),
    )
    error = (recon.float() - a_target_f).pow(2).mean(dim=(-2, -1))

    metadata = {
        "method": "fixed_point",
        "refine_steps": refine_steps,
        "w_norm_mean": noise.abs().mean().item(),
        "w_norm_max": noise.abs().max().item(),
        "roundtrip_mse_mean": error.mean().item(),
        "roundtrip_mse_max": error.max().item(),
    }
    logger.info(
        f"Fixed-point inversion: refine_steps={refine_steps} | "
        f"roundtrip MSE: {error.mean().item():.6f} (mean) | "
        f"||w||_mean={noise.abs().mean().item():.4f}"
    )
    return noise.float(), error, metadata


@torch.no_grad()
def perstep_fp_noise_map(
    flow_model,
    image,
    prompt,
    state,
    a_target,
    fp_per_step=3,
    num_steps=None,
    precomputed_hidden_state=None,
):
    """Per-step fixed-point inversion (Mike: "FP per step", the favourite).

    Lifted from hil_dsrl_pi0/flow_matching_inverter.py:110.

    Inverts the EXACT discrete Euler denoising step at each timestep.
    Forward Euler (denoising, t: 1->0): x_{t+dt_fwd} = x_t + dt_fwd * v(x_t, t)
    To go reverse (t: 0->1, dt_rev = -dt_fwd = 1/N) we need x_next satisfying:
        x_next = x_prev + dt_rev * v(x_next, t_next)
    Fixed-point iterate inside each step:
        x_next^(0) = x_prev + dt_rev * v(x_prev, t_prev)
        x_next^(j+1) = x_prev + dt_rev * v(x_next^(j), t_next)

    fp_per_step=3 is usually enough -- contraction factor ~0.1-0.3 per
    iteration at dt=1/10.
    """
    device = state.device
    dtype = flow_model.dtype

    if precomputed_hidden_state is None:
        precomputed = flow_model.get_image_text_hidden_state(image, prompt)
    else:
        precomputed = precomputed_hidden_state

    N = num_steps or flow_model.config.num_steps
    dt_rev = 1.0 / N

    max_action_dim = flow_model.config.max_action_dim

    x_prev = a_target.to(device=device, dtype=dtype)
    if x_prev.shape[-1] < max_action_dim:
        x_prev = torch.nn.functional.pad(x_prev, (0, max_action_dim - x_prev.shape[-1]))

    for step in range(N):
        t_prev = step * dt_rev
        t_next = t_prev + dt_rev

        v_init = _velocity_eval(flow_model, state, x_prev, t_prev, precomputed)
        x_next = x_prev + dt_rev * v_init

        for _ in range(fp_per_step):
            v_est = _velocity_eval(flow_model, state, x_next, t_next, precomputed)
            x_next = x_prev + dt_rev * v_est

        x_prev = x_next

    noise = x_prev.float()

    # Round-trip at full denoise steps.
    a_target_f = a_target.to(device=device, dtype=torch.float32)
    recon = flow_model.sample_actions_from_precomputed(
        state,
        precomputed,
        noise=noise.to(dtype),
    )
    error = (recon.float() - a_target_f).pow(2).mean(dim=(-2, -1))

    metadata = {
        "method": "perstep_fp",
        "num_steps": N,
        "fp_per_step": fp_per_step,
        "w_norm_mean": noise.abs().mean().item(),
        "w_norm_max": noise.abs().max().item(),
        "roundtrip_mse_mean": error.mean().item(),
        "roundtrip_mse_max": error.max().item(),
    }
    logger.info(
        f"Per-step FP inversion: N={N}, fp_per_step={fp_per_step} | "
        f"roundtrip MSE: {error.mean().item():.6f} (mean) | "
        f"||w||_mean={noise.abs().mean().item():.4f}"
    )
    return noise, error, metadata


@torch.enable_grad()
def inverse_noise_map(
    flow_model,
    image,
    prompt,
    state,
    a_target,
    n_restarts=1,
    optimizer_steps=20,
    lr=0.01,
    b_W=2.0,
    lambda_reg=0.01,
    precomputed_hidden_state=None,
    euler_steps_override=5,
    method="adam",
    init_noise=None,
):
    """
    Find w* such that flow_model.sample_actions(image, prompt, state, noise=w*) ≈ a_target.

    Supports five inversion methods:
    - "adam": Gradient-based optimization from random or provided init (original behavior).
    - "euler_reverse": Deterministic ODE reverse, no gradients needed.
    - "hybrid": Euler reverse first, then Adam refinement starting from that init.
    - "fixed_point" (alias "fp"): residual-correction loop on top of euler_reverse.
    - "perstep_fp" (alias "fp_per_step"): per-step fixed-point inversion of the
      discrete Euler denoise map. Highest accuracy; preferred in practice.

    All restarts run in parallel as one big batch for GPU efficiency.

    Args:
        flow_model: Flow matching model instance (weights should be frozen).
        image: Image input — List[List[Tensor]], length = batch_size.
        prompt: Text prompt — List[str], length = batch_size.
        state: Robot state tensor (batch_size, n_obs_steps, state_dim).
        a_target: Target actions (batch_size, chunk_size, action_dim).
        n_restarts: Number of random initializations (run in parallel). Ignored for euler_reverse.
        optimizer_steps: Number of Adam update steps to refine w.
        lr: Learning rate for Adam optimizer.
        b_W: Noise magnitude bound — clamp w to [-b_W, b_W].
        lambda_reg: Regularization weight for ||w||².
        precomputed_hidden_state: Optional (embed, mask) tuple to skip VLM forward.
        euler_steps_override: Euler denoising steps used during optimization.
                              Fewer steps = faster but approximate gradients.
                              Final eval always uses the model's full step count.
                              None = use full steps throughout.
        method: Inversion method — "adam", "euler_reverse", or "hybrid".
        init_noise: Optional (batch_size, chunk_size, max_action_dim) initial noise for Adam.
                    If provided, used instead of random init (n_restarts is forced to 1).

    Returns:
        best_w: (batch_size, chunk_size, max_action_dim) — recovered noise vectors.
        best_losses: (batch_size,) — per-sample final reconstruction MSE (at full Euler steps).
        metadata: dict with diagnostics.
    """
    device = state.device
    dtype = flow_model.dtype
    B = state.shape[0]

    if method == "euler_reverse":
        return euler_reverse_noise_map(
            flow_model,
            image,
            prompt,
            state,
            a_target,
            precomputed_hidden_state=precomputed_hidden_state,
            verify_roundtrip=True,
        )

    if method in ("fixed_point", "fp"):
        return fixed_point_noise_map(
            flow_model,
            image,
            prompt,
            state,
            a_target,
            refine_steps=optimizer_steps,  # reuse the existing knob
            precomputed_hidden_state=precomputed_hidden_state,
        )

    if method in ("perstep_fp", "fp_per_step"):
        return perstep_fp_noise_map(
            flow_model,
            image,
            prompt,
            state,
            a_target,
            fp_per_step=optimizer_steps if optimizer_steps and optimizer_steps < 10 else 3,
            precomputed_hidden_state=precomputed_hidden_state,
        )

    if method == "hybrid" and init_noise is None:
        # Run euler_reverse first to get init_noise, then fall through to Adam
        euler_w, euler_mse, euler_meta = euler_reverse_noise_map(
            flow_model,
            image,
            prompt,
            state,
            a_target,
            precomputed_hidden_state=precomputed_hidden_state,
            verify_roundtrip=False,
        )
        init_noise = euler_w
        logger.info(
            f"Hybrid: euler_reverse init done, ||w||_mean={euler_w.abs().mean().item():.4f}, "
            f"proceeding to Adam refinement"
        )

    # When init_noise is provided, force n_restarts=1 (we have a good starting point)
    R = 1 if init_noise is not None else n_restarts
    BR = B * R

    return _inverse_noise_map_inner(
        flow_model,
        image,
        prompt,
        state,
        a_target,
        B,
        R,
        BR,
        device,
        dtype,
        n_restarts=R,
        optimizer_steps=optimizer_steps,
        lr=lr,
        b_W=b_W,
        lambda_reg=lambda_reg,
        precomputed_hidden_state=precomputed_hidden_state,
        euler_steps_override=euler_steps_override,
        init_noise=init_noise,
    )


def _inverse_noise_map_inner(
    flow_model,
    image,
    prompt,
    state,
    a_target,
    B,
    R,
    BR,
    device,
    dtype,
    n_restarts,
    optimizer_steps,
    lr,
    b_W,
    lambda_reg,
    precomputed_hidden_state,
    euler_steps_override,
    init_noise=None,
):
    # Precompute VLM features once
    if precomputed_hidden_state is not None:
        img_text_embed, img_text_mask = precomputed_hidden_state
    else:
        with torch.no_grad():
            img_text_embed, img_text_mask = flow_model.get_image_text_hidden_state(image, prompt)

    # Expand everything by n_restarts: [B, ...] -> [B*R, ...]
    embed_br = img_text_embed.repeat_interleave(R, dim=0)
    mask_br = img_text_mask.repeat_interleave(R, dim=0)
    state_br = state.repeat_interleave(R, dim=0)
    a_target_br = a_target.to(device=device, dtype=torch.float32).detach().repeat_interleave(R, dim=0)
    image_br = [img for img in image for _ in range(R)]
    prompt_br = [p for p in prompt for _ in range(R)]
    precomputed_br = (embed_br, mask_br)

    noise_shape = (BR, flow_model.config.chunk_size, flow_model.config.max_action_dim)

    # Initialize w: use provided init_noise or random
    if init_noise is not None:
        # init_noise is (B, chunk_size, max_action_dim) — expand to (BR, ...)
        w = init_noise.to(device=device, dtype=torch.float32).detach().repeat_interleave(R, dim=0).clone()
    else:
        w = torch.randn(noise_shape, device=device, dtype=torch.float32)
    w = w.clamp(-b_W, b_W)
    w.requires_grad_(True)

    optimizer = torch.optim.Adam([w], lr=lr)
    loss_history = []

    for _step in range(optimizer_steps):
        optimizer.zero_grad()
        w.data.clamp_(-b_W, b_W)

        # euler_steps_override: use fewer Euler denoising steps for speed
        # (approximate gradients are good enough for optimizer direction)
        a_pred = flow_model.sample_actions(
            image_br,
            prompt_br,
            state_br,
            noise=w.to(dtype),
            precomputed_hidden_state=precomputed_br,
            num_steps=euler_steps_override,
        )

        # Per-sample MSE (reduce over chunk_size and action_dim, keep batch)
        per_sample_mse = (a_pred.float() - a_target_br).pow(2).mean(dim=(1, 2))  # (BR,)
        recon_loss = per_sample_mse.mean()
        reg_loss = lambda_reg * w.pow(2).mean()
        loss = recon_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([w], max_norm=10.0)
        optimizer.step()

        loss_history.append(recon_loss.item())

    # Final eval at FULL Euler steps (num_steps=None → model default)
    w_detached = w.detach()
    with torch.no_grad():
        a_final = flow_model.sample_actions(
            image_br,
            prompt_br,
            state_br,
            noise=w_detached.to(dtype),
            precomputed_hidden_state=precomputed_br,
            num_steps=None,  # full steps for accurate quality measurement
        )
        final_mse = (a_final.float() - a_target_br).pow(2).mean(dim=(1, 2))  # (BR,)

    # Reshape to (B, R) and pick best restart per sample
    final_mse_reshaped = final_mse.view(B, R)
    best_restart_idx = final_mse_reshaped.argmin(dim=1)  # (B,)

    w_reshaped = w_detached.view(B, R, *noise_shape[1:])
    best_w = w_reshaped[torch.arange(B, device=device), best_restart_idx]  # (B, chunk, action_dim)
    best_losses = final_mse_reshaped[torch.arange(B, device=device), best_restart_idx]  # (B,)

    full_euler = flow_model.config.num_steps
    metadata = {
        "loss_history": loss_history,
        "best_losses_per_sample": best_losses.tolist(),
        "all_restart_losses": final_mse_reshaped.tolist(),
        "w_norm_mean": best_w.abs().mean().item(),
        "w_norm_max": best_w.abs().max().item(),
        "w_norm_l2_per_sample": best_w.view(B, -1).pow(2).sum(dim=1).sqrt().tolist(),
        "within_bounds": (best_w.abs() <= b_W).all().item(),
        "euler_steps_override": euler_steps_override,
        "euler_steps_full": full_euler,
    }

    logger.info(
        f"Inverse map: B={B}, R={R}, optimizer_steps={optimizer_steps}, "
        f"euler={euler_steps_override or full_euler}/{full_euler} | "
        f"best MSE: {best_losses.mean().item():.6f} (mean), {best_losses.max().item():.6f} (worst) | "
        f"||w||_mean={best_w.abs().mean().item():.4f}"
    )

    return best_w, best_losses, metadata
