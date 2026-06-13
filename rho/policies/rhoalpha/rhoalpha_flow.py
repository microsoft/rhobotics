"""
Phi4MM Flow Matching Model.

Implements continuous action prediction using flow matching.
"""

import math

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.utils.device_utils import get_safe_dtype
from torch import Tensor, nn

from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_model import (
    CrossAttentionActionExpert,
    LayerwiseCrossAttentionExpert,
    RhoAlphaModel,
    SimpleActionExpert,
)


class RhoAlphaFlowMatchingModel(RhoAlphaModel):
    """
    Flow Matching model for continuous action prediction.

    Extends RhoAlphaModel with:
    - Action projection and time embedding networks
    - Action expert transformer
    - Action prediction head
    - Flow matching training and sampling
    """

    def __init__(self, config: RhoAlphaConfig, **kwargs):
        super().__init__(config, **kwargs)

        # Flow matching specific components
        self.action_in_proj = nn.Linear(self.max_action_dim, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )
        self.action_head = nn.Linear(self.embed_dim, self.max_action_dim).to(
            device=self.device, dtype=self.dtype
        )

        self.action_time_mlp_in = nn.Linear(self.embed_dim * 2, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )
        self.action_time_mlp_out = nn.Linear(self.embed_dim, self.embed_dim).to(
            device=self.device, dtype=self.dtype
        )

        self.time_mlp_in = nn.Linear(self.embed_dim, self.embed_dim).to(device=self.device, dtype=self.dtype)
        self.time_mlp_out = nn.Linear(self.embed_dim, self.embed_dim).to(device=self.device, dtype=self.dtype)

        if config.attention_type == "self":
            self.action_expert = SimpleActionExpert(config).to(device=self.device, dtype=self.dtype)
        elif config.attention_type == "layerwise_cross":
            self.action_expert = LayerwiseCrossAttentionExpert(config).to(
                device=self.device, dtype=self.dtype
            )
        else:
            self.action_expert = CrossAttentionActionExpert(config).to(device=self.device, dtype=self.dtype)
        # Preserve positional encodings in fp32
        self.action_expert.positional_encoding = self.action_expert.positional_encoding.float()

        self._use_layerwise_cross = config.attention_type == "layerwise_cross"

    def sample_noise(self, shape, device):
        """Sample noise for flow matching."""
        return torch.randn(shape, dtype=torch.float32, device=device).to(self.dtype)

    def sample_beta(self, alpha, beta, bsize, device):
        """Sample from Beta distribution for time sampling."""
        g1 = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1).pow(1 / alpha)
        g2 = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1).pow(1 / beta)
        return g1 / (g1 + g2)

    def sample_time(self, bsize, device):
        """Sample time steps for flow matching."""
        strategy = self.config.time_sampling_strategy

        if strategy == "beta_reverse":
            t = self.sample_beta(1.0, 1.5, bsize, device)
        elif strategy == "uniform":
            t = torch.empty((bsize,), device=device, dtype=torch.float32).uniform_(0, 1)
        elif strategy == "logit_normal":
            t = torch.sigmoid(torch.randn((bsize,), device=device, dtype=torch.float32))
        else:  # default: "beta"
            t = self.sample_beta(1.5, 1.0, bsize, device)  # fp32
        t = t * 0.999 + 0.001

        return t.to(self.dtype)  # cast once at the boundary

    def create_sinusoidal_pos_embedding(
        self, time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
    ) -> Tensor:
        """Computes sine-cosine positional embedding vectors for scalar positions."""
        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")
        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

        dtype = get_safe_dtype(torch.float64, device.type)
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction

        # Compute the outer product
        scaling_factor = 1.0 / period * 2 * math.pi
        sin_input = scaling_factor[None, :] * time.float()[:, None]
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
        return pos_emb.to(dtype=self.dtype)

    def embed_state(self, state, noisy_actions, timestep):  # based on embed_suffix() in modeling_pi0.py
        """Embed state, noisy_actions, timestep to prepare for further processing."""
        embs = []

        # Store as bf16 for memory; linears will run under autocast bf16
        state = state.to(device=self.device, dtype=self.dtype)
        noisy_actions = noisy_actions.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device, dtype=self.dtype)

        # Embed state
        state_emb = self.state_projector(state)  # (batch_size, n_obs_steps, embed_dim)
        embs.append(state_emb)
        dtype = state_emb.dtype
        device = state_emb.device

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = self.create_sinusoidal_pos_embedding(
            timestep.to(self.device), self.embed_dim, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)  # (batch_size, chunk_size, embed_dim)

        embs = torch.cat(embs, dim=1)
        return embs

    def embed_time_for_cond(self, timestep):
        timestep = timestep.to(device=self.device, dtype=self.dtype)

        time_emb = self.create_sinusoidal_pos_embedding(
            timestep.to(self.device), self.embed_dim, min_period=4e-3, max_period=4.0, device=timestep.device
        )

        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = time_emb.type(dtype=self.dtype)
        return time_emb

    def forward(
        self,
        image,
        prompt,
        state,
        actions,
        noise=None,
        time=None,
        image_mask=None,
        training_mode=None,
        precomputed_hidden_state=None,
    ):
        """
        Training forward pass with flow matching.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            actions: Ground truth actions
            noise: Optional noise (sampled if None)
            time: Optional time steps (sampled if None)
            image_mask: Optional image masking
            training_mode: Training mode (ignored, always uses flow matching)
            precomputed_hidden_state: Optional tuple of (hidden_state, mask) to avoid recomputing

        Returns:
            losses: Per-element flow matching losses
        """
        # Store inputs in bf16 for memory
        actions = actions.to(dtype=self.dtype)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        else:
            noise = noise.to(dtype=self.dtype)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        else:
            time = time.to(dtype=self.dtype)

        # Upcast to fp32 for arithmetic
        a32 = actions.float()
        n32 = noise.float()
        t32 = time.float()

        time_expanded32 = t32[:, None, None]
        x_t32 = time_expanded32 * n32 + (1 - time_expanded32) * a32
        u_t32 = n32 - a32

        # Cast x_t/time back for downstream bf16 modules
        x_t = x_t32.to(self.dtype)

        # Prepare all inputs - use precomputed if available
        if precomputed_hidden_state is not None:
            image_text_embed, image_text_mask = precomputed_hidden_state
        else:
            if self._use_layerwise_cross:
                image_text_embed, image_text_mask = self.get_all_hidden_states(
                    image, prompt, image_mask=image_mask
                )
            else:
                image_text_embed, image_text_mask = self.get_image_text_hidden_state(
                    image, prompt, image_mask=image_mask
                )
        state_embed = self.embed_state(state, x_t, time)

        time_embed = self.embed_time_for_cond(time)

        # Forward through action expert
        output_embed = self.action_expert.forward(
            image_text_embed, state_embed, time_embed, image_text_attn_mask=image_text_mask
        )

        action_token = output_embed[:, -self.config.chunk_size :]
        v_t = self.action_head(action_token)

        # Compute loss in fp32 for stability
        losses = F.mse_loss(u_t32.to(self.device), v_t.float(), reduction="none")

        return losses

    def velocity_eval(self, state, x_t, time_scalar, precomputed_hidden_state):
        """Evaluate the flow velocity v(x_t, t) at a single denoise step.

        Shared interface used by rho.hil.noise_inverse_map. Both rhoalpha and
        pi0's flow models implement this with matching semantics; the inverter
        treats ``precomputed_hidden_state`` opaquely.

        Mirrors the inner body of ``sample_actions``'s denoise loop.

        Args:
            state: Robot state tensor (B, n_obs_steps, state_dim).
            x_t:   Current noise/action latent (B, chunk_size, max_action_dim).
            time_scalar: Float in [0, 1] -- the current diffusion time.
            precomputed_hidden_state: ``(image_text_embed, image_text_mask)`` from
                ``get_image_text_hidden_state``. Reused across steps without
                re-running the VLM.

        Returns:
            v_t: Velocity tensor (B, chunk_size, max_action_dim) in x_t's dtype.
        """
        image_text_embed, image_text_mask = precomputed_hidden_state
        bsize = state.shape[0]
        device = state.device
        time = (
            torch.tensor(float(time_scalar), dtype=torch.float32, device=device).expand(bsize).to(self.dtype)
        )
        state_embed = self.embed_state(state, x_t.to(self.dtype), time)
        time_emb = self.embed_time_for_cond(time)
        output_embed = self.action_expert.forward(
            image_text_embed, state_embed, time_emb, image_text_attn_mask=image_text_mask
        )
        action_token = output_embed[:, -self.config.chunk_size :]
        return self.action_head(action_token).to(x_t.dtype)

    def sample_actions_from_precomputed(self, state, precomputed_hidden_state, noise, num_steps=None):
        """Round-trip denoising entry point used by inversion verification.

        Thin wrapper over ``sample_actions`` that names the precomputed prefix
        conditioning explicitly and drops the image/prompt args (unused when
        ``precomputed_hidden_state`` is provided).
        """
        return self.sample_actions(
            image=None,
            prompt=None,
            state=state,
            noise=noise,
            precomputed_hidden_state=precomputed_hidden_state,
            num_steps=num_steps,
        )

    def sample_actions(
        self,
        image,
        prompt,
        state,
        noise=None,
        image_mask=None,
        precomputed_hidden_state=None,
        num_steps=None,
    ) -> Tensor:
        """
        Inference: denoise from t=1 to t=0 to generate actions.

        Args:
            image: List of image lists
            prompt: List of text prompts
            state: Robot state tensor
            noise: Optional initial noise (sampled if None)
            image_mask: Optional image masking
            precomputed_hidden_state: Optional ``(image_text_embed, image_text_mask)``
                tuple. When provided, skips the VLM forward pass and reuses these
                features -- useful when iterating Euler steps many times with the
                same conditioning.
            num_steps: Override the denoising step count. None = use the
                configured ``self.config.num_steps``.

        Returns:
            Final denoised actions
        """
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            x_t = self.sample_noise(actions_shape, device)
        else:
            x_t = noise.to(dtype=self.dtype)

        # Stash the initial noise so the policy wrapper can return it to the
        # DSRL trainer for transition recording.
        self._last_initial_noise = x_t.detach().float().cpu().numpy()

        n_steps = num_steps if num_steps is not None else self.config.num_steps
        dt32 = torch.tensor(-1.0 / n_steps, dtype=torch.float32, device=device)
        time32 = torch.tensor(1.0, dtype=torch.float32, device=device)
        if precomputed_hidden_state is not None:
            image_text_embed, image_text_mask = precomputed_hidden_state
        elif self._use_layerwise_cross:
            image_text_embed, image_text_mask = self.get_all_hidden_states(
                image, prompt, image_mask=image_mask
            )
        else:
            image_text_embed, image_text_mask = self.get_image_text_hidden_state(
                image, prompt, image_mask=image_mask
            )

        # Denoise from t=1 to t=0
        while time32 >= -dt32 / 2:
            expanded_time_bf16 = time32.expand(bsize).to(self.dtype)
            state_embed = self.embed_state(state, x_t.to(self.device), expanded_time_bf16)

            time_emb = self.embed_time_for_cond(expanded_time_bf16)

            output_embed = self.action_expert.forward(
                image_text_embed, state_embed, time_emb, image_text_attn_mask=image_text_mask
            )
            action_token = output_embed[:, -self.config.chunk_size :]
            v_t = self.action_head(action_token)

            # Euler step in fp32
            x_t32 = x_t.float() + dt32 * v_t.float()
            x_t = x_t32.to(self.dtype)
            time32 = time32 + dt32

        return x_t  # Final denoised actions (bf16)

    @torch.no_grad
    def sample_actions_rtc(
        self,
        image,
        prompt,
        state,
        inference_delay,
        execution_horizon,
        prev_actions=None,
        noise=None,
        beta=40.0,
    ) -> Tensor:
        """
        Guided inference algorithm for RTC as described in the paper (https://www.physicalintelligence.company/download/real_time_chunking.pdf).
        Original Algorithm:
        24: compute W using Eq. 5; right-pad prev_actions to length H; initialize A_0 ~ N(0,I)
        25: for τ=0 to 1 with stepsize 1/n do
        26:     f_A^1 = A' → A' + (1-τ)v_π(A',o,τ)  ⊳ Define denoising function (Eq. 3)
        27:     e = (prev_actions - f_A^1(A_τ))^T diag(W)  ⊳ Weighted error term from Eq. 2
        28:     g = e · ∂f_A^1/∂A'|_{A'=A_τ}  ⊳ Compute vector-Jacobian product from Eq. 2 via autodiff
        29:     A_{τ+1/n} = A_τ + 1/n v_π(A_τ,o,τ) + min(β, (1-τ)/τ · r^2/τ) g  ⊳ Integration step (Eq. 1)
        return A_1

        Args:
            obs: Dictionary containing observations
            prev_actions: Previous actions tensor (batch_size, chunk_size - s, action_dim)
            d: Threshold parameter
            s: Step parameter
            beta: Maximum guidance strength
            r: Guidance scaling factor

        Returns:
            torch.Tensor: Final denoised actions
        """
        bsize = state.shape[0]
        device = state.device

        # Right-pad prev_actions to length H
        H = self.config.chunk_size  # noqa: N806
        action_dim = self.max_action_dim

        w = self.compute_W_matrix_rtc(inference_delay, execution_horizon, action_dim)  # (H, action_dim)

        # Pad prev_actions: (batch_size, chunk_size - s, feature_action_dim) -> (batch_size, H, action_dim)
        # Create target tensor and copy data
        prev_actions_padded = torch.zeros(
            bsize, H, action_dim, dtype=prev_actions.dtype, device=prev_actions.device
        )
        prev_actions_padded[:, : prev_actions.shape[-2], : prev_actions.shape[-1]] = prev_actions

        w[prev_actions.shape[-2] :, prev_actions.shape[-1] :] = (
            0.0  # zero out weights for padded action dimensions
        )

        # our denoising process goes from t = 1 to t = 0, as opposed to PI which goes from τ = 0 to τ = 1
        # Initialize A_1 ~ N(0, I)
        actions_shape = (bsize, H, action_dim)
        A_tau = (  # noqa: N806
            self.sample_noise(actions_shape, device) if noise is None else noise.to(dtype=self.dtype)
        )

        # Get image-text embeddings once (they don't change during denoising)
        if self._use_layerwise_cross:
            image_text_embed, _ = self.get_all_hidden_states(image, prompt)
        else:
            image_text_embed, _ = self.get_image_text_hidden_state(image, prompt)

        # Step 25: Denoising loop from t = 1 to t = 0
        n_steps = self.config.num_steps
        dt = -1.0 / n_steps

        for step in range(n_steps):
            tau = 1 + dt * step  # current time t
            tau_tensor = torch.tensor(tau, dtype=torch.float32, device=device).expand(bsize).to(self.dtype)

            # Step 26: Define denoising function f_A^0 - this is where we estimate
            # the final denoised version (i.e. A_0)
            def denoising_function(A_prime, tau_=tau, tau_tensor_=tau_tensor):  # noqa: N803, B023
                """f_A^1(A') = A' + (1-τ)v_π(A',o,τ)"""
                state_embed = self.embed_state(state, A_prime, tau_tensor_)
                time_embed = self.embed_time_for_cond(tau_tensor_)
                output_embed = self.action_expert.forward(image_text_embed, state_embed, time_embed)
                action_token = output_embed[:, -H:]
                v_pi = self.action_head(action_token)
                return A_prime - (tau_) * v_pi, v_pi

            A_0, vjp_fn, v_pi = torch.func.vjp(denoising_function, A_tau, has_aux=True)  # noqa: N806

            error = (prev_actions_padded - A_0) * w.unsqueeze(0)
            (vjp_result,) = vjp_fn(error)

            # flip tau to compute the guidance coefficient
            tau = 1 - tau

            # TODO: the formula from the paper results in the vjp result being weighted
            # extremely small, which is not what we want
            # that's why we just use beta directly
            guidance_coeff = beta

            # if tau > 0:
            #     inv_r2 = (tau ** 2 + (1 - tau) ** 2) / ((1 - tau) ** 2)
            #     c = (1 - tau) / tau
            #     guidance_coeff = min(beta, c * inv_r2)
            # else:
            #     guidance_coeff = beta

            # Step 29: Integration step
            # A_{τ+1/n} = A_τ + (1/n) v_π(A_τ,o,τ) + min(β, (1-τ)/τ · r²/τ) g
            # changed this to subtract vjp result since in our case dt is negative,
            # in PI's version dt is positive
            A_tau = A_tau + dt * (v_pi - (guidance_coeff * vjp_result))  # noqa: N806
            A_tau = A_tau.to(self.dtype)  # noqa: N806

        return A_tau

    def compute_W_matrix_rtc(self, inference_delay, execution_horizon, action_dim):  # noqa: N802
        """
        Compute weight matrix W for RTC (Real Time Chunking) according to (https://www.physicalintelligence.company/download/real_time_chunking.pdf).

        Wi = {
            1                     if i < d
            (ci * e^(ci) - 1) / (e - 1)        if d ≤ i < H-s
            0                     if i ≥ H-s
        }

        where ci = (H-s-i) / (H-s-d+1), i ∈ {0,...,H-1}
        H = self.config.chunk_size (horizon length)

        Args:
            d: Threshold parameter
            s: Step parameter

        Returns:
            torch.Tensor: Weight matrix of shape (chunk_size, max_action_dim)
        """
        horizon = self.config.chunk_size  # noqa: N806

        # Create weight vector for the temporal dimension
        weights = torch.zeros(horizon, dtype=self.dtype, device=self.device)

        for i in range(horizon):
            if i < inference_delay:
                weights[i] = 1.0
            elif inference_delay <= i < horizon - execution_horizon:
                # ci = (horizon-s-i) / (horizon-s-d+1)
                ci = (horizon - execution_horizon - i) / (horizon - execution_horizon - inference_delay + 1)
                weights[i] = (ci * (math.exp(ci) - 1)) / (math.exp(1) - 1)
            else:  # i >= horizon - execution_horizon
                weights[i] = 0.0

        # Expand to matrix shape (chunk_size, max_action_dim)
        # Each action dimension gets the same temporal weighting
        W_matrix = weights.unsqueeze(1).expand(horizon, action_dim)  # noqa: N806

        return W_matrix
