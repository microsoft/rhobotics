import logging
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.utils import populate_queues
from torch import Tensor, nn

from rho.common.constants import ACTION, ACTION_TACTILE
from rho.common.constants import OBSERVATION_IMAGE as OBS_IMAGES
from rho.common.constants import OBSERVATION_LANG as OBS_TASK
from rho.common.constants import OBSERVATION_STATE as OBS_ROBOT
from rho.common.constants import OBSERVATION_TACTILE as OBS_TACTILE
from rho.policies.rhoalpha.configuration_rhoalpha import RhoAlphaConfig
from rho.policies.rhoalpha.rhoalpha_flow import RhoAlphaFlowMatchingModel
from rho.policies.rhoalpha.rhoalpha_policy import OBS_IMAGES_IS_PAD, RhoAlphaPolicy

logger = logging.getLogger(__name__)


def weight_init(m):  # https://github.com/siddhanthaldar/BAKU/blob/main/baku/utils.py#L51
    if isinstance(m, nn.Linear):
        # Cast to float32 for orthogonal init
        if m.weight.dtype != torch.float32:
            orig_dtype = m.weight.dtype
            m.weight.data = m.weight.data.float()
            nn.init.orthogonal_(m.weight.data)
            m.weight.data = m.weight.data.to(orig_dtype)
        else:
            nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)
    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        gain = nn.init.calculate_gain("relu")
        if m.weight.dtype != torch.float32:
            orig_dtype = m.weight.dtype
            m.weight.data = m.weight.data.float()
            nn.init.orthogonal_(m.weight.data, gain)
            m.weight.data = m.weight.data.to(orig_dtype)
        else:
            nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


# ============================================================================
# Phi4MM Tactile Flow Matching Model
# ============================================================================


class RhoAlphaTactileFlowMatching(RhoAlphaFlowMatchingModel):
    def __init__(
        self,
        config: RhoAlphaConfig | None = None,
    ):
        """
        Phi4MMFlowMatching wraps the Phi-4 multimodal backbone and projects state/action information
        into a shared embedding space for downstream action prediction.

        Args:
            config (RhoAlphaConfig): Configuration object specifying model hyperparameters.

        Key Attributes:
            - self.config.hidden_state_idx: Selects which layer embedding to use from Phi4MM.
                * 0: After the encoders, before the first transformer block.
                * 1: After the first transformer block.
                * ...
                * 33: After the last transformer block (final embedding).

        Main Components:
            - Loads and freezes the Phi-4 multimodal backbone.
            - Projects robot state and action vectors into the shared embedding space.
            - Combines image, text, and state embeddings for action prediction.
            - Uses a transformer-based expert to predict actions from embeddings.

        Notes:
            - Only image and text features are used for hidden state extraction.
            - The model supports selecting intermediate or final hidden states for downstream tasks.
        """
        super().__init__(config)

        self.max_tactile_dim = self.config.max_tactile_dim
        self.reshaped_tactile_dim = self.config.reshaped_tactile_dim
        self.use_tactile_head = self.config.use_tactile_head
        self.tactile_head_loss = self.config.tactile_head_loss
        self.beta = 0  # we will assign this later if needed
        self.combine_action_tactile_head = self.config.combine_action_tactile_head
        self.tactile_encoder = self.config.tactile_encoder

        if self.tactile_encoder == "linear":
            self.tactile_projector = nn.Linear(self.reshaped_tactile_dim, self.embed_dim).to(
                device=self.device, dtype=self.dtype
            )
        elif self.tactile_encoder == "mlp_relu":
            self.tactile_projector = nn.Sequential(
                nn.Linear(self.reshaped_tactile_dim, 2 * self.embed_dim),
                nn.ReLU(),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
            ).to(device=self.device, dtype=self.dtype)
        elif self.tactile_encoder == "mlp_gelu":
            self.tactile_projector = nn.Sequential(
                nn.Linear(self.reshaped_tactile_dim, 2 * self.embed_dim),
                nn.GELU(),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
            ).to(device=self.device, dtype=self.dtype)
        elif self.tactile_encoder == "mlp_silu":
            self.tactile_projector = nn.Sequential(
                nn.Linear(self.reshaped_tactile_dim, 2 * self.embed_dim),
                nn.SiLU(),
                nn.Linear(2 * self.embed_dim, self.embed_dim),
            ).to(device=self.device, dtype=self.dtype)
        else:
            raise NotImplementedError(f"Unknown tactile encoder type: {config.tactile_encoder}")

        # [WARN!] combine_action_tactile_head currently overrides use_tactile_head
        if self.combine_action_tactile_head:
            # override these action modules for adding in (action+tactile)
            input_dim = self.max_action_dim + self.max_tactile_dim
            self.action_in_proj = nn.Linear(input_dim, self.embed_dim).to(
                device=self.device, dtype=self.dtype
            )
            self.action_head = nn.Linear(self.embed_dim, input_dim).to(device=self.device, dtype=self.dtype)
            self.beta = config.tactile_loss_beta
        elif self.use_tactile_head:
            self.tactile_head = nn.Linear(self.embed_dim, self.max_tactile_dim).to(
                device=self.device, dtype=self.dtype
            )
            self.beta = config.tactile_loss_beta

        if self.config.baku_weight_init:
            self.apply_custom_weight_init()

    def apply_custom_weight_init(self):
        """
        apply orthogonal weight init and zero biases to all the modules OUTSIDE of vlm_backbone
        """
        self.state_projector.apply(weight_init)
        self.vlm_projector.apply(weight_init)
        self.action_in_proj.apply(weight_init)
        self.action_head.apply(weight_init)
        self.action_time_mlp_in.apply(weight_init)
        self.action_time_mlp_out.apply(weight_init)
        self.action_expert.apply(weight_init)
        if hasattr(self, "tactile_projector"):
            self.tactile_projector.apply(weight_init)
        if hasattr(self, "tactile_head"):
            self.tactile_head.apply(weight_init)

    def print_freezing_status(self):
        components = {
            "VLM Backbone": self.vlm_backbone,
            "  Language Encoder": self.vlm_backbone.model.embed_tokens,
            "  Vision Encoder": self.vlm_backbone.model.embed_tokens_extend.image_embed,
            "    Vision Transformer": self.vlm_backbone.model.embed_tokens_extend.image_embed.img_processor,
            "    Vision Projector": self.vlm_backbone.model.embed_tokens_extend.image_embed.img_projection,
            "Tactile Projector": self.tactile_projector,
            # "Tactile Head": self.tactile_head,
            "Action Expert": self.action_expert,
            "RhoAlphaPolicy": self,
        }

        logger.info("Freezing Status:")
        logger.info("-" * 50)
        for name, module in components.items():
            try:
                total = sum(p.numel() for p in module.parameters())
                trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
                status = "FROZEN" if trainable == 0 else f"UNLOCKED {trainable / total * 100:.0f}% trainable"
                logger.info(f"{name:20}\t: {status:15} ({trainable:>8,}/{total:>8,})")
            except Exception as e:
                logger.warning(f"{name:20}: NOT FOUND. {e}")

    def embed_state(
        self, state, tactile, noisy_actions, timestep
    ):  # based on embed_suffix() in modeling_pi0.py
        """Embed state, noisy_actions, timestep to prepare for further processing."""
        embs = []

        # Store as bf16 for memory; linears will run under autocast bf16
        state = state.to(device=self.device, dtype=self.dtype)
        tactile = tactile.to(device=self.device, dtype=self.dtype)
        noisy_actions = noisy_actions.to(device=self.device, dtype=self.dtype)
        timestep = timestep.to(device=self.device, dtype=self.dtype)

        # Embed state
        state_emb = self.state_projector(state)  # (batch_size, n_obs_steps, embed_dim)
        embs.append(state_emb)
        dtype = state_emb.dtype
        device = state_emb.device

        tactile_emb = self.tactile_projector(tactile)  # (batch_size, n_obs_steps, embed_dim)
        embs.append(tactile_emb)

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

    def forward(
        self, image, prompt, state, tactile, actions, action_tactile, noise=None, time=None, image_mask=None
    ):
        # based on PI0FlowMatching.forward()
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""

        if self.combine_action_tactile_head:
            actions = actions.to(dtype=self.dtype)
            action_tactile = action_tactile.to(dtype=self.dtype)
            # [WARN!] overwriting this var to keep other parts of the pass cleaner.
            actions = torch.cat([actions, action_tactile], dim=-1)
        else:
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
        # prepare all the inputs
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )
        state_embed = self.embed_state(state, tactile, x_t, time)

        time_embed = self.embed_time_for_cond(time)

        # embeds go in/out as (batch, num_tokens, embed_dim)
        # TODO pass in masks for attention
        output_embed = self.action_expert.forward(
            image_text_embed, state_embed, time_embed, image_text_attn_mask=image_text_mask
        )

        action_token = output_embed[:, -self.config.chunk_size :, :]  # (batch_size, chunk_size, embed_dim)
        v_t = self.action_head(action_token)  # (batch_size, chunk_size, action_dim)

        # Compute loss in fp32 for stability
        if self.combine_action_tactile_head:
            combined_loss = F.mse_loss(u_t32.to(self.device), v_t.float(), reduction="none")
            action_loss = combined_loss[:, :, : self.max_action_dim]
            tactile_loss = combined_loss[:, :, self.max_action_dim :]
        else:
            action_loss = F.mse_loss(u_t32.to(self.device), v_t.float(), reduction="none")

            tactile_loss = torch.zeros(
                *action_loss.shape[:-1],
                self.max_tactile_dim,
                device=action_loss.device,
                dtype=action_loss.dtype,
            )
            if self.use_tactile_head:
                tactile_token = output_embed[:, : self.config.chunk_size, :]  # (batch, chunk_size, embed_dim)
                pred_tactile = self.tactile_head(tactile_token)  # (batch, chunk_size, max_tactile_dim)
                if self.tactile_head_loss == "mse":
                    tactile_loss = F.mse_loss(action_tactile, pred_tactile.float(), reduction="none")
                elif self.tactile_head_loss == "flow":
                    raise NotImplementedError("individual tactile head -> flow loss not implemented yet")
                else:
                    raise ValueError(f"Unknown tactile loss type: {self.tactile_head_loss}")

        return action_loss, tactile_loss

    def sample_actions(
        self, image, prompt, state, tactile, noise=None, image_mask=None
    ) -> Tensor:  # from modeling_pi0.py
        """Do a full inference forward and compute the action (batch_size x chunk_size x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if self.combine_action_tactile_head:
            actions_shape = (
                bsize,
                self.config.chunk_size,
                self.config.max_action_dim + self.config.max_tactile_dim,
            )
        else:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)

        # if noise is None:
        #     x_t = self.sample_noise(actions_shape, device)
        # else:
        #     x_t = noise.to(dtype=self.dtype)
        x_t = self.sample_noise(actions_shape, device) if noise is None else noise.to(dtype=self.dtype)

        dt32 = torch.tensor(-1.0 / self.config.num_steps, dtype=torch.float32, device=device)
        time32 = torch.tensor(1.0, dtype=torch.float32, device=device)
        image_text_embed, image_text_mask = self.get_image_text_hidden_state(
            image, prompt, image_mask=image_mask
        )

        # Denoise from t=1 to t=0
        while time32 >= -dt32 / 2:
            expanded_time_bf16 = time32.expand(bsize).to(self.dtype)
            state_embed = self.embed_state(state, tactile, x_t.to(self.device), expanded_time_bf16)

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

        # if self.use_tactile_head:
        #     # in addition to the denoising action loop, get the tactile prediction too
        #     tactile_token = output_embed[:, : self.config.chunk_size, :]
        #     pred_tactile = self.tactile_head(tactile_token)
        return x_t  # Final denoised actions (bf16)

    @torch.no_grad
    def sample_actions_rtc(
        self,
        image,
        prompt,
        state,
        tactile,
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
            inference_delay: Inference delay parameter
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

        if self.combine_action_tactile_head:
            action_dim = self.max_action_dim + self.max_tactile_dim
        else:
            action_dim = self.max_action_dim

        w_matrix = self.compute_W_matrix_rtc(
            inference_delay, execution_horizon, action_dim
        )  # (H, action_dim)

        w_matrix[prev_actions.shape[-2] :, prev_actions.shape[-1] :] = (
            0.0  # zero out weights for padded action dimensions
        )

        # Pad prev_actions: (batch_size, chunk_size - s, feature_action_dim) -> (batch_size, H, action_dim)
        # Create target tensor and copy data
        prev_actions_padded = torch.zeros(
            bsize, H, action_dim, dtype=prev_actions.dtype, device=prev_actions.device
        )
        prev_actions_padded[:, : prev_actions.shape[-2], : prev_actions.shape[-1]] = prev_actions

        # our denoising process goes from t = 1 to t = 0, as opposed to PI which goes from τ = 0 to τ = 1
        # Initialize A_1 ~ N(0, I)
        actions_shape = (bsize, H, action_dim)
        A_tau = self.sample_noise(actions_shape, device) if noise is None else noise.to(dtype=self.dtype)  # noqa: N806

        # Get image-text embeddings once (they don't change during denoising)
        image_text_embed, _ = self.get_image_text_hidden_state(image, prompt)

        # Step 25: Denoising loop from t = 1 to t = 0
        n_steps = self.config.num_steps
        dt = -1.0 / n_steps

        for step in range(n_steps):  # noqa: B023
            tau = 1 + dt * step  # current time t
            tau_tensor = torch.tensor(tau, dtype=torch.float32, device=device).expand(bsize).to(self.dtype)

            # Step 26: Define denoising function f_A^0
            # this is where we estimate the final denoised version (i.e. A_0)
            def denoising_function(A_prime, tau_=tau, tau_tensor_=tau_tensor):  # noqa: N803
                """f_A^1(A') = A' + (1-τ)v_π(A',o,τ)"""
                state_embed = self.embed_state(state, tactile, A_prime, tau_tensor_)
                time_embed = self.embed_time_for_cond(tau_tensor_)
                output_embed = self.action_expert.forward(image_text_embed, state_embed, time_embed)
                action_token = output_embed[:, -H:]
                v_pi = self.action_head(action_token)
                return A_prime - (tau_) * v_pi, v_pi

            A_0, vjp_fn, v_pi = torch.func.vjp(denoising_function, A_tau, has_aux=True)  # noqa: N806

            error = (prev_actions_padded - A_0) * w_matrix.unsqueeze(0)
            (vjp_result,) = vjp_fn(error)

            # flip tau to compute the guidance coefficient
            tau = 1 - tau

            # TODO: the formula from the paper results in the vjp result being weighted extremely small,
            # which is not what we want - that's why we just use beta directly
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


# ============================================================================
# Phi4MM Tactile Policy
# ============================================================================


class RhoAlphaTactilePolicy(RhoAlphaPolicy):
    """
    Wrapper class around Phi4MMFlowMatching model to train and run inference within LeRobot.
    RhoAlphaPolicy: goal is to interface with LeRobot code infra,
        taking a batch of data and preparing inputs for Phi4MMFlowMatching
    Phi4MMFlowMatching: goal is take the batch data,
        prepare embeddings (either from Phi4MM or it's own projections), and pass them to ActionExpert
    ActionExpert: goal is to take input embeddings,
        pass them through a transformer, and provide output embeddings
    """

    config_class = RhoAlphaConfig
    name = "rhoalpha"

    def __init__(
        self,
        config: RhoAlphaConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here,
                it is expected that they will be passed with a call to `load_state_dict`
                before the policy is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.device = config.device

        # Model is already moved to device in its constructor
        self.model = RhoAlphaTactileFlowMatching(config)
        self.model.print_freezing_status()

        self.n_action_steps = config.n_action_steps

        # queues are populated during rollout of the policy, they contain the n latest
        # observations and actions
        self._queues = None

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""

        super().reset()
        if self.config.tactile_features:
            self._queues[OBS_TACTILE] = deque(maxlen=self.config.n_obs_steps)

    def load_from_pretrained(self, checkpoint_path):
        """
        Load pretrained weights from a RhoAlphaPolicy checkpoint into this tactile policy.

        Handles the key remapping between the two architectures:
        - RhoAlphaPolicy: model = RhoAlphaRoboticsModel, keys are model.flow_model.<component>
        - RhoAlphaTactilePolicy: model = RhoAlphaTactileFlowMatching, keys are model.<component>

        Modules that will be loaded from the pretrained checkpoint:
            vlm_backbone, vlm_projector, state_projector, action_expert,
            action_time_mlp_in/out, time_mlp_in/out,
            action_in_proj (if not combine_action_tactile_head),
            action_head (if not combine_action_tactile_head)

        Modules that remain randomly initialized (tactile-specific):
            tactile_projector, tactile_head (if use_tactile_head),
            action_in_proj (if combine_action_tactile_head, due to shape mismatch),
            action_head (if combine_action_tactile_head, due to shape mismatch)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading pretrained weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

        # Handle different checkpoint formats
        if "policy_state_dict" in checkpoint:
            state_dict = checkpoint["policy_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # Remove 'module.' prefix if present (from DDP training)
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # Remap RhoAlphaPolicy keys (model.flow_model.*) to tactile format (model.*)
        has_flow_model_prefix = any(k.startswith("model.flow_model.") for k in state_dict)
        if has_flow_model_prefix:
            logger.info("Detected RhoAlphaPolicy checkpoint (model.flow_model.*), remapping to model.*")
            remapped = {}
            for key, value in state_dict.items():
                if key.startswith("model.flow_model."):
                    new_key = key.replace("model.flow_model.", "model.", 1)
                    remapped[new_key] = value
                else:
                    remapped[key] = value
            state_dict = remapped

        # Determine which keys to skip, partially load, or fully load
        own_state = self.state_dict()
        skipped_keys = []
        loaded_keys = []
        partial_keys = []

        # Keys eligible for partial loading when combine_action_tactile_head is True.
        # action_in_proj: pretrained (embed, act_dim) → (embed, act_dim + tactile_dim)
        # action_head: pretrained (act_dim, embed) → (act_dim + tactile_dim, embed)
        partial_load_prefixes = {"model.action_in_proj.", "model.action_head."}

        for key, value in list(state_dict.items()):
            if key not in own_state:
                skipped_keys.append((key, "not in tactile model"))
                del state_dict[key]
            elif own_state[key].shape != value.shape:
                # Check if this key is eligible for partial loading
                if any(key.startswith(prefix) for prefix in partial_load_prefixes):
                    partial_keys.append((key, value))
                else:
                    skipped_keys.append(
                        (key, f"shape mismatch: ckpt {value.shape} vs model {own_state[key].shape}")
                    )
                del state_dict[key]
            else:
                loaded_keys.append(key)

        # Partially load action_in_proj and action_head weights.
        # The combined head concatenates [action, tactile] along the action dim,
        # so pretrained action weights occupy the first max_action_dim slice.
        if partial_keys:
            max_action_dim = self.config.max_action_dim
            logger.info(
                f"Partially loading {len(partial_keys)} keys "
                f"(action dims [:, :{max_action_dim}] from pretrained):"
            )

        for key, pretrained_value in partial_keys:
            param = own_state[key]
            logger.info(f"  {key}: pretrained {pretrained_value.shape} -> model {param.shape}")

            if key.startswith("model.action_in_proj.weight"):
                # weight shape: (embed_dim, in_features)
                # Copy pretrained columns for the action dimensions
                param[:, :max_action_dim] = pretrained_value
            elif key.startswith("model.action_head.weight"):
                # weight shape: (out_features, embed_dim)
                # Copy pretrained rows for the action dimensions
                param[:max_action_dim, :] = pretrained_value
            elif key.startswith("model.action_head.bias"):
                # bias shape: (out_features,)
                param[:max_action_dim] = pretrained_value
            elif key.startswith("model.action_in_proj.bias"):
                # bias shape: (embed_dim,) — same shape, should have been fully loaded
                # but handle it here for safety
                param[:] = pretrained_value
            else:
                logger.warning(f"  Unexpected partial key {key}, skipping")
                skipped_keys.append((key, "unexpected partial load key"))
                continue

            # Write the partially-filled param back into state_dict for loading
            state_dict[key] = param
            loaded_keys.append(key)

        # Find keys in the tactile model that have no pretrained weights
        missing_keys = [k for k in own_state if k not in state_dict]

        self.load_state_dict(state_dict, strict=False)

        # Log summary
        logger.info(f"Loaded {len(loaded_keys)} parameter tensors from pretrained checkpoint")
        if skipped_keys:
            logger.info(f"Skipped {len(skipped_keys)} keys from checkpoint:")
            for key, reason in skipped_keys:
                logger.info(f"  {key}: {reason}")
        if missing_keys:
            logger.info(f"Randomly initialized {len(missing_keys)} keys (not in pretrained checkpoint):")
            for key in missing_keys:
                logger.info(f"  {key} {own_state[key].shape}")

        del checkpoint, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._post_checkpoint_load()

        if hasattr(self, "device"):
            self.to(self.device)
            logger.info(f"Pretrained weights loaded and moved to {self.device}")
        else:
            logger.info("Pretrained weights loaded successfully")

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()

        batch = dict(batch)
        # consolidate all the image_features into one key: OBS_IMAGES
        batch = self.consolidate_images(batch)

        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._queues["action"]) == 0:
            batch = {
                OBS_IMAGES: torch.cat(
                    list(self._queues[OBS_IMAGES]), dim=1
                ),  # (batch_size, n_obs_steps*num_img_features, 3, H, W)
                OBS_IMAGES_IS_PAD: torch.cat(
                    list(self._queues[OBS_IMAGES_IS_PAD]), dim=1
                ),  # (batch_size, n_obs_steps*num_img_features)
                OBS_ROBOT: torch.stack(
                    list(self._queues[OBS_ROBOT]), dim=1
                ),  # (batch_size, n_obs_steps, max_action_dim)
                OBS_TACTILE: torch.stack(
                    list(self._queues[OBS_TACTILE]), dim=1
                ),  # (batch_size, n_obs_steps, max_tactile_dim)
                OBS_TASK: list(self._queues[OBS_TASK][0]),  # list of length batch_size
            }

            image, image_mask = self.prepare_image(batch)
            state = self.prepare_state(batch).to(self.device)
            tactile = self.prepare_tactile(batch).to(self.device)
            prompt = self.prepare_prompt(batch)

            actions = self.model.sample_actions(
                image, prompt, state, tactile, noise=noise, image_mask=image_mask
            )

            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            actions = actions[:, : self.config.n_action_steps, :original_action_dim]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._queues["action"].extend(actions.transpose(0, 1))
        return self._queues["action"].popleft()

    @torch.no_grad
    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """
        Sample a chunk_size action given environment observations.

        Args:
            batch (dict[str, Tensor]):
                Dictionary containing the following keys and shapes:
                    - 'observation.state': Tensor of shape (B, 1, state_dim)
                    - 'observation.image.agentview': Tensor of shape (B, C, H, W)
                    - 'observation.image.left_wrist': Tensor of shape (B, C, H, W)
                    - 'observation.image.right_wrist': Tensor of shape (B, C, H, W)
                    - 'observation.tactile': Tensor of shape (B, 1, tactile_dim)
                    - 'task': list of length B (e.g., ["put the plug into the socket"])

        Returns:
            dict: {
                "actions": Tensor of shape (B, chunk_size, action_dim)
                    where chunk_size and action_dim are determined by the policy configuration.
                "tactile_action": Tensor of shape (B, chunk_size, tactile_dim) or None
            }

        Notes:
            - All tensors should be on the same device (e.g., cuda).
            - Image stacking and batch formatting are handled internally.
        """
        self.eval()

        batch = dict(batch)
        # consolidate all the image_features into one key: OBS_IMAGES
        batch = self.consolidate_images(batch)

        image, _ = self.prepare_image(batch)  # assumes `observation.images`
        prompt = self.prepare_prompt(batch)  # assumes "task" key is present in the batch
        state = self.prepare_state(batch)  # assumes 'observation.state' key is present in the batch
        tactile = self.prepare_tactile(batch)  # assumes 'observation.tactile' key is present in the batch

        actions = self.model.sample_actions(image, prompt, state, tactile, noise=noise)

        # separate action head from action+tactile head
        tactile_actions = None
        output_dim = actions.shape[-1]
        if output_dim > self.config.max_action_dim:
            tactile_actions = actions[:, :, self.config.max_action_dim :]
            original_tactile_dim = self.config.input_features[OBS_TACTILE].shape[0]
            tactile_actions = tactile_actions[:, :, :original_tactile_dim]
            actions = actions[:, :, : self.config.max_action_dim]
        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return {"actions": actions, "tactile_action": tactile_actions}

    @torch.no_grad
    def sample_actions_rtc(
        self,
        batch: dict[str, Tensor],
        inference_delay: int,
        execution_horizon: int,
        prev_actions: Tensor | None = None,
        noise: Tensor | None = None,
        beta: float = 40.0,
    ) -> Tensor:
        # first time inferencing, so just return default actions
        if prev_actions is None:
            return self.sample_actions(batch, noise=noise)

        self.eval()

        batch = dict(batch)
        # consolidate all the image_features into one key: OBS_IMAGES
        batch = self.consolidate_images(batch)

        image, _ = self.prepare_image(batch)  # assumes `observation.images`
        prompt = self.prepare_prompt(batch)  # assumes "task" key is present in the batch
        state = self.prepare_state(batch)  # assumes 'observation.state' key is present in the batch
        tactile = self.prepare_tactile(batch)

        actions = self.model.sample_actions_rtc(
            image,
            prompt,
            state,
            tactile,
            inference_delay=inference_delay,
            execution_horizon=execution_horizon,
            prev_actions=prev_actions,
            noise=noise,
            beta=beta,
        )

        # separate action head from action+tactile head
        tactile_actions = None
        output_dim = actions.shape[-1]
        if output_dim > self.config.max_action_dim:
            tactile_actions = actions[:, :, self.config.max_action_dim :]
            original_tactile_dim = self.config.input_features[OBS_TACTILE].shape[0]
            tactile_actions = tactile_actions[:, :, :original_tactile_dim]
            actions = actions[:, :, : self.config.max_action_dim]

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        return {"actions": actions, "tactile_action": tactile_actions}

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""

        batch = dict(batch)
        # consolidate all the image_features into one key: OBS_IMAGES
        batch = self.consolidate_images(batch)

        # note: assumptions on the key names in the batch dict
        image, image_mask = self.prepare_image(batch)  # assumes `observation.images`
        prompt = self.prepare_prompt(batch)  # assumes "task" key is present in the batch
        state = self.prepare_state(batch)  # assumes 'observation.state' key is present in the batch
        tactile = self.prepare_tactile(batch)  # assumes 'observation.tactile' key is present in the batch
        action = self.prepare_action(batch)  # assumes 'action' key is present in the batch
        actions_is_pad = batch.get("action_is_pad")
        action_tactile = self.prepare_action_tactile(batch)

        loss_dict = {}
        action_loss, tactile_loss = self.model.forward(
            image,
            prompt,
            state,
            tactile,
            action,
            action_tactile,
            noise=None,
            time=None,
            image_mask=image_mask,
        )
        # loss_dict["losses_after_forward"] = losses.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_loss = action_loss * in_episode_bound.unsqueeze(-1)  # 64, 50, 1
            # loss_dict["losses_after_in_ep_bound"] = losses.clone()

        # Remove padding
        # losses = losses[:, :, : self.config.action_feature.shape[0]]
        # TODO get feedback, this seems like a bug, we don't want to consider loss on the action padding
        action_loss = action_loss[:, :, : self.config.max_action_dim]
        tactile_loss = tactile_loss[:, :, : self.config.max_tactile_dim]

        # loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = action_loss.mean() + self.model.beta * tactile_loss.mean()
        # For logging
        loss_dict["action_loss"] = action_loss.mean().item()
        loss_dict["tactile_loss"] = tactile_loss.mean().item()
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_tactile(self, batch):
        """Pad tactile inputs, optionally masking to a subset of the signal."""
        tactile = batch[OBS_TACTILE]  # (B, HISTORY, TACTILE_DIM)

        if self.config.tactile_mask == "anyskin":
            tactile = tactile[..., :15]
        elif self.config.tactile_mask == "anypressure":
            tactile = tactile[..., -4:]

        tactile = self.pad_vector(tactile, self.config.max_tactile_dim)  # (B, HISTORY, MAX_TACTILE_DIM)

        # reshape so n_steps dim is 1
        batch_size = tactile.shape[0]
        flattened_size = tactile.shape[1:].numel()
        tactile = tactile.reshape(batch_size, 1, flattened_size)  # (B, 1, HISTORY*MAX_TACTILE_DIM)
        return tactile

    def prepare_action_tactile(self, batch):
        """Pad action tactile"""
        actions = self.pad_vector(batch[ACTION_TACTILE], self.config.max_tactile_dim)
        return actions
