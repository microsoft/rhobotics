OBSERVATION_PREFIX = "observation"
OBSERVATION_IMAGE = "observation.image"
OBSERVATION_ENVIRONMENT_STATE = "observation.environment_state"
OBSERVATION_STATE = "observation.state"
OBSERVATION_TACTILE = "observation.force"
OBSERVATION_LANG = "task"
ACTION = "action"
ACTION_TACTILE = "action.force"
LANGUAGE_ACTION_TARGET = "language_action_target"
LANGUAGE_ACTION_FRAME = "language_action_frame"

# Fixed max byte length for task (language instruction) tensor encoding.
# Stored as a per-sample (MAX_TASK_BYTES,) uint8 tensor, null-padded. 256 bytes
# covers all observed instruction strings in OXE/Agibot/XDOF datasets with
# headroom. Load-bearing: accelerate's dispatch_batches path requires all batch
# fields to be tensors so they can be broadcast/split across ranks.
MAX_TASK_BYTES = 256
