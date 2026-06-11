## Adapted from lerobot/configs/types.py
## Moving it into a new file so we can begin modifying these

from dataclasses import dataclass
from enum import Enum


class ActionType(str, Enum):
    POSITION = "POSITION"  # format: (x, y, z) or joint positions
    EE_QUAT_POS_XYZW = "EE_QUAT_POS_XYZW"  # format: (x, y, z, qx, qy, qz, qw)
    EE_QUAT_POS_WXYZ = "EE_QUAT_POS_WXYZ"  # format: (x, y, z, qw, qx, qy, qz)
    EE_EULER_POS = "EE_EULER_POS"  # format: (x, y, z, roll, pitch, yaw)
    EE_6D_POS = "EE_6D_POS"  # format: (x, y, z, 6D representation)
    SIX_D = "SIX_D"  # format: (6D representation) (no position or gripper)
    QUAT_XYZW = "QUAT_XYZW"  # format: (qx, qy, qz, qw) (no position or gripper)
    QUAT_WXYZ = "QUAT_WXYZ"  # format: (qw, qx, qy, qz) (no position or gripper)
    EULER = "EULER"  # format: (roll, pitch, yaw) (no position or gripper)


class FeatureType(str, Enum):
    STATE = "STATE"
    TACTILE = "TACTILE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    REWARD = "REWARD"


class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"
    QUANTILE = "QUANTILE"
    ACTIONCHUNK_PERDIM_MEAN_STD = "ACTIONCHUNK_PERDIM_MEAN_STD"
    ACTIONCHUNK_PERDIM_MIN_MAX = "ACTIONCHUNK_PERDIM_MIN_MAX"
    ACTIONCHUNK_PERDIM_QUANTILE = "ACTIONCHUNK_PERDIM_QUANTILE"
    ACTIONCHUNK_MIN_MAX = "ACTIONCHUNK_MIN_MAX"
    ACTIONCHUNK_MEAN_STD = "ACTIONCHUNK_MEAN_STD"
    ACTIONCHUNK_QUANTILE = "ACTIONCHUNK_QUANTILE"


class TrainingMode(str, Enum):
    ROBOT_FLOWMATCH = "ROBOT_FLOWMATCH"
    ROBOT_AUTOREGRESSIVE = "ROBOT_AUTOREGRESSIVE"
    ROBOT_KNOWLEDGE_INSULATION = "ROBOT_KNOWLEDGE_INSULATION"
    VQA = "VQA"
    BOUNDING_BOX = "BOUNDING_BOX"
    POINTING = "POINTING"


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple[int, ...]
