"""Type definitions and aliases for the PHALP pipeline.

This module provides jaxtyping aliases and common type definitions
used throughout the pipeline stages.
"""

from typing import Final, TypeAlias

from torch import Tensor
from jaxtyping import Bool, Float, Int
from numpy import ndarray

# ---------------------------------------------------------------------------
# Model-wide dimension constants
# ---------------------------------------------------------------------------

NUM_JOINTS: Final[int] = 44
NUM_JOINTS_WITH_ROOT: Final[int] = 45
BODY_POSE_JOINTS: Final[int] = 23
BETAS_DIM: Final[int] = 10
VERTEX_COUNT: Final[int] = 6890

UV_CHANNELS: Final[int] = 4
UV_RES: Final[int] = 256

CAMERA_DIM: Final[int] = 3  # (s, tx, ty) weak-persp parameters
CAMERA_T_DIM: Final[int] = 3  # camera translation
CAMERA_FOCAL_DIM: Final[int] = 2  # (fx, fy)
CAMERA_BBOX_DIM: Final[int] = 3  # raw HMAR predictor output

APPE_EMBED_DIM: Final[int] = 4096
POSE_EMBED_DIM: Final[int] = 229
LOCA_EMBED_DIM: Final[int] = 99
EMBED_DIM: Final[int] = APPE_EMBED_DIM + POSE_EMBED_DIM + LOCA_EMBED_DIM

# ---------------------------------------------------------------------------
# Tensor type aliases
# ---------------------------------------------------------------------------

Array: TypeAlias = ndarray | Tensor

# Image types
RGBImage: TypeAlias = Float[Tensor, "batch 3 height width"]
RGBAImage: TypeAlias = Float[Tensor, f"batch {UV_CHANNELS} height width"]
Mask: TypeAlias = Float[Tensor, "batch height width"]

# Detection types
BBoxes: TypeAlias = Float[ndarray, "num_detections 4"]  # (x1, y1, x2, y2)
BBoxArray: TypeAlias = Float[ndarray, "*batch 4"]
BBox: TypeAlias = Float[ndarray, "4"]
MaskArray: TypeAlias = Bool[ndarray, "height width"]
Masks: TypeAlias = Bool[ndarray, "num_detections height width"]
Scores: TypeAlias = Float[ndarray, "num_detections"]
Classes: TypeAlias = Int[ndarray, "num_detections"]

# Embedding types (batched)
AppearanceEmbed: TypeAlias = Float[ndarray, f"*batch {APPE_EMBED_DIM}"]
PoseEmbed: TypeAlias = Float[ndarray, f"*batch {POSE_EMBED_DIM}"]
LocationEmbed: TypeAlias = Float[ndarray, f"*batch {LOCA_EMBED_DIM}"]
TrackingEmbed: TypeAlias = Float[ndarray, f"*batch {EMBED_DIM}"]

# SMPL types
GlobalOrient: TypeAlias = Float[Tensor, "batch 1 3 3"]
BodyPose: TypeAlias = Float[Tensor, f"batch {BODY_POSE_JOINTS} 3 3"]
Betas: TypeAlias = Float[Tensor, f"batch {BETAS_DIM}"]
Vertices: TypeAlias = Float[Tensor, f"batch {VERTEX_COUNT} 3"]
Joints3D: TypeAlias = Float[Tensor, f"*batch {NUM_JOINTS} 3"]
Joints2D: TypeAlias = Float[Tensor, f"*batch {NUM_JOINTS} 2"]
Joints2DFlatWithRoot: TypeAlias = Float[ndarray, f"*batch 2*{NUM_JOINTS_WITH_ROOT}"]
Joints3DWithRoot: TypeAlias = Float[ndarray, f"*batch {NUM_JOINTS_WITH_ROOT} 3"]
Joints2DWithRoot: TypeAlias = Float[ndarray, f"*batch {NUM_JOINTS_WITH_ROOT} 2"]

# Camera types
CameraBBox: TypeAlias = Float[Array, f"*batch {CAMERA_BBOX_DIM}"]
WeakPerspCamera: TypeAlias = Float[Array, f"*batch {CAMERA_DIM}"]
CameraTranslation: TypeAlias = Float[Array, f"*batch {CAMERA_T_DIM}"]
FocalLength: TypeAlias = Float[Array, f"*batch {CAMERA_FOCAL_DIM}"]


# UV maps
UVMap: TypeAlias = Float[ndarray, f"{UV_CHANNELS} {UV_RES} {UV_RES}"]
UVImage: TypeAlias = Float[Tensor, f"batch {UV_CHANNELS} {UV_RES} {UV_RES}"]

# Tracking types
TrackID: TypeAlias = int
CostMatrix: TypeAlias = Float[ndarray, "num_tracks num_detections"]
Matches: TypeAlias = list[tuple[int, int]]  # List of (track_idx, detection_idx)
