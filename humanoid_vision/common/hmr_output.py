from dataclasses import dataclass
from torch import Tensor
from beartype import beartype
from jaxtyping import jaxtyped

from humanoid_vision.common.smpl_output import HMRSMPLOutput
from humanoid_vision.common.types import (
    CameraTranslation,
    FocalLength,
    Joints2D,
    Joints3D,
    Vertices,
    WeakPerspCamera,
)


@jaxtyped(typechecker=beartype)
@dataclass
class HMROutput(HMRSMPLOutput):
    """Output of the HMR2 model.

    Inherits from HMRSMPLOutput and adds additional fields for camera parameters,
    keypoints, and vertices.

    Attributes:
        pred_cam: Camera parameters of shape (B, 3)
        pred_cam_t: Camera translation of shape (B, 3)
        focal_length: Focal length of shape (B, 2)
        pred_keypoints_3d: 3D keypoints of shape (B, N, 3)
        pred_keypoints_2d: 2D keypoints of shape (B, N, 2)
        pred_vertices: Mesh vertices of shape (B, V, 3)
        losses: Optional dictionary of losses
    """

    pred_cam: WeakPerspCamera
    pred_cam_t: CameraTranslation
    focal_length: FocalLength
    pred_keypoints_3d: Joints3D
    pred_keypoints_2d: Joints2D
    pred_vertices: Vertices
    losses: dict[str, Tensor] | None = None
