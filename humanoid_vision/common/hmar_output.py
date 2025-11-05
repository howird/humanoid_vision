from dataclasses import dataclass
from jaxtyping import jaxtyped
from beartype import beartype
from torch import Tensor

from humanoid_vision.common.smpl_output import HMRSMPLOutput
from humanoid_vision.common.types import (
    CameraTranslation,
    FocalLength,
    Joints2D,
    Joints3D,
    UVImage,
    Vertices,
    WeakPerspCamera,
)


@jaxtyped(typechecker=beartype)
@dataclass
class HMAROutput(HMRSMPLOutput):
    """Output of the HMAR2 model.

    Inherits from HMROutput and adds additional fields for appearance.

    Attributes:
        pred_cam: Camera parameters of shape (B, 3)
        pred_cam_t: Camera translation of shape (B, 3)
        focal_length: Focal length of shape (B, 2)
        pred_keypoints_3d: 3D keypoints of shape (B, N, 3)
        pred_keypoints_2d: 2D keypoints of shape (B, N, 2)
        pred_vertices: Mesh vertices of shape (B, V, 3)
        uv_image: RGBA texture map in UV space (B, 4, 256, 256)
        uv_vector: Processed UV texture for autoencoder (B, 4, 256, 256)
        losses: Optional dictionary of losses
    """

    pred_cam: WeakPerspCamera
    pred_cam_t: CameraTranslation
    focal_length: FocalLength
    pred_keypoints_3d: Joints3D
    pred_keypoints_2d: Joints2D
    pred_vertices: Vertices
    uv_image: UVImage
    uv_vector: UVImage
    losses: dict[str, Tensor] | None = None
