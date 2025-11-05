"""Feature extraction stage of the PHALP pipeline.

This module handles extracting appearance, pose, and location features
from detected persons using HMR2 and HMAR models.
"""

from jaxtyping import Float, UInt8, jaxtyped
from beartype import beartype
import numpy as np
from numpy import ndarray
import torch

from humanoid_vision.common.detection import Detection
from humanoid_vision.common.hmar_output import HMAROutput
from humanoid_vision.models.hmar.hmr2 import HMR2023TextureSampler
from humanoid_vision.common.types import (
    AppearanceEmbed,
    BBoxes,
    Classes,
    Joints2DFlatWithRoot,
    Joints3DWithRoot,
    LocationEmbed,
    Masks,
    PoseEmbed,
    Scores,
)
from humanoid_vision.utils.bbox import get_cropped_image
from humanoid_vision.utils.utils import smpl_to_pose_camera_vector


@jaxtyped(typechecker=beartype)
def crop_and_preprocess_detections(
    image: UInt8[ndarray, "height width 3"],
    bbox: BBoxes,
    bbox_padded: BBoxes,
    masks: Masks,
) -> tuple[
    Float[torch.Tensor, "num_valid 4 256 256"],  # masked_images
    list[Float[ndarray, "2"]],  # centers
    list[Float[ndarray, "2"]],  # scales
    list,  # rles (RLE encoded masks)
]:
    """Crop and preprocess detected persons for HMR model input.

    Args:
        image: Original image in BGR format
        bbox: Bounding boxes (N, 4)
        bbox_padded: Padded bounding boxes (N, 4)
        masks: Segmentation masks (N, H, W)

    Returns:
        masked_images: Stacked RGBA crops, shape (num_valid, 4, 256, 256)
        centers: List of bbox centers
        scales: List of bbox scales
        rles: List of RLE encoded masks
    """
    masked_image_list = []
    center_list = []
    scale_list = []
    rles_list = []

    for i in range(len(bbox)):
        masked_image, _center, _scale, rles, center_pad, scale_pad = get_cropped_image(
            image, bbox[i], bbox_padded[i], masks[i]
        )
        masked_image_list.append(masked_image)
        center_list.append(center_pad)
        scale_list.append(scale_pad)
        rles_list.append(rles)

    if len(masked_image_list) == 0:
        return (
            torch.zeros((0, 4, 256, 256)),
            [],
            [],
            [],
        )

    masked_images = torch.stack(masked_image_list, dim=0)
    return masked_images, center_list, scale_list, rles_list


@jaxtyped(typechecker=beartype)
def extract_hmr_features(
    hmr_model: HMR2023TextureSampler,
    masked_images: Float[torch.Tensor, "batch 4 256 256"],
    centers: list[Float[ndarray, "2"]],
    scales: list[Float[ndarray, "2"]],
    img_offset: tuple[int, int],  # (left, top) padding
    render_res: int,
) -> tuple[
    HMAROutput,
    AppearanceEmbed,
    PoseEmbed,
    LocationEmbed,
    Joints3DWithRoot,
    Joints2DFlatWithRoot,
    Float[ndarray, "batch 3"],
]:
    """Extract appearance, pose, and location features using HMR model.

    Args:
        hmr_model: HMR2023TextureSampler model
        masked_images: Preprocessed RGBA crops (num_persons, 4, 256, 256)
        centers: Bounding box centers
        scales: Bounding box scales
        img_offset: Image padding offset (left, top)
        render_res: Rendering resolution for full image
        use_joints_for_pose: If True, use 3D joints for pose embedding (135-dim),
                            else use SMPL parameters (229-dim)

    Returns:
        hmar_out: Complete HMAROutput with all predictions
        appe_embedding: Appearance features (num_persons, 4096)
        pose_embedding: Pose features (num_persons, 135 or 229)
        loca_embedding: Location features (num_persons, 99)
        pred_joints_2d: Projected 2D joints (num_persons, 45, 2)
        pred_cam: Camera translation (num_persons, 3)
    """
    num_persons = masked_images.shape[0]

    if num_persons == 0:
        return (
            None,
            np.zeros((0, 4096)),
            np.zeros((0, 229)),
            np.zeros((0, 99)),
            np.zeros((0, 45, 3)),
            np.zeros((0, 45 * 2)),
            np.zeros((0, 3)),
        )

    with torch.no_grad():
        # Run HMR forward pass
        hmar_out: HMAROutput = hmr_model(masked_images.cuda())

        # Extract appearance embedding from UV texture
        uv_vector = hmar_out.uv_vector
        appe_embedding = hmr_model.hmar.autoencoder_hmar(uv_vector, en=True)
        appe_embedding = appe_embedding.view(appe_embedding.shape[0], -1)

        # Get 3D joints from SMPL
        pred_joints = hmr_model.smpl(hmar_out).joints

        # Compute camera parameters and project to 2D
        left, top = img_offset
        ratio = 1.0 / max(centers[0]) * render_res if len(centers) > 0 else 1.0

        centers_array = np.array(centers)
        scales_array = np.array(scales)

        pred_joints_2d, pred_joints, pred_cam = hmr_model.hmar.get_3d_parameters(
            pred_joints,
            hmar_out.pred_cam,
            center=(centers_array + np.array([left, top])) * ratio,
            img_size=render_res,
            scale=np.max(scales_array, axis=1, keepdims=True) * ratio,
        )

        # Create pose embedding
        # Use SMPL parameters (9 + 207 + 10 + 3 = 229 dim)
        pose_embedding_list = []
        for i in range(num_persons):
            pred_smpl_params = {
                "body_pose": hmar_out.body_pose[i].cpu().numpy(),
                "betas": hmar_out.betas[i].cpu().numpy(),
                "global_orient": hmar_out.global_orient[i].cpu().numpy(),
            }
            pose_vec = smpl_to_pose_camera_vector(pred_smpl_params, pred_cam[i])
            pose_embedding_list.append(torch.from_numpy(pose_vec[0]))
        pose_embedding = torch.stack(pose_embedding_list, dim=0).numpy()

        # Create location embedding
        pred_joints_2d_normalized = (
            (pred_joints_2d.reshape(num_persons, -1) / render_res).cpu().numpy()
        )
        pred_cam_array = pred_cam.view(num_persons, -1).cpu().numpy()

        # Location embedding: [2D joints (90), camera repeated 3 times (9)]
        loca_embedding = np.concatenate(
            (pred_joints_2d_normalized, pred_cam_array, pred_cam_array, pred_cam_array),
            axis=1,
        )

    return (
        hmar_out,
        appe_embedding.cpu().numpy(),
        pose_embedding,
        loca_embedding,
        pred_joints.cpu().numpy(),
        pred_joints_2d_normalized,
        pred_cam.cpu().numpy(),
    )


@jaxtyped(typechecker=beartype)
def create_detection_data_list(
    hmar_out: HMAROutput,
    appe_embedding: AppearanceEmbed,
    pose_embedding: PoseEmbed,
    loca_embedding: LocationEmbed,
    bbox: BBoxes,
    masks: list,  # RLE encoded
    scores: Scores,
    classes: Classes,
    centers: list,
    scales: list,
    pred_joints: Joints3DWithRoot,
    pred_joints_2d: Joints2DFlatWithRoot,
    pred_cam: Float[ndarray, "batch 3"],
    frame_path,
    frame_time: int,
    img_size: tuple[int, int],
    ground_truth_ids: list,
    annotations: list,
) -> list[Detection]:
    """Create Detection objects from extracted features.

    Args:
        hmar_out: HMAROutput from HMR model
        appe_embedding: Appearance embeddings (num_persons, 4096)
        pose_embedding: Pose embeddings (num_persons, 135 or 229)
        loca_embedding: Location embeddings (num_persons, 99)
        bbox: Bounding boxes (num_persons, 4) in (x, y, w, h) format
        masks: RLE encoded masks
        scores: Detection scores
        classes: Class IDs
        centers: Bounding box centers
        scales: Bounding box scales
        pred_joints_2d: 2D joint projections (num_persons, 45, 2)
        pred_cam: Camera translations (num_persons, 3)
        frame_path: Path to current frame
        frame_time: Frame timestamp/index
        img_size: Image dimensions (height, width)
        ground_truth_ids: Ground truth track IDs (for evaluation)
        annotations: Ground truth annotations (for evaluation)

    Returns:
        List of Detection objects, one per detected person
    """
    num_persons = len(bbox)
    detection_list = []

    img_height, img_width = img_size

    for i in range(num_persons):
        # Prepare SMPL parameters as dictionary
        pred_smpl_params = {
            "body_pose": hmar_out.body_pose[i].cpu().numpy(),
            "betas": hmar_out.betas[i].cpu().numpy(),
            "global_orient": hmar_out.global_orient[i].cpu().numpy(),
        }

        uv_map = hmar_out.uv_vector[i].cpu().numpy()
        camera_flat = pred_cam[i]
        camera_bbox = hmar_out.pred_cam[i].cpu().numpy()
        joints_3d = pred_joints[i]
        joints_2d = pred_joints_2d[i]

        hmar_out_cam_flat = hmar_out.pred_cam[i].cpu().numpy()
        hmar_out_cam_t_flat = hmar_out.pred_cam_t[i].cpu().numpy()
        hmar_out_focal_flat = hmar_out.focal_length[i].cpu().numpy()

        full_embedding = np.concatenate(
            [appe_embedding[i], pose_embedding[i], loca_embedding[i]]
        )

        detection_list.append(
            Detection(
                bbox=bbox[i],
                mask=masks[i],
                conf=float(scores[i]),
                appe=appe_embedding[i],
                pose=pose_embedding[i],
                loca=loca_embedding[i],
                uv=uv_map,
                embedding=full_embedding,
                center=centers[i],
                scale=scales[i],
                smpl=pred_smpl_params,
                camera=camera_flat,
                camera_bbox=camera_bbox,
                joints_3d=joints_3d,
                joints_2d=joints_2d,
                size=[img_height, img_width],
                img_path=frame_path,
                img_name=(
                    frame_path.split("/")[-1] if isinstance(frame_path, str) else None
                ),
                class_name=int(classes[i]),
                time=frame_time,
                ground_truth=ground_truth_ids[i] if i < len(ground_truth_ids) else None,
                annotations=annotations[i] if i < len(annotations) else [],
                hmar_out_cam=hmar_out_cam_flat,
                hmar_out_cam_t=hmar_out_cam_t_flat,
                hmar_out_focal_length=hmar_out_focal_flat,
            )
        )

    return detection_list
