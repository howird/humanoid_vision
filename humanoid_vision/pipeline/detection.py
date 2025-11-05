"""Detection stage of the PHALP pipeline.

This module handles person detection using Detectron2 and bounding box preprocessing.
"""

from jaxtyping import Float, UInt8, jaxtyped
from beartype import beartype
import numpy as np
from numpy import ndarray

from humanoid_vision.datasets.utils import expand_bbox_to_aspect_ratio
from humanoid_vision.common.types import BBoxes, Masks, Scores, Classes


@jaxtyped(typechecker=beartype)
def run_detection(
    detector,
    image: UInt8[ndarray, "height width 3"],
    confidence_threshold: float = 0.8,
    expand_bbox_shape: tuple[int, int] | None = None,
) -> tuple[
    BBoxes,  # pred_bbox
    BBoxes,  # pred_bbox_padded
    Masks,  # pred_masks
    Scores,  # pred_scores
    Classes,  # pred_classes
]:
    """Run person detection on an image frame.

    Args:
        detector: Detectron2 predictor instance
        image: Input image in BGR format (H, W, 3)
        confidence_threshold: Minimum detection confidence to keep
        expand_bbox_shape: Target aspect ratio (H, W) for bbox expansion, or None

    Returns:
        pred_bbox: Bounding boxes in (x1, y1, x2, y2) format
        pred_bbox_padded: Expanded bounding boxes for cropping
        pred_masks: Binary segmentation masks
        pred_scores: Detection confidence scores
        pred_classes: Class IDs (all 0 for person)
    """
    outputs = detector(image)
    instances = outputs["instances"]

    # Filter for person class (class 0)
    instances = instances[instances.pred_classes == 0]

    # Filter by confidence threshold
    instances = instances[instances.scores > confidence_threshold]

    # Extract detection components
    pred_bbox = instances.pred_boxes.tensor.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()
    pred_classes = instances.pred_classes.cpu().numpy()

    # Pad bounding boxes to target aspect ratio if specified
    if expand_bbox_shape is not None:
        pred_bbox_padded = expand_bbox_to_aspect_ratio(pred_bbox, expand_bbox_shape)
    else:
        pred_bbox_padded = pred_bbox.copy()

    return (
        pred_bbox,
        pred_bbox_padded,
        pred_masks,
        pred_scores,
        pred_classes,
    )


@jaxtyped(typechecker=beartype)
def filter_small_detections(
    bbox: BBoxes,
    bbox_padded: BBoxes,
    masks: Masks,
    scores: Scores,
    classes: Classes,
    min_width: int,
    min_height: int,
) -> tuple[list[int], BBoxes, BBoxes, Masks, Scores, Classes]:
    """Filter out detections with bounding boxes that are too small.

    Args:
        bbox: Original bounding boxes
        bbox_padded: Padded bounding boxes
        masks: Segmentation masks
        scores: Detection scores
        classes: Class IDs
        min_width: Minimum bbox width in pixels
        min_height: Minimum bbox height in pixels

    Returns:
        selected_indices: Indices of kept detections
        bbox: Filtered bounding boxes
        bbox_padded: Filtered padded bounding boxes
        masks: Filtered masks
        scores: Filtered scores
        classes: Filtered class IDs
    """
    selected_indices = []

    for i in range(len(bbox)):
        width = bbox[i, 2] - bbox[i, 0]
        height = bbox[i, 3] - bbox[i, 1]

        if width >= min_width and height >= min_height:
            selected_indices.append(i)

    if len(selected_indices) == 0:
        # Return empty arrays with correct shapes
        return (
            selected_indices,
            np.zeros((0, 4), dtype=bbox.dtype),
            np.zeros((0, 4), dtype=bbox_padded.dtype),
            np.zeros((0, masks.shape[1], masks.shape[2]), dtype=masks.dtype),
            np.zeros((0,), dtype=scores.dtype),
            np.zeros((0,), dtype=classes.dtype),
        )

    selected_indices = np.array(selected_indices)
    return (
        selected_indices.tolist(),
        bbox[selected_indices],
        bbox_padded[selected_indices],
        masks[selected_indices],
        scores[selected_indices],
        classes[selected_indices],
    )
