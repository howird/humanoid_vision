"""
Modified code from https://github.com/nwojke/deep_sort
"""

from __future__ import absolute_import
from typing import Callable

from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
import numpy as np
from numpy import ndarray

from humanoid_vision.common.detection import Detection
from humanoid_vision.common.track import Track
from humanoid_vision.common.types import CostMatrix, Matches


INFTY_COST = 1e5


@jaxtyped(typechecker=beartype)
def linear_assignment(
    cost_matrix: Float[ndarray, "num_rows num_cols"],
) -> Int[ndarray, "num_matches 2"]:
    # TODO(howird): benchmark the following
    # try:
    #     from sklearn.utils.linear_assignment_ import linear_assignment
    #
    #     return linear_assignment(cost_matrix)
    # except ImportError:
    #     try:
    #         import lap
    #
    #         _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    #         return np.array([[y[i], i] for i in x if i >= 0])
    #     except ImportError:
    from scipy.optimize import linear_sum_assignment

    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))


@jaxtyped(typechecker=beartype)
def gated_metric(
    distance_fn: Callable,
    tracks: list[Track],
    dets: list[Detection],
    track_indices: list[int],
    detection_indices: list[int],
) -> CostMatrix:
    appe_emb = np.array([dets[i].appe for i in detection_indices])
    loca_emb = np.array([dets[i].loca for i in detection_indices])
    pose_emb = np.array([dets[i].pose for i in detection_indices])
    uv_maps = np.array([dets[i].uv for i in detection_indices])
    targets = np.array([tracks[i].track_id for i in track_indices])

    return distance_fn(
        [appe_emb, loca_emb, pose_emb, uv_maps],
        targets,
    )


@jaxtyped(typechecker=beartype)
def min_cost_matching(
    distance_fn: Callable,
    max_distance: float,
    tracks: list[Track],
    detections: list[Detection],
    track_indices: list[int],
    detection_indices: list[int],
) -> tuple[
    Matches,
    list[int],
    list[int],
    CostMatrix,
]:
    if len(detection_indices) == 0 or len(track_indices) == 0:
        return (
            [],
            track_indices,
            detection_indices,
            np.zeros((len(track_indices), len(detection_indices)), dtype=float),
        )

    cost_matrix_a = gated_metric(
        distance_fn, tracks, detections, track_indices, detection_indices
    )
    cost_matrix = cost_matrix_a

    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # cost_matrix = np.log(cost_matrix)
    # max_distance = np.log(max_distance)

    indices = linear_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)

    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections, cost_matrix


@jaxtyped(typechecker=beartype)
def matching_simple(
    distance_fn: Callable,
    max_distance: float,
    cascade_depth: int,
    tracks: list[Track],
    detections: list[Detection],
    track_indices: list[int] | None = None,
    detection_indices: list[int] | None = None,
) -> tuple[
    Matches,
    list[int],
    list[int],
    CostMatrix,
]:
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices

    matches, _, unmatched_detections, cost = min_cost_matching(
        distance_fn,
        max_distance,
        tracks,
        detections,
        track_indices,
        unmatched_detections,
    )

    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections, cost
