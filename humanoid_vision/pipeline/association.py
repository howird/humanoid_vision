"""Data association utilities for the PHALP pipeline.

This module provides helper functions for track-detection association.
"""

from jaxtyping import jaxtyped
from beartype import beartype

from humanoid_vision.common.detection import Detection
from humanoid_vision.common.track import Track
from humanoid_vision.common.types import CostMatrix, Matches


@jaxtyped(typechecker=beartype)
def associate_detections_to_tracks(
    tracks: list[Track],
    detections: list[Detection],
    distance_metric,
    max_distance: float,
    confirmed_track_indices: list[int],
) -> tuple[Matches, list[int], list[int], CostMatrix]:
    """Associate detections to existing tracks using a distance metric.

    Args:
        tracks: List of active Track objects
        detections: List of Detection objects from current frame
        distance_metric: Distance metric object with a distance() method
        max_distance: Maximum distance threshold for valid associations
        confirmed_track_indices: Indices of confirmed/tentative tracks to consider

    Returns:
        matches: List of (track_idx, detection_idx) pairs
        unmatched_tracks: List of track indices without matches
        unmatched_detections: List of detection indices without matches
        cost_matrix: Full cost matrix (num_confirmed_tracks, num_detections)
    """
    from humanoid_vision.deep_sort.linear_assignment import matching_simple

    matches, unmatched_tracks, unmatched_detections, cost_matrix = matching_simple(
        distance_fn=distance_metric.distance,
        max_distance=max_distance,
        cascade_depth=30,  # max_age, not actually used in simple matching
        tracks=tracks,
        detections=detections,
        track_indices=confirmed_track_indices,
    )

    return matches, unmatched_tracks, unmatched_detections, cost_matrix


@jaxtyped(typechecker=beartype)
def update_matched_tracks(
    tracks: list[Track],
    detections: list[Detection],
    matches: Matches,
    shot_change: bool,
) -> None:
    """Update tracks that were matched with detections.

    Args:
        tracks: List of Track objects
        detections: List of Detection objects
        matches: List of (track_idx, detection_idx) pairs
        shot_change: Whether a shot change was detected this frame
    """
    for track_idx, detection_idx in matches:
        tracks[track_idx].update(
            detections[detection_idx],
            detection_idx,
            shot=1 if shot_change else 0,
        )


@jaxtyped(typechecker=beartype)
def mark_unmatched_tracks(
    tracks: list[Track],
    unmatched_track_indices: list[int],
) -> None:
    """Mark tracks without matches as missed.

    Args:
        tracks: List of Track objects
        unmatched_track_indices: Indices of tracks without matches
    """
    for track_idx in unmatched_track_indices:
        tracks[track_idx].mark_missed()


@jaxtyped(typechecker=beartype)
def initiate_new_tracks(
    tracks: list[Track],
    detections: list[Detection],
    unmatched_detection_indices: list[int],
    cfg,
    next_track_id: int,
    n_init: int,
    max_age: int,
    dims: tuple[int, int, int],
) -> int:
    """Create new tracks from unmatched detections.

    Args:
        tracks: List of Track objects to append to
        detections: List of Detection objects
        unmatched_detection_indices: Indices of detections to create tracks from
        cfg: Configuration object
        next_track_id: Next available track ID
        n_init: Number of frames before track is confirmed
        max_age: Maximum frames a track can be missed before deletion
        dims: Feature dimensions (appe_dim, pose_dim, loca_dim)

    Returns:
        Updated next_track_id value
    """
    for detection_idx in unmatched_detection_indices:
        new_track = Track(
            cfg=cfg,
            track_id=next_track_id,
            n_init=n_init,
            max_age=max_age,
            detection_data=detections[detection_idx].as_legacy_dict(),
            detection_id=detection_idx,
            dims=dims,
        )
        new_track.add_predicted()
        tracks.append(new_track)
        next_track_id += 1

    return next_track_id


@jaxtyped(typechecker=beartype)
def remove_deleted_tracks(tracks: list[Track]) -> list[Track]:
    """Remove tracks marked as deleted.

    Args:
        tracks: List of Track objects

    Returns:
        Filtered list with deleted tracks removed
    """
    return [t for t in tracks if not t.is_deleted()]
