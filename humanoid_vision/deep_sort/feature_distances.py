from jaxtyping import Float, jaxtyped
from beartype import beartype
import numpy as np
from numpy import ndarray


@jaxtyped(typechecker=beartype)
def get_pose_distance(
    track_pose: Float[ndarray, "num_tracks pose_dim"],
    detect_pose: Float[ndarray, "num_detections pose_dim"],
    pose_distance: str = "smpl",
) -> Float[ndarray, "num_tracks num_detections"]:
    """Compute pair-wise squared L2 distances between track and detection poses.

    Args:
        track_pose: Pose embeddings for tracks (num_tracks, pose_dim)
        detect_pose: Pose embeddings for detections (num_detections, pose_dim)
        pose_distance: Distance type, "smpl" removes last 3 camera elements

    Returns:
        Distance matrix of shape (num_tracks, num_detections)
    """
    track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

    if pose_distance == "smpl":
        # remove additional dimension used for encoding location (last 3 elements)
        track_pose = track_pose[:, :-3]
        detect_pose = detect_pose[:, :-3]

    if len(track_pose) == 0 or len(detect_pose) == 0:
        return np.zeros((len(track_pose), len(detect_pose)))
    track_pose2, detect_pose2 = (
        np.square(track_pose).sum(axis=1),
        np.square(detect_pose).sum(axis=1),
    )
    r2 = (
        -2.0 * np.dot(track_pose, detect_pose.T)
        + track_pose2[:, None]
        + detect_pose2[None, :]
    )
    r2 = np.clip(r2, 0.0, float(np.inf))

    return r2
