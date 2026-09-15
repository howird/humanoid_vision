from pathlib import Path

import cv2
import numpy as np
import torchvision
from scenedetect import AdaptiveDetector, detect

from humanoid_vision.utils.pylogger_phalp import get_pylogger

log = get_pylogger(__name__)


def get_list_of_shots(
    video_name: str,
    list_of_frames: list,
    output_dir: Path,
) -> list[int]:
    """Detect shot boundaries in a video using PySceneDetect.

    See https://github.com/Breakthrough/PySceneDetect.

    Args:
        video_name: Name of the video, used for the temporary re-encoded clip.
        list_of_frames: Either paths to extracted frames, or (video_path, pts)
            tuples pointing into a single video file.
        output_dir: Where the temporary clip is written when frames must be
            re-encoded.

    Returns: Frame indices at which a shot change occurs.
    """
    remove_tmp_video = False

    first = list_of_frames[0]
    if isinstance(first, tuple):
        # Frames index into an existing video file, detect on it directly.
        video_tmp_name = Path(first[0])
    elif isinstance(first, (str, Path)):
        # Frames were extracted to disk, re-encode them into a temporary video.
        tmp_dir = output_dir / "_TMP"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        video_tmp_name = tmp_dir / f"{video_name}.mp4"

        video_file = None
        for ft_, fname_ in enumerate(list_of_frames):
            im_ = cv2.imread(str(fname_))
            if ft_ == 0:
                video_file = cv2.VideoWriter(
                    str(video_tmp_name),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    24,
                    frameSize=(im_.shape[1], im_.shape[0]),
                )
            video_file.write(im_)
        video_file.release()
        remove_tmp_video = True
    else:
        raise TypeError(f"Unknown type of list_of_frames: {type(first)}")

    scene_list = detect(str(video_tmp_name), AdaptiveDetector())

    if remove_tmp_video:
        video_tmp_name.unlink()

    boundaries = []
    for scene in scene_list:
        boundaries.append(scene[0].get_frames())
        boundaries.append(scene[1].get_frames())

    # Drop the first and last boundaries: they are the video's own endpoints,
    # not shot changes.
    list_of_shots = [int(f) for f in np.unique(boundaries)[1:-1]]
    log.info(f"Detected shot change at frames: {list_of_shots}.")

    return list_of_shots


def read_frame(frame_path):
    frame = None
    # frame path can be either a path to an image or a list of [video_path, frame_id in pts]
    if isinstance(frame_path, tuple):
        frame = torchvision.io.read_video(
            frame_path[0],
            pts_unit="pts",
            start_pts=frame_path[1],
            end_pts=frame_path[1] + 1,
        )[0][0]
        frame = frame.numpy()[:, :, ::-1]
    elif isinstance(frame_path, Path):
        frame = cv2.imread(str(frame_path))
    else:
        raise Exception("Invalid frame path")

    return frame


def read_from_video_pts(video_path, frame_pts):
    frame = torchvision.io.read_video(
        video_path, pts_unit="pts", start_pts=frame_pts, end_pts=frame_pts + 1
    )[0][0]
    frame = frame.numpy()[:, :, ::-1]
    return frame
