"""
HMR Tracking Script v2 - Uses the new modular pipeline architecture.

This script demonstrates how to use the refactored pipeline modules directly
for human tracking, providing the same functionality as hmr_track.py but with
a modular, composable approach.
"""

import warnings

from torch import nn

warnings.filterwarnings("ignore")

from dataclasses import dataclass
from pathlib import Path

import tyro
import cv2
import joblib
import numpy as np

from humanoid_vision.utils.video_writer import VideoWriter
from humanoid_vision.visualize.visualizer import Visualizer
from humanoid_vision.configs.base import PhalpConfig
from humanoid_vision.utils.video_io_manager import VideoIOManager
from humanoid_vision.models.hmar.hmr2 import HMR2023TextureSampler
from humanoid_vision.trackers import setup_predictor, setup_detectron2
from humanoid_vision.utils.pylogger_phalp import get_pylogger

# Import new pipeline modules
from humanoid_vision.pipeline import detection, feature_extraction, association
from humanoid_vision.deep_sort.tracker import Tracker

log = get_pylogger(__name__)


@dataclass
class Human4DTrackingConfig(PhalpConfig):
    """Configuration for Human4D tracking."""

    video_source: tyro.conf.Positional[str] = tyro.MISSING


def track_video_modular(
    cfg: PhalpConfig,
    video_name: str,
    list_of_frames: list[Path],
    hmr_model: HMR2023TextureSampler,
    pose_predictor,
    detector,
) -> dict:
    """Track humans in video using modular pipeline stages.

    This function demonstrates the refactored pipeline architecture by
    explicitly calling each stage:
    1. Detection
    2. Feature Extraction
    3. Track Prediction
    4. Data Association
    5. Track Update
    6. Future Prediction
    7. Result Recording

    Args:
        cfg: Configuration object
        video_name: Name of the video being processed
        list_of_frames: List of frame file paths
        hmr_model: HMR2023TextureSampler model for pose/appearance
        pose_predictor: PoseTransformerV2 for temporal prediction
        detector: Detectron2 detector for person detection

    Returns:
        Dictionary mapping frame names to tracking results
    """
    # Initialize tracker
    tracker = Tracker(
        cfg=cfg,
        hmar=hmr_model.hmar,
        pose_predictor=pose_predictor,
        max_age=cfg.phalp.max_age_track,
        n_init=cfg.phalp.n_init,
        dims=(4096, 4096, 99),  # (appe_dim, pose_dim, loca_dim)
    )

    # Storage for results
    final_visuals_dic = {}
    tracked_frames = []

    # Define keys for result storage (same as original PHALP)
    eval_keys = ["tracked_ids", "tracked_bbox", "tid", "bbox", "tracked_time"]
    history_keys = (
        (["appe", "loca", "pose", "uv"] if cfg.render.enable else [])
        + [
            "center",
            "scale",
            "size",
            "img_path",
            "img_name",
            "class_name",
            "conf",
            "annotations",
        ]
        + ["smpl", "camera", "camera_bbox", "3d_joints", "2d_joints", "mask"]
    )
    prediction_keys = (
        ["prediction_uv", "prediction_pose", "prediction_loca"]
        if cfg.render.enable
        else []
    )
    visual_store_ = eval_keys + history_keys + prediction_keys
    tmp_keys = ["uv", "prediction_uv", "prediction_pose", "prediction_loca"]

    # Process start/end frames
    if cfg.phalp.start_frame != -1:
        list_of_frames = list_of_frames[cfg.phalp.start_frame : cfg.phalp.end_frame]

    log.info(f"Processing {len(list_of_frames)} frames from {video_name}")

    # Main tracking loop
    for t, frame_path in enumerate(list_of_frames):
        frame_name = frame_path.name
        image_frame = cv2.imread(str(frame_path))
        img_height, img_width, _ = image_frame.shape
        new_image_size = max(img_height, img_width)
        top, left = (
            (new_image_size - img_height) // 2,
            (new_image_size - img_width) // 2,
        )
        # TODO(howird): add shot change detection
        # shot_change = 1 if t in list_of_shots else 0
        shot_change = 0

        log.info(f"[{t + 1}/{len(list_of_frames)}] Processing {frame_name}")

        # ============================================================
        # STAGE 1: DETECTION
        # ============================================================
        log.debug("Stage 1: Running person detection")
        bbox, bbox_padded, masks, scores, classes = detection.run_detection(
            detector=detector,
            image=image_frame,
            confidence_threshold=cfg.phalp.low_th_c,
            expand_bbox_shape=cfg.expand_bbox_shape,
        )

        # Filter small detections
        (
            selected_indices,
            bbox,
            bbox_padded,
            masks,
            scores,
            classes,
        ) = detection.filter_small_detections(
            bbox=bbox,
            bbox_padded=bbox_padded,
            masks=masks,
            scores=scores,
            classes=classes,
            min_width=cfg.phalp.small_w,
            min_height=cfg.phalp.small_h,
        )

        if len(bbox) == 0:
            log.warning(f"No valid detections in {frame_name}")
            # Still need to age tracks and record results
            tracker.predict()
            _record_frame_results(
                final_visuals_dic,
                tracked_frames,
                tracker,
                frame_name,
                frame_path,
                t,
                shot_change,
                visual_store_,
                history_keys,
                prediction_keys,
                cfg,
            )
            continue

        log.debug(f"  Found {len(bbox)} valid detections")

        # ============================================================
        # STAGE 2: FEATURE EXTRACTION
        # ============================================================
        log.debug("Stage 2: Extracting features from detections")

        # Preprocess detections
        masked_images, centers, scales, rles = (
            feature_extraction.crop_and_preprocess_detections(
                image=image_frame,
                bbox=bbox,
                bbox_padded=bbox_padded,
                masks=masks,
            )
        )

        # Extract HMR features
        hmar_out, appe_emb, pose_emb, loca_emb, joints, joints_2d, cam = (
            feature_extraction.extract_hmr_features(
                hmr_model=hmr_model,
                masked_images=masked_images,
                centers=centers,
                scales=scales,
                img_offset=(left, top),
                render_res=cfg.render.res,
            )
        )

        # Create Detection objects
        detections = feature_extraction.create_detection_data_list(
            hmar_out=hmar_out,
            appe_embedding=appe_emb,
            pose_embedding=pose_emb,
            loca_embedding=loca_emb,
            bbox=np.array(
                [
                    [
                        bbox[i, 0],
                        bbox[i, 1],
                        bbox[i, 2] - bbox[i, 0],
                        bbox[i, 3] - bbox[i, 1],
                    ]
                    for i in range(len(bbox))
                ]
            ),  # Convert to (x, y, w, h)
            masks=rles,
            scores=scores,
            classes=classes,
            centers=centers,
            scales=scales,
            pred_joints=joints,
            pred_joints_2d=joints_2d,
            pred_cam=cam,
            frame_path=frame_path,
            frame_time=t,
            img_size=(img_height, img_width),
            ground_truth_ids=[1] * len(bbox),
            annotations=[[]] * len(bbox),
        )

        log.debug(f"  Extracted features for {len(detections)} detections")

        # ============================================================
        # STAGE 3: TRACK PREDICTION
        # ============================================================
        log.debug("Stage 3: Predicting track states")
        tracker.predict()

        # ============================================================
        # STAGE 4: DATA ASSOCIATION
        # ============================================================
        log.debug("Stage 4: Associating detections to tracks")
        confirmed_indices = [
            i
            for i, t in enumerate(tracker.tracks)
            if t.is_confirmed() or t.is_tentative()
        ]

        matches, unmatched_tracks, unmatched_detections, cost_matrix = (
            association.associate_detections_to_tracks(
                tracks=tracker.tracks,
                detections=detections,
                distance_metric=tracker.metric,
                max_distance=cfg.phalp.hungarian_th,
                confirmed_track_indices=confirmed_indices,
            )
        )

        log.debug(
            f"  Matches: {len(matches)}, "
            f"Unmatched tracks: {len(unmatched_tracks)}, "
            f"Unmatched detections: {len(unmatched_detections)}"
        )

        # ============================================================
        # STAGE 5: TRACK UPDATE
        # ============================================================
        log.debug("Stage 5: Updating tracks")

        # Update matched tracks
        association.update_matched_tracks(
            tracks=tracker.tracks,
            detections=detections,
            matches=matches,
            shot_change=(shot_change == 1),
        )

        # Mark missed tracks
        association.mark_unmatched_tracks(
            tracks=tracker.tracks,
            unmatched_track_indices=unmatched_tracks,
        )

        # Create new tracks
        tracker._next_id = association.initiate_new_tracks(
            tracks=tracker.tracks,
            detections=detections,
            unmatched_detection_indices=unmatched_detections,
            cfg=cfg,
            next_track_id=tracker._next_id,
            n_init=cfg.phalp.n_init,
            max_age=cfg.phalp.max_age_track,
            dims=(4096, 4096, 99),
        )

        # ============================================================
        # STAGE 6: FUTURE PREDICTION
        # ============================================================
        log.debug("Stage 6: Predicting future states")
        matched_and_unmatched = [i[0] for i in matches] + unmatched_tracks
        tracker.accumulate_vectors(matched_and_unmatched)

        # Update distance metric gallery
        active_targets = [
            t.track_id for t in tracker.tracks if t.is_confirmed() or t.is_tentative()
        ]
        appe_features = [
            track.track_data["prediction"]["appe"][-1]
            for track in tracker.tracks
            if track.is_confirmed() or track.is_tentative()
        ]
        loca_features = [
            track.track_data["prediction"]["loca"][-1]
            for track in tracker.tracks
            if track.is_confirmed() or track.is_tentative()
        ]
        pose_features = [
            track.track_data["prediction"]["pose"][-1]
            for track in tracker.tracks
            if track.is_confirmed() or track.is_tentative()
        ]
        uv_maps = [
            track.track_data["prediction"]["uv"][-1]
            for track in tracker.tracks
            if track.is_confirmed() or track.is_tentative()
        ]

        if len(active_targets) > 0:
            tracker.metric.partial_fit(
                np.asarray(appe_features),
                np.asarray(loca_features),
                np.asarray(pose_features),
                np.asarray(uv_maps),
                np.asarray(active_targets),
                active_targets,
            )

        # Remove deleted tracks
        tracker.tracks = association.remove_deleted_tracks(tracker.tracks)

        log.debug(f"  Active tracks: {len(tracker.tracks)}")

        # ============================================================
        # STAGE 7: RESULT RECORDING
        # ============================================================
        _record_frame_results(
            final_visuals_dic,
            tracked_frames,
            tracker,
            frame_name,
            frame_path,
            t,
            shot_change,
            visual_store_,
            history_keys,
            prediction_keys,
            cfg,
        )

        # Clean up temporary keys for rendering
        if cfg.render.enable and t >= cfg.phalp.n_init:
            d = cfg.phalp.n_init + 1 if (t + 1 == len(list_of_frames)) else 1
            for t_ in range(t, t + d):
                frame_key = list_of_frames[t_ - cfg.phalp.n_init].name
                for tkey in tmp_keys:
                    if tkey in final_visuals_dic.get(frame_key, {}):
                        del final_visuals_dic[frame_key][tkey]

    log.info(f"Tracking complete. Processed {len(final_visuals_dic)} frames.")
    return final_visuals_dic


def _record_frame_results(
    final_visuals_dic,
    tracked_frames,
    tracker,
    frame_name,
    frame_path,
    t,
    shot_change,
    visual_store_,
    history_keys,
    prediction_keys,
    cfg,
):
    """Record tracking results for the current frame."""
    # Initialize frame entry
    final_visuals_dic.setdefault(
        frame_name,
        {"time": t, "shot": shot_change, "frame_path": frame_path},
    )

    for key_ in visual_store_:
        final_visuals_dic[frame_name][key_] = []

    # Record track data
    for tracks_ in tracker.tracks:
        if frame_name not in tracked_frames:
            tracked_frames.append(frame_name)
        if not tracks_.is_confirmed():
            continue

        track_id = tracks_.track_id
        track_data_hist = tracks_.track_data["history"][-1]
        track_data_pred = tracks_.track_data["prediction"]

        final_visuals_dic[frame_name]["tid"].append(track_id)
        final_visuals_dic[frame_name]["bbox"].append(track_data_hist["bbox"])
        final_visuals_dic[frame_name]["tracked_time"].append(tracks_.time_since_update)

        for hkey in history_keys:
            final_visuals_dic[frame_name][hkey].append(track_data_hist[hkey])

        for pkey in prediction_keys:
            final_visuals_dic[frame_name][pkey].append(
                track_data_pred[pkey.split("_")[1]][-1]
            )

        if tracks_.time_since_update == 0:
            final_visuals_dic[frame_name]["tracked_ids"].append(track_id)
            final_visuals_dic[frame_name]["tracked_bbox"].append(
                track_data_hist["bbox"]
            )

            # Handle track initialization
            if tracks_.hits == cfg.phalp.n_init:
                for pt in range(cfg.phalp.n_init - 1):
                    track_data_hist_ = tracks_.track_data["history"][-2 - pt]
                    track_data_pred_ = tracks_.track_data["prediction"]
                    frame_name_ = tracked_frames[-2 - pt]
                    final_visuals_dic[frame_name_]["tid"].append(track_id)
                    final_visuals_dic[frame_name_]["bbox"].append(
                        track_data_hist_["bbox"]
                    )
                    final_visuals_dic[frame_name_]["tracked_ids"].append(track_id)
                    final_visuals_dic[frame_name_]["tracked_bbox"].append(
                        track_data_hist_["bbox"]
                    )
                    final_visuals_dic[frame_name_]["tracked_time"].append(0)

                    for hkey in history_keys:
                        final_visuals_dic[frame_name_][hkey].append(
                            track_data_hist_[hkey]
                        )
                    for pkey in prediction_keys:
                        final_visuals_dic[frame_name_][pkey].append(
                            track_data_pred_[pkey.split("_")[1]][-1]
                        )


def main():
    cfg = tyro.cli(Human4DTrackingConfig)

    log.info("=" * 60)
    log.info("HMR Tracking v2 - Using Modular Pipeline Architecture")
    log.info("=" * 60)

    # Set up models (same as original)
    log.info("Loading models...")
    hmr_model = HMR2023TextureSampler(cfg)
    hmr_model.eval()
    hmr_model.to(cfg.device)
    log.info("  ✓ HMR2023TextureSampler loaded")

    pose_predictor = setup_predictor(cfg, hmr_model.smpl)
    pose_predictor.eval()
    pose_predictor.to(cfg.device)
    log.info("  ✓ Pose predictor loaded")

    detector = setup_detectron2(cfg)
    log.info("  ✓ Detectron2 detector loaded")

    # Get video frames
    log.info(f"Loading video: {cfg.video_source}")
    with VideoIOManager(cfg.video_io, cfg.render.fps) as video_io:
        video_name, list_of_frames = video_io.get_frames_from_source(cfg.video_source)
    log.info(f"  ✓ Loaded {len(list_of_frames)} frames")

    # Do tracking using modular pipeline
    log.info("Starting tracking with modular pipeline...")
    tracklets_data = track_video_modular(
        cfg=cfg,
        video_name=video_name,
        list_of_frames=list_of_frames,
        hmr_model=hmr_model,
        pose_predictor=pose_predictor,
        detector=detector,
    )

    # Save data to joblib pickle
    pkl_path = cfg.video_io.output_dir / f"{cfg.track_dataset}_{video_name}.pkl"
    if not cfg.overwrite and pkl_path.is_file():
        raise ValueError(
            f"{pkl_path} already exists, video has likely already been processed."
        )

    log.info(f"Saving results to {pkl_path}")
    joblib.dump(tracklets_data, pkl_path, compress=3)
    log.info("  ✓ Results saved")

    # Do rendering (same as original)
    if cfg.render.enable:
        log.info("Rendering visualization...")
        video_path = cfg.video_io.output_dir / f"{cfg.base_tracker}_{video_name}.mp4"
        visualizer = Visualizer(
            cfg.render, hmr_model.smpl, cfg.EXTRA.FOCAL_LENGTH, cfg.SMPL.TEXTURE
        )
        visualizer.reset_render(cfg.render.res * cfg.render.up_scale)

        with VideoWriter(cfg.video_io, cfg.render.fps) as vwriter:
            for i, frame_key in enumerate(sorted(tracklets_data.keys())):
                frame_path = tracklets_data[frame_key]["frame_path"]
                image_frame = cv2.imread(str(frame_path))
                rendered, f_size = visualizer.render_video(
                    image_frame, tracklets_data[frame_key]
                )

                # Save the rendered frame
                vwriter.save_video(video_path, rendered, f_size, t=i)

                if (i + 1) % 10 == 0:
                    log.info(f"  Rendered {i + 1}/{len(tracklets_data)} frames")

        log.info(f"  ✓ Video saved to {video_path}")

    log.info("=" * 60)
    log.info("Tracking complete!")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
