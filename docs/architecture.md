# PHALP Tracking System Architecture

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Pipeline Stages](#pipeline-stages)
4. [Models](#models)
5. [Data Structures](#data-structures)
6. [Type Annotations](#type-annotations)
7. [Data Flow](#data-flow)

---

## Overview

PHALP (Pose and HMR Articulated Localization Pipeline) is a multi-person
tracking system that combines state-of-the-art human pose and mesh estimation
with appearance-based tracking. The system processes video frames to track
people across time, estimating their 3D pose, shape, and appearance.

### Key Features

- **Multi-person tracking**: Track multiple people simultaneously across video
  frames
- **3D pose estimation**: Recover full 3D body pose using SMPL model
- **Appearance features**: Extract texture-based appearance features for robust
  re-identification
- **Temporal consistency**: Use transformer-based predictors for smooth
  trajectories
- **Multi-modal association**: Combine appearance, pose, and location for track
  matching

### Technology Stack

- **Deep Learning**: PyTorch, PyTorch Lightning
- **3D Body Model**: SMPL (Skinned Multi-Person Linear model)
- **Detection**: Detectron2 (Mask R-CNN)
- **Tracking**: Modified Deep SORT with multi-modal features
- **Type Safety**: jaxtyping with beartype runtime checking

---

## System Architecture

The PHALP system is organized into modular components:

```
humanoid_vision/
├── pipeline/              # Modular pipeline stages
│   ├── detection.py      # Person detection stage
│   ├── feature_extraction.py  # Feature extraction stage
│   └── association.py    # Track-detection association
├── models/               # Neural network models
│   ├── hmr2.py          # HMR2 pose/shape estimator
│   ├── hmar.py          # HMAR texture/appearance model
│   ├── smpl_wrapper.py  # SMPL body model wrapper
│   └── predictors/      # Temporal prediction models
├── deep_sort/           # Tracking algorithms
│   ├── tracker.py       # Main tracker implementation
│   ├── nn_matching.py   # Distance metrics
│   ├── forward_prediction.py  # Future state prediction
│   └── linear_assignment.py   # Hungarian matching
├── common/              # Shared data structures
│   ├── types.py         # jaxtyping aliases for tensor shapes
│   ├── detection.py     # Detection class
│   ├── track.py         # Track class
│   └── *_output.py      # Output dataclasses
├── utils/
│   ├── uv_texture_renderer.py  # Image → UV atlas projection
│   └── video.py         # Frame reading, shot detection
└── trackers/            # Model setup helpers
    └── __init__.py      # setup_predictor / setup_detectron2
```

The stages are composed by `scripts/track.py`; there is no tracker class that
owns the loop.

---

## Pipeline Stages

The PHALP tracking pipeline consists of several sequential stages, each with
well-defined inputs and outputs.

### Stage 1: Detection

**Module**: `humanoid_vision.pipeline.detection`

**Purpose**: Detect all people in the current video frame

**Input**:

- `image`: Raw video frame, shape `(H, W, 3)`, BGR format, `np.ndarray`
- `detector`: Detectron2 predictor instance
- `confidence_threshold`: Minimum detection confidence (default: 0.8)
- `expand_bbox_shape`: Optional target aspect ratio for bbox expansion

**Process**:

1. Run Detectron2 person detector on image
2. Filter detections for person class (class ID = 0)
3. Filter by confidence threshold
4. Optionally expand bounding boxes to target aspect ratio

**Output**:

- `pred_bbox`: Bounding boxes in (x1, y1, x2, y2) format, shape `(N, 4)`
- `pred_bbox_padded`: Expanded bounding boxes, shape `(N, 4)`
- `pred_masks`: Binary segmentation masks, shape `(N, H, W)`
- `pred_scores`: Detection confidence scores, shape `(N,)`
- `pred_classes`: Class IDs (all 0 for person), shape `(N,)`

**Key Functions**:

```python
@jaxtyped(typechecker=beartype)
def run_detection(
    detector,
    image: Float[Array, "height width 3"],
    confidence_threshold: float = 0.8,
    expand_bbox_shape: tuple[int, int] | None = None,
) -> tuple[BBoxes, BBoxes, Masks, Scores, Classes]:
    ...
```

---

### Stage 2: Feature Extraction

**Module**: `humanoid_vision.pipeline.feature_extraction`

**Purpose**: Extract appearance, pose, and location features from detected
people

**Input**:

- `image`: Original video frame, shape `(H, W, 3)`
- `bbox`: Bounding boxes from detection, shape `(N, 4)`
- `bbox_padded`: Padded bounding boxes, shape `(N, 4)`
- `masks`: Segmentation masks, shape `(N, H, W)`
- `hmr2_model`: HMR2 model instance (pose/shape)
- `hmar_model`: HMAR model instance (appearance)
- `uv_renderer`: UVTextureRenderer instance (texture sampling)

**Process**:

#### 2.1 Image Preprocessing

1. Crop each detected person using padded bounding box
2. Apply segmentation mask to create RGBA image
3. Resize to 256×256 pixels
4. Stack into batch tensor: `(num_persons, 4, 256, 256)`

#### 2.2 HMR Forward Pass

1. **Backbone (HMR2)**: Extract visual features using ViT backbone
   - Input: `(B, 3, 256, 192)` (cropped for aspect ratio)
   - Output: `(B, 1280, 16, 12)` feature maps

2. **SMPL Head (HMR2)**: Predict body pose and shape parameters
   - Input: Feature maps from backbone
   - Output:
     - `global_orient`: `(B, 1, 3, 3)` - Global rotation matrix
     - `body_pose`: `(B, 23, 3, 3)` - Joint rotation matrices
     - `betas`: `(B, 10)` - Shape parameters

3. **SMPL Forward**: Generate 3D mesh and joints
   - Input: SMPL parameters
   - Output:
     - `vertices`: `(B, 6890, 3)` - 3D mesh vertices
     - `joints`: `(B, 45, 3)` - 3D joint locations

4. **Texture Extraction**: Sample appearance from image
   - Unproject UV map to 3D mesh vertices (UVTextureRenderer)
   - Project to 2D image coordinates (UVTextureRenderer)
   - Render depth for visibility checking (UVTextureRenderer)
   - Sample RGB+mask at visible UV locations (UVTextureRenderer)
   - Output: `uv_image` `(B, 4, 256, 256)` - RGBA texture in UV space

#### 2.3 Appearance Embedding

1. Process UV image through HMAR autoencoder
2. Extract latent code as appearance embedding
3. Output: `appe_embedding` `(B, 4096)` - Compressed appearance features

#### 2.4 Pose Embedding

1. **Option A** (joints): Flatten 3D joints
   - Output: `pose_embedding` `(B, 135)` - 45 joints × 3

2. **Option B** (SMPL): Concatenate SMPL parameters
   - Flatten rotation matrices: 9 + 207 dimensions
   - Add shape parameters: 10 dimensions
   - Add camera parameters: 3 dimensions
   - Output: `pose_embedding` `(B, 229)` - Full SMPL vector

#### 2.5 Location Embedding

1. Compute camera translation from weak perspective camera
2. Project 3D joints to 2D image coordinates
3. Normalize 2D joints by image size
4. Concatenate: [2D joints (90), camera (3), camera (3), camera (3)]
5. Output: `loca_embedding` `(B, 99)` - Location features

**Output**:

- `hmar_out`: Complete `HMAROutput` with all predictions (built from HMR2 + UVTextureRenderer + HMAR)
- `appe_embedding`: Appearance features, shape `(num_persons, 4096)`
- `pose_embedding`: Pose features, shape `(num_persons, 135 or 229)`
- `loca_embedding`: Location features, shape `(num_persons, 99)`
- `pred_joints_2d`: 2D joint projections, shape `(num_persons, 45, 2)`
- `pred_cam`: Camera translation, shape `(num_persons, 3)`
- `list[Detection]`: Detection objects containing all features

**Key Functions**:

```python
@jaxtyped(typechecker=beartype)
def extract_hmr_features(
    hmr2_model,
    hmar_model,
    uv_renderer,
    masked_images: Float[Array, "num_persons 4 256 256"],
    centers: list[tuple[float, float]],
    scales: list[tuple[float, float]],
    img_offset: tuple[float, float],
    render_res: int,
) -> tuple[HMAROutput, AppearanceEmbed, PoseEmbed, LocationEmbed, ...]:
    ...
```

---

### Stage 3: Track Prediction

**Module**: `humanoid_vision.deep_sort.tracker.Tracker.predict()`

**Purpose**: Age existing tracks and prepare for association

**Input**:

- `self.tracks`: List of active Track objects

**Process**:

1. For each track, increment age counter
2. Increment time_since_update counter
3. Prepare track state for association

**Output**:

- Updated track states (modified in-place)

---

### Stage 4: Data Association

**Module**: `humanoid_vision.deep_sort.tracker.Tracker._match()`

**Purpose**: Associate detections with existing tracks using multi-modal
distance metric

**Input**:

- `detections`: List of Detection objects from current frame
- `self.tracks`: List of existing Track objects

**Process**:

#### 4.1 Compute Distance Matrix

For each track and detection pair, compute multi-modal distance:

1. **Appearance Distance** (if using texture features):
   ```python
   # Extract UV maps from track history and detections
   for track in tracks:
       track_uv = track.track_data["prediction"]["uv"][-1]  # (4, 256, 256)
   ```

for detection in detections: detect_uv = detection.uv # (4, 256, 256)

# Compute UV embedding distance

track_emb, detect_emb, overlap = HMAR.get_uv_distance(track_uv, detect_uv)
appearance_dist = sum((track_emb - detect_emb) ** 2) * 100

2. **Pose Distance**:

```python
# Extract pose embeddings
track_pose = track_embedding[229 or 135 dim]
detect_pose = detection_embedding[229 or 135 dim]

# Compute L2 distance
pose_distance = sqrt(sum((track_pose - detect_pose) ** 2))
```

3. **Location Distance**:
   ```python
   # Extract 2D joint centroids (joint 44 is pelvis)
   track_loc = track_location[:, 44, :]  # (2,) normalized coordinates
   detect_loc = detect_location[:, 44, :]  # (2,) normalized coordinates

   # Euclidean distance
   loc_distance = sqrt(sum((track_loc - detect_loc) ** 2))
   ```

4. **Depth/Nearness**:
   ```python
   # Extract depth from camera z-coordinate
   track_depth = track_camera[2]  # z-translation
   detect_depth = detect_camera[2]

   # Ratio of depths (clamped)
   nearness = min(track_depth, detect_depth) / max(track_depth, detect_depth)
   depth_cost = -log(nearness)
   ```

5. **Combined Distance** (e.g., "EQ_010" metric):
   ```python
   # Weighted combination with learned parameters
   betas = [3.8303, 1.5207, 0.4930, 4.5831]

   combined_distance = (
       (1 + appearance_dist * betas[0]) *
       (1 + pose_distance * betas[1]) *
       exp(loc_distance / betas[2]) *
       exp(depth_cost / betas[3])
   )
   ```

Result: Cost matrix of shape `(num_tracks, num_detections)`

#### 4.2 Hungarian Matching

1. Apply Hungarian algorithm (linear sum assignment) to find optimal matches
2. Threshold matches by maximum distance
3. Separate into:
   - `matches`: Valid track-detection pairs
   - `unmatched_tracks`: Tracks without detections
   - `unmatched_detections`: Detections without tracks

**Output**:

- `matches`: List of `(track_idx, detection_idx)` tuples
- `unmatched_tracks`: List of track indices
- `unmatched_detections`: List of detection indices
- `cost_matrix`: Full distance matrix `(num_tracks, num_detections)`

**Key Functions**:

```python
@jaxtyped(typechecker=beartype)
def _pdist(
    a: List,  # Track features
    b: List,  # Detection features
    prediction_features: str,
    distance_type: str,
    shot: int,
    HMAR: Optional[HMAR] = None,
) -> Float[Array, "num_tracks num_detections"]:
    ...
```

---

### Stage 5: Track Update

**Module**: `humanoid_vision.deep_sort.tracker.Tracker.update()`

**Purpose**: Update tracks with matched detections and manage track lifecycle

**Input**:

- `matches`: List of `(track_idx, detection_idx)` pairs
- `unmatched_tracks`: List of unmatched track indices
- `unmatched_detections`: List of unmatched detection indices
- `detections`: List of Detection objects
- `shot`: Shot change flag (0 or 1)

**Process**:

#### 5.1 Update Matched Tracks

For each matched pair:

1. Append detection data to track history
2. Update UV texture with temporal smoothing:
   ```python
   mixing_alpha = config.alpha * (detection.conf ** 2)
   new_uv = (1 - mixing_alpha) * old_uv + mixing_alpha * new_uv
   ```
3. Increment hit counter, reset time_since_update
4. Confirm track if hits >= n_init

#### 5.2 Mark Missed Tracks

For unmatched tracks:

1. Mark as missed
2. If tentative, delete immediately
3. If confirmed, delete if time_since_update > max_age

#### 5.3 Initialize New Tracks

For unmatched detections:

1. Create new Track object
2. Initialize with detection data
3. Add to tracks list
4. Assign new track ID

**Output**:

- Updated `self.tracks` list
- Confirmed/deleted tracks

---

### Stage 6: Future Prediction

**Module**: `humanoid_vision.deep_sort.tracker.Tracker.accumulate_vectors()`

**Purpose**: Predict future states for all tracks using temporal models

**Input**:

- `track_ids`: List of track indices to predict
- Historical data from tracks:
  - Pose history: `(track_history, 229)` per track
  - Location history: `(track_history, 99)` per track
  - Time stamps: `(track_history,)` per track

**Process**:

#### 6.1 Pose Prediction (if "P" in config.predict)

Use transformer-based model:

```python
# Collect historical pose data
p_features = np.array([
    track.track_data["history"][i]["pose"]
    for i in range(track_history)
])  # (num_tracks, track_history, 229)

# Prepare auxiliary data
p_data = np.array([
    [x, y, scale, scale, time, track_id]
    for each history entry
])  # (num_tracks, track_history, 6)

# Predict future pose
p_pred = predict_future_pose(
    p_features, p_data, t_features, time_since_update, pose_predictor
)  # (num_tracks, 229)
```

**PoseTransformerV2 Architecture**:

1. **Input Encoding**: Encode 229-dim pose vectors
2. **Positional Encoding**: Add learned temporal positions
3. **Transformer 1**: Multi-head self-attention over time
4. **Conv Encoder/Decoder**: Temporal convolution for smoothing
5. **Transformer 2**: Final refinement
6. **Readout Heads**:
   - SMPL parameter head: Predict pose/shape
   - Location head: Predict 3D camera position
   - Action head: Predict action class (optional)

#### 6.2 Location Prediction (if "L" in config.predict)

Use Ridge regression for extrapolation:

```python
# Collect historical location data
l_features = np.array([
    track.track_data["history"][i]["loca"]
    for i in range(track_history)
])  # (num_tracks, track_history, 99)

# Extract 2D centroid and depth
xy_positions = l_features[:, :, 44*2:44*2+2]  # Pelvis joint
depth = l_features[:, :, 90]  # Camera z

# Fit polynomial regression (degree 1)
for each track:
    # Predict x, y, depth separately
    x_pred = Ridge(alpha=1.2).fit(time, x).predict(time + delta)
    y_pred = Ridge(alpha=2.0).fit(time, y).predict(time + delta)
    depth_pred = Ridge(alpha=5.0).fit(time, log(depth)).predict(time + delta)
    
    # Compute prediction intervals
    x_interval = get_prediction_interval(x, x_hat, time, time + delta)
    y_interval = get_prediction_interval(y, y_hat, time, time + delta)
    depth_interval = get_prediction_interval(...)

l_pred = [x_pred, y_pred, exp(depth_pred), x_interval, y_interval, ...]
```

#### 6.3 Store Predictions

Store predicted features in track's prediction queue:

```python
track.track_data["prediction"]["pose"].append(p_pred)
track.track_data["prediction"]["loca"].append(l_pred)
```

**Output**:

- Updated prediction data for all tracks
- Predictions used in next frame's distance computation

**Key Functions**:

```python
@jaxtyped(typechecker=beartype)
def predict_future_pose(
    p_features: Float[Array, "num_tracks track_history pose_dim"],
    p_data: Float[Array, "num_tracks track_history 6"],
    t_feature: Int[Array, "num_tracks track_history"],
    time: Int[Array, "num_tracks"],
    pose_predictor: PoseTransformerV2,
) -> Float[Array, "num_tracks pose_dim"]:
    ...

@jaxtyped(typechecker=beartype)
def predict_future_location(
    l_features: Float[Array, "num_tracks track_history loca_dim"],
    t_feature: Int[Array, "num_tracks track_history"],
    confidence: Float[Array, "num_tracks track_history"],
    time: Int[Array, "num_tracks"],
    distance_type: str,
) -> Float[Array, "num_tracks loca_dim"]:
    ...
```

---

### Stage 7: Result Recording

**Module**: `scripts/track.py` (`_record_frame_results`)

**Purpose**: Record tracking results for each frame

**Input**:

- `tracker.tracks`: Updated tracks after association
- `frame_name`: Current frame identifier
- `frame_time`: Frame index/timestamp

**Process**:

1. Create frame entry in results dictionary
2. For each confirmed track:
   - Record track ID
   - Record current bounding box
   - Record all history data (pose, camera, joints, appearance)
   - Record prediction data
   - Record time since last detection

3. Handle track initialization:
   - When a track is first confirmed (hits == n_init)
   - Retroactively add track ID to previous n_init-1 frames

**Output**:

- `final_visuals_dic`: Dictionary mapping frame names to tracking data
  - Structure:
    ```python
    {
        "frame_001.jpg": {
            "time": 0,
            "shot": 0,
            "frame_path": Path(...),
            "tid": [1, 2, 3],  # Track IDs
            "bbox": [...],  # Bounding boxes
            "tracked_ids": [1, 2],  # IDs with detections this frame
            "tracked_bbox": [...],  # Corresponding bboxes
            "tracked_time": [0, 0, 5],  # Frames since last detection
            # Per-track data:
            "smpl": [...],  # SMPL parameters
            "camera": [...],  # Camera translations
            "3d_joints": [...],  # 3D joint locations
            "2d_joints": [...],  # 2D joint projections
            "appe": [...],  # Appearance embeddings
            "loca": [...],  # Location embeddings
            "pose": [...],  # Pose embeddings
            "uv": [...],  # UV texture maps
            # ... and more
        },
        # ... more frames
    }
    ```

---

## Models

### 1. HMR2 (Human Mesh Recovery 2)

**Module**: `humanoid_vision.models.hmr2.HMR2`

**Type**: PyTorch Lightning module

**Purpose**: Estimate 3D human pose and shape from a single RGB image

**Architecture**:

- **Backbone**: Vision Transformer (ViT) for feature extraction
  - Input: `(B, 3, 256, 192)` RGB crops
  - Output: `(B, 1280, 16, 12)` feature maps

- **SMPL Head**: Transformer decoder for SMPL parameter regression
  - Input: Visual features
  - Output: Rotation matrices and shape parameters

**Forward Pass**:

```python
@jaxtyped(typechecker=beartype)
def forward(self, batch: dict) -> HMROutput:
    # Extract features
    features = self.backbone(batch["img"][:, :, :, 32:-32])
    
    # Predict SMPL parameters
    smpl_params, pred_cam, _ = self.smpl_head(features)
    
    # Run SMPL forward
    smpl_output = self.smpl(smpl_params)
    
    # Project to 2D
    keypoints_2d = perspective_projection(
        smpl_output.joints, pred_cam_t, focal_length
    )
    
    return HMROutput(...)
```

**Training**:

- Losses: 2D/3D keypoint loss, SMPL parameter loss
- Optional adversarial loss for realistic pose distribution
- Optimizer: AdamW with learning rate 1e-4

---

### 2. HMAR (Human Mesh and Appearance Recovery)

**Module**: `humanoid_vision.models.hmar.HMAR`

**Type**: PyTorch module

**Purpose**: Extract appearance features and UV texture maps

**Architecture**:

- **Backbone**: ResNet for feature extraction
- **Texture Head**: Predicts dense correspondence (flow field)
- **Encoding Head**: Autoencoder for texture compression
- **SMPL Head**: MLP-based SMPL predictor (frozen during texture extraction)

**Key Method**:

```python
def get_uv_distance(t_uv, d_uv) -> (embedding1, embedding2, overlap):
    # Compute overlap mask
    mask = (t_uv[3, :, :] > 0.5) & (d_uv[3, :, :] > 0.5)
    
    # Encode to latent space
    t_emb = self.autoencoder_hmar(t_uv, en=True)
    d_emb = self.autoencoder_hmar(d_uv, en=True)
    
    # Return embeddings and overlap ratio
    return t_emb, d_emb, overlap_ratio
```

---

### 3. UVTextureRenderer

**Module**: `humanoid_vision.utils.uv_texture_renderer.UVTextureRenderer`

**Type**: PyTorch module wrapping `neural_renderer`

**Purpose**: Project image evidence onto the canonical UV atlas

Unprojects the canonical UV map (`bmap_256.npy` / `fmap_256.npy`) onto the
predicted SMPL surface, perspective-projects those points back to the image
plane, and bilinearly samples the RGBA image there. A depth render of the mesh
masks out self-occluded texels.

`HMR2`, `HMAR`, and `UVTextureRenderer` are instantiated independently and
composed by `pipeline.feature_extraction.extract_hmr_features`.

---

### 4. SMPL (Skinned Multi-Person Linear Model)

**Module**: `humanoid_vision.models.smpl_wrapper.SMPL`

**Type**: Extends smplx.SMPLLayer

**Purpose**: Statistical 3D body model that generates mesh from parameters

**Input**:

- `global_orient`: Global rotation, `(B, 1, 3, 3)`
- `body_pose`: Joint rotations, `(B, 23, 3, 3)`
- `betas`: Shape parameters, `(B, 10)`

**Output**:

- `vertices`: 3D mesh vertices, `(B, 6890, 3)`
- `joints`: 3D joint locations, `(B, 45, 3)` (25 base + 20 extra)

**Forward Pass**:

```python
@jaxtyped(typechecker=beartype)
def forward(self, hmr_smpl_output: HMRSMPLOutput) -> SMPLOutput:
    # Run base SMPL
    smpl_output = super().forward(pose2rot=False, **asdict(hmr_smpl_output))
    
    # Remap joints to OpenPose convention
    joints = smpl_output.joints[:, self.joint_map, :]
    
    # Optionally adjust hip locations
    if self.update_hips:
        joints[:, [9, 12]] = ...  # Hip adjustment
    
    # Add extra joints from regressor
    if hasattr(self, "joint_regressor_extra"):
        extra_joints = vertices2joints(self.joint_regressor_extra, vertices)
        joints = torch.cat([joints, extra_joints], dim=1)
    
    return SMPLOutput(vertices=vertices, joints=joints, ...)
```

---

### 5. PoseTransformerV2

**Module**:
`humanoid_vision.models.predictors.pose_transformer_v2.PoseTransformerV2`

**Type**: PyTorch module

**Purpose**: Predict future poses and smooth trajectories using temporal
transformers

**Architecture**:

```
Input: (batch, time, num_people, 229)
  ↓
Pose Encoder: Linear (229 → 512)
  ↓
Positional Encoding: Learned (time)
  ↓
Transformer 1: Multi-head attention
  ↓
Conv Encoder: 1D convolution (stride=2)
  ↓
Conv Decoder: 1D transposed convolution
  ↓
Positional Encoding: Learned (time)
  ↓
Transformer 2: Multi-head attention
  ↓
Readout Heads:
  ├─ SMPL Head: → (global_orient, body_pose, betas)
  ├─ Location Head: → (camera_x, camera_y, camera_z)
  └─ Action Head: → (action_classes)
```

**Key Method**:

```python
def predict_next(
    en_pose: Float[Tensor, "batch time 229"],
    en_data: Float[Tensor, "batch time 6"],
    en_time: Float[Tensor, "batch time"],
    time_to_predict: int
) -> Float[Tensor, "batch 229"]:
    # Normalize input
    pose_norm = (en_pose - self.mean) / (self.std + 1e-10)
    
    # Create input data structure
    input_data = {
        "pose_shape": pose_norm,
        "has_detection": detection_mask,
        "mask_detection": zeros,
    }
    
    # Forward through encoder
    output, _ = self.encoder(input_data, mask_type="zero")
    
    # Readout predicted pose at target time
    decoded = self.readout_pose(output[:, self.cfg.max_people:, :])
    predicted_pose = decoded["pose_camera"][:, target_time, 0, :]
    
    return predicted_pose
```

---

### 6. Deep SORT Tracker

**Module**: `humanoid_vision.deep_sort.tracker.Tracker`

**Type**: Custom multi-object tracker

**Purpose**: Manage track lifecycle and associate detections to tracks

**State**:

- `self.tracks`: List of active Track objects
- `self.metric`: NearestNeighborDistanceMetric for distance computation
- `self.pose_predictor`: PoseTransformerV2 for future prediction
- `self._next_id`: Counter for assigning new track IDs

**Key Methods**:

```python
def predict():
    # Age all tracks
    for track in self.tracks:
        track.age += 1
        track.time_since_update += 1

def update(detections, frame_t, image_name, shot):
    # Associate detections to tracks
    matches, unmatched_tracks, unmatched_detections = self._match(detections)
    
    # Update matched tracks
    for track_idx, detection_idx in matches:
        self.tracks[track_idx].update(detections[detection_idx], ...)
    
    # Mark missed tracks
    for track_idx in unmatched_tracks:
        self.tracks[track_idx].mark_missed()
    
    # Create new tracks
    for detection_idx in unmatched_detections:
        self._initiate_track(detections[detection_idx], ...)
    
    # Predict future states
    self.accumulate_vectors(track_indices)
    
    # Update distance metric gallery
    self.metric.partial_fit(features, targets, ...)
    
    # Remove deleted tracks
    self.tracks = [t for t in self.tracks if not t.is_deleted()]
```

---

## Data Structures

### Detection

**Module**: `humanoid_vision.common.detection.Detection`

**Purpose**: Encapsulate all information about a detected person

**Fields**:

```python
detection_data = {
    # Geometry
    "bbox": Float[Array, "4"],  # (x, y, w, h)
    "mask": RLE,  # Run-length encoded mask
    "center": tuple[float, float],  # Bbox center
    "scale": tuple[float, float],  # Bbox scale
    "xy": tuple[float, float],  # Normalized center
    "size": tuple[int, int],  # Image dimensions
    
    # Features
    "appe": Float[Array, "4096"],  # Appearance embedding
    "pose": Float[Array, "229"],  # Pose embedding
    "loca": Float[Array, "99"],  # Location embedding
    "uv": Float[Array, "4 256 256"],  # UV texture map
    "embedding": Float[Array, "4324"],  # Concatenated features
    
    # 3D Data
    "smpl": dict,  # {body_pose, betas, global_orient}
    "camera": Float[Array, "3"],  # Camera translation
    "camera_bbox": Float[Array, "3"],  # Weak perspective camera
    "3d_joints": Float[Array, "45 3"],  # 3D joint locations
    "2d_joints": Float[Array, "90"],  # Normalized 2D joints
    
    # Metadata
    "conf": float,  # Detection confidence
    "class_name": int,  # Class ID
    "time": int,  # Frame timestamp
    "img_path": Path,  # Frame file path
    "img_name": str,  # Frame filename
    "ground_truth": int,  # GT track ID (for evaluation)
    "annotations": List,  # GT annotations
}
```

**Methods**:

```python
@jaxtyped(typechecker=beartype)
def to_tlbr() -> Float[Array, "4"]:
    """Convert bbox to (x1, y1, x2, y2) format"""
    ...

@jaxtyped(typechecker=beartype)
def to_xyah() -> Float[Array, "4"]:
    """Convert bbox to (cx, cy, aspect_ratio, height) format"""
    ...
```

---

### Track

**Module**: `humanoid_vision.common.track.Track`

**Purpose**: Represent a tracked person over time

**State**:

```python
class Track:
    # Identity
    track_id: int  # Unique track identifier
    
    # Counters
    age: int  # Total frames since creation
    hits: int  # Number of successful detections
    time_since_update: int  # Frames since last detection
    
    # State
    state: TrackState  # Tentative, Confirmed, or Deleted
    
    # Data
    track_data = {
        "history": deque(maxlen=track_history),  # Historical detections
        "prediction": {
            "appe": deque(maxlen=n_init+1),  # Predicted appearance
            "loca": deque(maxlen=n_init+1),  # Predicted location
            "pose": deque(maxlen=n_init+1),  # Predicted pose
            "uv": deque(maxlen=n_init+1),  # Predicted UV texture
        }
    }
```

**Methods**:

```python
def predict(increase_age=True):
    """Age the track"""
    self.age += 1
    self.time_since_update += 1

def update(detection, detection_id, shot):
    """Update track with new detection"""
    self.track_data["history"].append(detection.as_legacy_dict())
    
    # Temporal UV smoothing
    mixing_alpha = config.alpha * (detection.conf ** 2)
    new_uv = (1 - alpha) * old_uv + alpha * new_uv
    self.track_data["prediction"]["uv"].append(new_uv)
    
    self.hits += 1
    self.time_since_update = 0
    if self.hits >= self._n_init:
        self.state = TrackState.Confirmed

def mark_missed():
    """Mark track as missed"""
    if self.state == TrackState.Tentative:
        self.state = TrackState.Deleted
    elif self.time_since_update > self._max_age:
        self.state = TrackState.Deleted

def add_predicted(appe=None, pose=None, loca=None):
    """Add predicted features from temporal model"""
    self.track_data["prediction"]["appe"].append(appe or last_appe)
    self.track_data["prediction"]["loca"].append(loca or last_loca)
    self.track_data["prediction"]["pose"].append(pose or last_pose)
```

---

### Output Dataclasses

#### HMRSMPLOutput

```python
@dataclass
class HMRSMPLOutput:
    global_orient: Float[Tensor, "batch 1 3 3"]  # Global rotation
    body_pose: Float[Tensor, "batch 23 3 3"]  # Joint rotations
    betas: Float[Tensor, "batch 10"]  # Shape parameters
```

#### HMROutput

```python
@dataclass
class HMROutput(HMRSMPLOutput):
    # Inherited: global_orient, body_pose, betas
    
    pred_cam: Float[Tensor, "batch 3"]  # Weak perspective camera
    pred_cam_t: Float[Tensor, "batch 3"]  # Camera translation
    focal_length: Float[Tensor, "batch 2"]  # Focal length
    pred_keypoints_3d: Float[Tensor, "batch num_joints 3"]  # 3D joints
    pred_keypoints_2d: Float[Tensor, "batch num_joints 2"]  # 2D joints
    pred_vertices: Float[Tensor, "batch 6890 3"]  # Mesh vertices
    losses: Optional[dict[str, Tensor]]  # Training losses
```

#### HMAROutput

```python
@dataclass
class HMAROutput(HMROutput):
    # Inherited: all HMROutput fields
    
    uv_image: Float[Tensor, "batch 4 256 256"]  # RGBA texture map
    uv_vector: Float[Tensor, "batch 4 256 256"]  # Processed texture
```

---

## Type Annotations

The codebase uses `jaxtyping` with `beartype` for runtime type checking. This
provides:

1. **Self-documenting code**: Types clearly show tensor shapes
2. **Runtime validation**: Catch shape mismatches early
3. **IDE support**: Better autocomplete and error detection

### Type Aliases

**Module**: `humanoid_vision.common.types`

```python
# Tensor types
TensorBatch: TypeAlias = Float[Array, "batch ..."]
ArrayBatch: TypeAlias = Float[Array, "batch ..."]

# Image types
RGBImage: TypeAlias = Float[Tensor, "batch 3 height width"]
RGBAImage: TypeAlias = Float[Tensor, "batch 4 height width"]
Mask: TypeAlias = Float[Tensor, "batch height width"]

# Detection types
BBoxes: TypeAlias = Float[Array, "num_detections 4"]
Masks: TypeAlias = Bool[Array, "num_detections height width"]
Scores: TypeAlias = Float[Array, "num_detections"]

# SMPL types
GlobalOrient: TypeAlias = Float[Tensor, "batch 1 3 3"]
BodyPose: TypeAlias = Float[Tensor, "batch 23 3 3"]
Betas: TypeAlias = Float[Tensor, "batch 10"]
Vertices: TypeAlias = Float[Tensor, "batch 6890 3"]
Joints3D: TypeAlias = Float[Tensor, "batch num_joints 3"]

# Embedding types
AppearanceEmbed: TypeAlias = Float[Array, "batch 4096"]
PoseEmbed: TypeAlias = Float[Array, "batch pose_dim"]
LocationEmbed: TypeAlias = Float[Array, "batch 99"]
UVMap: TypeAlias = Float[Array, "batch 4 256 256"]
```

### Example Usage

```python
from jaxtyping import Float, jaxtyped
from beartype import beartype

@jaxtyped(typechecker=beartype)
def extract_features(
    images: Float[Tensor, "batch 4 256 256"],
    hmr2_model: HMR2,
    hmar_model: HMAR,
) -> tuple[
    Float[Array, "batch 4096"],  # Appearance
    Float[Array, "batch 229"],   # Pose
    Float[Array, "batch 99"],    # Location
]:
    # Runtime checks ensure:
    # - images is a float tensor
    # - First dimension is batch size
    # - Second dimension is exactly 4 (RGBA)
    # - Third and fourth dimensions are 256
    
    hmar_out = hmr2_model({"img": images[:, :3], "mask": images[:, 3].clip(0, 1)})
    appe = hmar_model.autoencoder_hmar(hmar_out.uv_vector, en=True)
    pose = hmar_out.pose_embed
    loca = hmar_out.location_embed
    
    # Return types are also validated
    return appe, pose, loca
```

---

## Data Flow

### Complete Pipeline Flow Diagram

```
Video Frame (H, W, 3)
        ↓
   [Detection]
        ↓
Bounding Boxes (N, 4)
Masks (N, H, W)
        ↓
   [Crop & Preprocess]
        ↓
RGBA Crops (N, 4, 256, 256)
        ↓
   [HMR2 Forward]
        ├─→ [ViT Backbone]
        │        ↓
        │   Features (N, 1280, 16, 12)
        │        ↓
        ├─→ [SMPL Head]
        │        ↓
        │   SMPL Params: global_orient (N, 1, 3, 3)
        │                 body_pose (N, 23, 3, 3)
        │                 betas (N, 10)
        │        ↓
        ├─→ [SMPL Forward]
        │        ↓
        │   Vertices (N, 6890, 3)
        │   Joints (N, 45, 3)
        │        ↓
        └─→ [Texture Extraction]
                 ↓
            UV Images (N, 4, 256, 256)
                 ↓
            [Appearance Encoder]
                 ↓
            Appearance Embed (N, 4096)
        
   [Create Embeddings]
        ├─→ Appearance: (N, 4096)
        ├─→ Pose: (N, 229)
        └─→ Location: (N, 99)
                 ↓
        [Create Detections]
                 ↓
    list[Detection] (N items)
                 ↓
        [Track Prediction]
                 ↓
    Age existing tracks
                 ↓
        [Data Association]
                 ↓
    Compute distance matrix (T, N)
    using: appearance + pose + location
                 ↓
    Hungarian matching
                 ↓
    matches: [(track_idx, det_idx), ...]
    unmatched_tracks: [track_idx, ...]
    unmatched_detections: [det_idx, ...]
                 ↓
        [Track Update]
        ├─→ Update matched tracks
        ├─→ Mark missed tracks
        └─→ Initialize new tracks
                 ↓
        [Future Prediction]
        ├─→ Pose Predictor (Transformer)
        │        ↓
        │   Future poses (T, 229)
        │        ↓
        └─→ Location Predictor (Ridge)
                 ↓
            Future locations (T, 99)
                 ↓
        [Store Predictions]
                 ↓
    Updated track.track_data["prediction"]
                 ↓
        [Record Results]
                 ↓
    frame_results = {
        "tid": [track IDs],
        "bbox": [bboxes],
        "smpl": [SMPL params],
        "camera": [cameras],
        "3d_joints": [joints],
        ...
    }
                 ↓
    final_visuals_dic[frame_name] = frame_results
```

### Tensor Shape Evolution

| Stage        | Data            | Shape               | Type            |
| ------------ | --------------- | ------------------- | --------------- |
| Input        | Video frame     | `(H, W, 3)`         | `np.uint8`      |
| Detection    | Bounding boxes  | `(N, 4)`            | `np.float32`    |
| Detection    | Masks           | `(N, H, W)`         | `bool`          |
| Crop         | RGBA crops      | `(N, 4, 256, 256)`  | `torch.float32` |
| Backbone     | Features        | `(N, 1280, 16, 12)` | `torch.float32` |
| SMPL Head    | Global orient   | `(N, 1, 3, 3)`      | `torch.float32` |
| SMPL Head    | Body pose       | `(N, 23, 3, 3)`     | `torch.float32` |
| SMPL Head    | Betas           | `(N, 10)`           | `torch.float32` |
| SMPL Forward | Vertices        | `(N, 6890, 3)`      | `torch.float32` |
| SMPL Forward | Joints          | `(N, 45, 3)`        | `torch.float32` |
| Texture      | UV image        | `(N, 4, 256, 256)`  | `torch.float32` |
| Encoder      | Appearance      | `(N, 4096)`         | `np.float32`    |
| Encoder      | Pose            | `(N, 229)`          | `np.float32`    |
| Encoder      | Location        | `(N, 99)`           | `np.float32`    |
| Association  | Cost matrix     | `(T, N)`            | `np.float32`    |
| Prediction   | Future pose     | `(T, 229)`          | `torch.float32` |
| Prediction   | Future location | `(T, 99)`           | `torch.float32` |

---

## Configuration

### Key Configuration Parameters

```python
cfg.phalp = {
    # Detection
    "low_th_c": 0.8,  # Detection confidence threshold
    "small_w": 20.0,  # Minimum bbox width
    "small_h": 20.0,  # Minimum bbox height
    
    # Tracking
    "max_age_track": 30,  # Max frames before deleting track
    "n_init": 3,  # Frames before confirming track
    "track_history": 7,  # Length of history for prediction
    "past_lookback": 20,  # Gallery size for distance metric
    
    # Features
    "predict": "APL",  # Which features to predict (A=appearance, P=pose, L=location)
    "pose_distance": "smpl",  # Type of pose embedding
    "distance_type": "EQ_010",  # Distance metric formula
    "hungarian_th": 100.0,  # Maximum matching distance
    
    # Temporal
    "alpha": 0.5,  # UV temporal smoothing factor
    
    # Prediction
    "shot": 0,  # Shot change flag
}

cfg.MODEL = {
    "IMAGE_SIZE": 256,  # Input image size
    "BACKBONE": {
        "TYPE": "vit",
        "PRETRAINED_WEIGHTS": "...",
    },
}

cfg.SMPL = {
    "MODEL_PATH": "...",
    "GENDER": "neutral",
    "NUM_BODY_JOINTS": 23,
    "JOINT_REGRESSOR_EXTRA": "...",
    "UPDATE_HIPS": True,
}

cfg.EXTRA = {
    "FOCAL_LENGTH": 5000.0,  # Camera focal length
    "NUM_LOG_IMAGES": 4,  # Images to log in tensorboard
}
```

---

## Usage Example

Tracking is driven by composing the pipeline stages directly (see
`scripts/track.py`):

```python
from humanoid_vision.pipeline import detection, feature_extraction, association
from humanoid_vision.deep_sort.tracker import Tracker
from humanoid_vision.models.hmr2 import HMR2
from humanoid_vision.models.hmar import HMAR
from humanoid_vision.utils.uv_texture_renderer import UVTextureRenderer

# Instantiate models separately
hmar_model = HMAR(cfg)
hmar_model.load_weights(cfg.hmr.hmar_path)
hmr2_model = HMR2.load_from_checkpoint(CKPT_PATH, strict=False, cfg=cfg).to(cfg.device).eval()
uv_renderer = UVTextureRenderer().to(cfg.device)

# Initialize tracker (needs HMAR for appearance distance)
tracker = Tracker(cfg, hmar_model, pose_predictor, max_age=30, n_init=3, dims=(4096, 4096, 99))

# Process each frame
for frame_path in frames:
    image = cv2.imread(str(frame_path))
    bbox, bbox_pad, masks, scores, classes = detection.run_detection(detector, image, confidence_threshold=0.8)
    masked_images, centers, scales, rles = feature_extraction.crop_and_preprocess_detections(image, bbox, bbox_pad, masks)
    hmar_out, appe, pose, loca, joints_3d, joints_2d, cam = feature_extraction.extract_hmr_features(
        hmr2_model, hmar_model, uv_renderer, masked_images, centers, scales, (left, top), render_res
    )
    detections = feature_extraction.create_detection_data_list(
        hmar_out, appe, pose, loca, bbox, rles, scores, classes, centers, scales,
        joints_3d, joints_2d, cam, frame_path, t, (height, width), [], []
    )
    # ... association and update as before
```

**Complete Example**: See `scripts/track.py` for a full working implementation.

---

## Performance Considerations

### Computational Bottlenecks

1. **HMR Forward Pass**: ~50-100ms per person on GPU
2. **Texture Extraction**: ~20-30ms per person (neural renderer)
3. **Appearance Encoding**: ~10ms per person
4. **Distance Computation**: ~1ms per track-detection pair
5. **Pose Prediction**: ~10ms per track (amortized over batch)

### Optimization Strategies

1. **Batch Processing**: Process all detections in a single forward pass
2. **GPU Utilization**: Keep models on GPU, minimize CPU-GPU transfers
3. **Lazy Evaluation**: Only predict when tracks are unmatched
4. **Feature Caching**: Store recent embeddings in distance metric gallery
5. **Spatial Pruning**: Only match tracks/detections within spatial proximity

### Memory Usage

- **HMR2 Model**: ~500MB
- **HMAR Model**: ~200MB
- **Pose Predictor**: ~100MB
- **Per Detection**: ~5MB (UV map + features)
- **Per Track**: ~35MB (history of 7 frames)

---

## References

1. **HMR2**: "4D-Humans: Reconstructing and Tracking Humans with Transformers"
   (Goel et al., 2023)
2. **SMPL**: "SMPL: A Skinned Multi-Person Linear Model" (Loper et al., 2015)
3. **Deep SORT**: "Simple Online and Realtime Tracking with a Deep Association
   Metric" (Wojke et al., 2017)
4. **PHALP**: "Tracking People by Predicting 3D Appearance, Location and Pose"
   (Rajasegaran et al., 2022)

---

## Changelog

### v2.0 (Current)

- Refactored pipeline into modular stages
- Added comprehensive jaxtyping annotations throughout
- Created pipeline module for reusable components
- Documented all stages with input/output specifications
- Added type aliases for common tensor shapes

### v1.0 (Original)

- Initial monolithic implementation
- Basic tracking functionality
- Limited documentation

---

_Last Updated: 2025-10-30_
