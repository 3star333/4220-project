# GPU Accelerated Real-Time Object Detection
## CPU Baseline — ROS2 + OpenCV

---

## Project Structure

```
gpu_object_detection/
├── include/gpu_object_detection/
│   ├── detection_pipeline.hpp   # IPipeline interface + CpuPipeline declaration
│   ├── detection_pipeline_cuda.hpp  # CUDA pipeline declaration
│   └── detector_node.hpp        # ROS2 node declaration
├── src/
│   ├── detection_pipeline.cpp   # All image processing logic (CPU)
│   ├── detection_pipeline_cuda.cu  # CUDA backend (optional build)
│   ├── detector_node.cpp        # ROS2 subscriber / publisher
│   └── main.cpp                 # Entry point
├── launch/
│   └── detector.launch.py       # Parameterised launch file
├── config/
│   └── detector_params.yaml     # Default parameter file
├── CMakeLists.txt
└── package.xml
```

---

## Prerequisites

| Dependency | Version |
|---|---|
| ROS 2 | Humble / Iron / Jazzy |
| OpenCV | ≥ 4.5 (with `dnn` module) |
| cv_bridge | from `ros-$ROS_DISTRO-cv-bridge` |
| image_transport | from `ros-$ROS_DISTRO-image-transport` |

Install system dependencies:

```bash
sudo apt install \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-image-transport \
  ros-$ROS_DISTRO-image-transport-plugins \
  libopencv-dev
```

### CUDA backend prerequisites (optional)

- NVIDIA driver + CUDA toolkit (with `nvcc` available to CMake).
- OpenCV build that includes:
  - DNN with CUDA support (`DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA`)
  - CUDA image modules (`cudafilters`, `cudaimgproc`, `cudawarping`)

If these are not available, the package still builds and runs with CPU only.

---

## Build

```bash
# 1. Place this package inside your ROS2 workspace
mkdir -p ~/ros2_ws/src
cp -r gpu_object_detection ~/ros2_ws/src/

# 2. Install rosdep dependencies
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y

# 3. Build
colcon build --packages-select gpu_object_detection --cmake-args -DCMAKE_BUILD_TYPE=Release

# 4. Source the workspace
source install/setup.bash
```

During configure, CMake prints whether the CUDA pipeline is built. If CUDA/OpenCV CUDA
components are missing, only `CpuPipeline` is compiled.

---

## Run

### Mode A – Canny fallback (no model files needed)

```bash
ros2 launch gpu_object_detection detector.launch.py \
    input_topic:=/camera/image_raw
```

### Mode B – YOLO on CPU

Download YOLOv4 weights and config:

```bash
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

Launch:

```bash
ros2 launch gpu_object_detection detector.launch.py \
    model_cfg:=$(pwd)/yolov4.cfg \
    model_weights:=$(pwd)/yolov4.weights \
    class_names:=$(pwd)/coco.names \
    conf_thresh:=0.4
```

### Mode C – YOLO with CUDA backend

```bash
ros2 launch gpu_object_detection detector.launch.py \
    model_cfg:=$(pwd)/yolov4.cfg \
    model_weights:=$(pwd)/yolov4.weights \
    class_names:=$(pwd)/coco.names \
    use_cuda:=true
```

Backend selection is runtime-configurable via parameter:

- `use_cuda:=false` (default) -> CPU pipeline
- `use_cuda:=true` -> CUDA pipeline if built/available, otherwise logs a warning and falls back to CPU

### With a webcam (v4l2)

```bash
# Terminal 1 – camera driver
ros2 run v4l2_camera v4l2_camera_node --ros-args -p video_device:=/dev/video0

# Terminal 2 – detector
ros2 launch gpu_object_detection detector.launch.py \
    input_topic:=/image_raw
```

### With a ROS2 bag file

```bash
# Terminal 1 – play bag
ros2 bag play /path/to/your.bag --loop

# Terminal 2 – detector (remap topic to match bag)
ros2 launch gpu_object_detection detector.launch.py \
    input_topic:=/your/bag/image_topic
```

### Visualise in RViz2

```bash
rviz2
# Add → By topic → /detector/image_annotated → Image
```

Or use `rqt_image_view`:

```bash
ros2 run rqt_image_view rqt_image_view /detector/image_annotated
```

---

## Monitor performance

```bash
# See the timing logs live
ros2 node info /detector_node

# Topic Hz
ros2 topic hz /detector/image_annotated

# Full timing output in the terminal where the node runs:
# [Frame 30] detections=3 | pre=0.4ms inf=42.1ms post=0.3ms total=42.8ms | rolling_fps=23.4
```

---

## CUDA backend notes

- `CudaPipeline::preprocess()` uploads each frame once to `cv::cuda::GpuMat`,
  does resize + BGR→RGB on GPU, then downloads once to create the final DNN blob.
- The final blob creation currently uses CPU `cv::dnn::blobFromImage` as a clean
  cross-version fallback because a consistent on-device blob API is not guaranteed.
- YOLO inference is configured with `DNN_BACKEND_CUDA` and `DNN_TARGET_CUDA`.
- If CUDA DNN support is missing at runtime, the node logs a clear warning and
  falls back to `CpuPipeline`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `cv_bridge not found` | `sudo apt install ros-$ROS_DISTRO-cv-bridge` |
| `image_transport not found` | `sudo apt install ros-$ROS_DISTRO-image-transport` |
| YOLO model loads but no detections | Lower `conf_thresh` to `0.3` |
| Very low FPS with YOLO | Use `input_width:=320 input_height:=320` |
| `Failed to load YOLO model` | Check absolute paths; `.cfg` and `.weights` must match |
