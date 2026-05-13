# GPU Accelerated Real-Time Object Detection
## ROS2 + OpenCV · CPU Baseline + CUDA Backend

---

## Project Structure

```
gpu_object_detection/
├── include/gpu_object_detection/
│   ├── detection_pipeline.hpp       # IPipeline interface + CpuPipeline declaration
│   ├── detection_pipeline_cuda.hpp  # CUDA pipeline declaration (optional build)
│   └── detector_node.hpp            # ROS2 node declaration
├── src/
│   ├── detection_pipeline.cpp       # CPU image processing (preprocess/infer/postprocess)
│   ├── detection_pipeline_cuda.cu   # CUDA backend (optional build)
│   ├── detector_node.cpp            # ROS2 subscriber / publisher / timing
│   └── main.cpp                     # Entry point
├── launch/
│   ├── detector.launch.py           # Single-instance launch (CPU or CUDA)
│   ├── dual_detector.launch.py      # Two instances, minimal args
│   └── detector_compare.launch.py   # Full CPU vs CUDA comparison demo
├── config/
│   ├── detector_params.yaml         # Default parameter values
│   └── compare.rviz                 # RViz side-by-side layout
├── CMakeLists.txt
└── package.xml
```

---

## Prerequisites

| Dependency | Version |
|---|---|
| ROS 2 | Humble (Ubuntu 22.04) |
| OpenCV | >= 4.5 (with `dnn` module) |
| cv_bridge | `ros-$ROS_DISTRO-cv-bridge` |
| image_transport | `ros-$ROS_DISTRO-image-transport` |

```bash
sudo apt install \
  ros-$ROS_DISTRO-cv-bridge \
  ros-$ROS_DISTRO-image-transport \
  ros-$ROS_DISTRO-image-transport-plugins \
  ros-$ROS_DISTRO-v4l2-camera \
  ros-$ROS_DISTRO-rviz2 \
  libopencv-dev
```

### CUDA backend prerequisites (optional)

- NVIDIA driver + CUDA toolkit (with `nvcc` on `PATH`).
- OpenCV built with CUDA DNN support (`DNN_BACKEND_CUDA` / `DNN_TARGET_CUDA`).

If unavailable, the package builds CPU-only and `use_cuda:=true` logs a warning then falls
back to the CPU pipeline automatically.

---

## Build

```bash
mkdir -p ~/ros2_ws/src
cp -r gpu_object_detection ~/ros2_ws/src/

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gpu_object_detection \
             --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

CMake prints whether the CUDA pipeline was compiled. CPU-only is the safe default.

---

## Single-Node Usage

### Canny fallback (no model files needed)

```bash
ros2 launch gpu_object_detection detector.launch.py
```

### YOLO on CPU

```bash
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names

ros2 launch gpu_object_detection detector.launch.py \
    model_cfg:=$(pwd)/yolov4.cfg \
    model_weights:=$(pwd)/yolov4.weights \
    class_names:=$(pwd)/coco.names \
    conf_thresh:=0.4
```

### YOLO with CUDA backend

```bash
ros2 launch gpu_object_detection detector.launch.py \
    model_cfg:=$(pwd)/yolov4.cfg \
    model_weights:=$(pwd)/yolov4.weights \
    class_names:=$(pwd)/coco.names \
    use_cuda:=true
```

---

## CPU vs CUDA Comparison Demo

Full demo workflow. Open **5 terminals** inside the VM, each sourced:

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
```

---

### Step 0 — Download YOLO model files (once)

```bash
mkdir -p $HOME/yolo && cd $HOME/yolo
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names
```

> Skip to use Canny edge detection — no model files required.

---

### Phase A — Record a bag (once; reuse for all demo runs)

**Terminal 1 — Start webcam**

```bash
ros2 run v4l2_camera v4l2_camera_node \
    --ros-args -p video_device:=/dev/video0 \
               -p image_size:=[640,480] \
               -p camera_frame_id:=camera
```

**Verify topic is live:**

```bash
ros2 topic hz /image_raw
# Expected: ~30 Hz
```

**Terminal 2 — Record ~30 seconds**

```bash
mkdir -p $HOME/bags
ros2 bag record /image_raw \
    --output $HOME/bags/demo_recording \
    --max-bag-duration 30
# Ctrl+C after ~30 s
```

---

### Phase B — Replay + comparison (the actual demo)

**Terminal 1 — Replay the bag on loop**

```bash
ros2 bag play $HOME/bags/demo_recording --loop
```

**Terminal 2 — Launch both CPU and CUDA detectors**

```bash
# Canny fallback (no model files required):
ros2 launch gpu_object_detection detector_compare.launch.py \
    input_topic:=/image_raw

# OR with YOLO:
ros2 launch gpu_object_detection detector_compare.launch.py \
    input_topic:=/image_raw \
    model_cfg:=$HOME/yolo/yolov4.cfg \
    model_weights:=$HOME/yolo/yolov4.weights \
    class_names:=$HOME/yolo/coco.names
```

**Terminal 3 — Open RViz with the pre-configured side-by-side layout**

```bash
rviz2 -d ~/ros2_ws/install/gpu_object_detection/share/gpu_object_detection/config/compare.rviz
```

> Or add `launch_rviz:=true` to the Terminal 2 command to open RViz automatically.

**Terminal 4 — Watch live timing logs**

```bash
ros2 topic echo /rosout | grep "detector_cpu\|detector_cuda"
```

**Terminal 5 — Check topic rates**

```bash
ros2 topic hz /detector_cpu/image_annotated
ros2 topic hz /detector_cuda/image_annotated
```

---

### Topic Map

```
/image_raw  (bag replay / live camera — shared input)
      │
      ├──> [detector_cpu]  --> /detector_cpu/image_annotated
      └──> [detector_cuda] --> /detector_cuda/image_annotated
                                        |
                                        └──> RViz (compare.rviz)
```

---

### Performance Log Format

Both nodes print timing every `fps_window` frames:

```
[detector_cpu] [Frame 30]  detections=3 | pre=0.4ms inf=118ms post=0.3ms total=119ms | rolling_fps=8.4
[detector_cuda][Frame 30]  detections=3 | pre=0.2ms inf=14ms  post=0.2ms total=15ms  | rolling_fps=67.1
```

---

### Demo Troubleshooting

| Problem | Fix |
|---|---|
| No image in RViz | Check bag is playing: `ros2 topic hz /image_raw` |
| `detector_cuda` same FPS as CPU | CUDA pipeline not compiled — check CMake output |
| 0 detections on both nodes | Lower `conf_thresh:=0.3`, or drop model paths for Canny |
| `v4l2_camera` not found | `sudo apt install ros-humble-v4l2-camera` |
| `/dev/video0` missing | Run `ls /dev/video*`; adjust `-p video_device:=` |
| Low CPU FPS | Use `input_width:=320 input_height:=320` |

---

## General Troubleshooting

| Problem | Fix |
|---|---|
| `cv_bridge not found` | `sudo apt install ros-$ROS_DISTRO-cv-bridge` |
| `image_transport not found` | `sudo apt install ros-$ROS_DISTRO-image-transport` |
| YOLO model loads but no detections | Lower `conf_thresh` to `0.3` |
| `Failed to load YOLO model` | Check absolute paths; `.cfg` and `.weights` must match |

---

## CUDA Backend Notes

- Preprocessing: each frame is uploaded once to `cv::cuda::GpuMat`, resized and
  colour-converted on GPU, then the DNN blob is created on CPU as a stable cross-version fallback.
- Inference: `DNN_BACKEND_CUDA` + `DNN_TARGET_CUDA`.
- Runtime fallback: if CUDA is unavailable the node logs a warning and uses `CpuPipeline`.
