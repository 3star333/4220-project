# GPU Accelerated Real-Time Object Detection

This repository contains a ROS 2 package for CPU and optional CUDA-backed object detection:

- **Package:** `gpu_object_detection`
- **Full documentation:** [`gpu_object_detection/README.md`](./gpu_object_detection/README.md)

## Quick Start

```bash
mkdir -p ~/ros2_ws/src
cp -r gpu_object_detection ~/ros2_ws/src/

cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select gpu_object_detection \
             --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

Then run:

```bash
ros2 launch gpu_object_detection detector.launch.py
```

For full setup, CUDA options, and comparison demo details, see:

[`gpu_object_detection/README.md`](./gpu_object_detection/README.md)
