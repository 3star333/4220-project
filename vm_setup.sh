#!/usr/bin/env bash
# =============================================================================
# vm_setup.sh
#
# One-shot setup script for Ubuntu 22.04 VM (no CUDA, no GPU required).
# Run this script ONCE after a fresh Ubuntu 22.04 install:
#
#   chmod +x vm_setup.sh
#   ./vm_setup.sh
#
# Then follow the "Build & Run" steps at the bottom of this file.
# =============================================================================
set -euo pipefail

ROS_DISTRO="humble"
WS="$HOME/ros2_ws"

echo "================================================================"
echo "  GPU Object Detection — VM Setup (CPU-only, Ubuntu 22.04)"
echo "================================================================"

# ── Step 1: System update ─────────────────────────────────────────────────────
echo "[1/7] Updating system packages..."
sudo apt update && sudo apt upgrade -y

# ── Step 2: Locale ────────────────────────────────────────────────────────────
echo "[2/7] Configuring locale..."
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ── Step 3: ROS 2 Humble ──────────────────────────────────────────────────────
echo "[3/7] Installing ROS 2 Humble..."
sudo apt install -y software-properties-common curl
sudo add-apt-repository universe -y

curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
  signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu jammy main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y \
  ros-${ROS_DISTRO}-desktop \
  ros-${ROS_DISTRO}-cv-bridge \
  ros-${ROS_DISTRO}-image-transport \
  ros-${ROS_DISTRO}-image-transport-plugins \
  ros-${ROS_DISTRO}-rqt-image-view \
  ros-${ROS_DISTRO}-usb-cam

# ── Step 4: Build tools ───────────────────────────────────────────────────────
echo "[4/7] Installing build tools..."
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  build-essential \
  cmake \
  git

# ── Step 5: OpenCV (with DNN module) ─────────────────────────────────────────
echo "[5/7] Installing OpenCV..."
sudo apt install -y \
  libopencv-dev \
  python3-opencv

echo "OpenCV version: $(python3 -c 'import cv2; print(cv2.__version__)')"

# ── Step 6: rosdep init ───────────────────────────────────────────────────────
echo "[6/7] Initialising rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init
fi
rosdep update

# ── Step 7: Source ROS 2 in .bashrc ──────────────────────────────────────────
echo "[7/7] Adding ROS 2 source to ~/.bashrc..."
BASHRC_LINE="source /opt/ros/${ROS_DISTRO}/setup.bash"
if ! grep -qF "$BASHRC_LINE" "$HOME/.bashrc"; then
  echo "$BASHRC_LINE" >> "$HOME/.bashrc"
fi
source /opt/ros/${ROS_DISTRO}/setup.bash

# ── Create workspace ──────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Setting up ROS2 workspace at $WS"
echo "================================================================"
mkdir -p "$WS/src"

echo ""
echo "================================================================"
echo "  Setup complete!  Next steps:"
echo "================================================================"
echo ""
echo "  1. Copy the package into the workspace:"
echo "     cp -r gpu_object_detection $WS/src/"
echo ""
echo "  2. Build:"
echo "     cd $WS"
echo "     rosdep install --from-paths src --ignore-src -r -y"
echo "     colcon build --packages-select gpu_object_detection \\"
echo "                  --cmake-args -DCMAKE_BUILD_TYPE=Release"
echo "     source install/setup.bash"
echo ""
echo "  3a. Run with Canny fallback (no model files needed):"
echo "     ros2 launch gpu_object_detection detector.launch.py"
echo ""
echo "  3b. Run with YOLO on CPU:"
echo "     bash $(dirname "$0")/download_yolo.sh        # see below"
echo "     ros2 launch gpu_object_detection detector.launch.py \\"
echo "         model_cfg:=\$HOME/yolo/yolov4.cfg \\"
echo "         model_weights:=\$HOME/yolo/yolov4.weights \\"
echo "         class_names:=\$HOME/yolo/coco.names"
echo ""
echo "  4. View output:"
echo "     ros2 run rqt_image_view rqt_image_view /detector/image_annotated"
echo ""
