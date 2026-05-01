#!/usr/bin/env bash
# =============================================================================
# build_and_run.sh
#
# Build the package and run it in Canny fallback mode.
# Run this AFTER vm_setup.sh has completed.
# =============================================================================
#
# NOTE: -u (treat unbound variables as errors) must be OFF when sourcing any
#       ROS setup.bash because those scripts reference variables such as
#       AMENT_TRACE_SETUP_FILES before they are initialised, which causes an
#       immediate fatal error under set -u.
#       We use -eo pipefail here and temporarily lift -u around every source.
set -eo pipefail

ROS_DISTRO="humble"
WS="$HOME/ros2_ws"
PKG_DIR="$(cd "$(dirname "$0")" && pwd)/gpu_object_detection"

# ── Source ROS base (disable -u for the duration) ────────────────────────────
set +u
source /opt/ros/${ROS_DISTRO}/setup.bash
set -u

# ── Copy package into workspace ───────────────────────────────────────────────
echo "[1/3] Copying package to workspace..."
mkdir -p "$WS/src"
cp -r "$PKG_DIR" "$WS/src/"

# ── Install dependencies & build ─────────────────────────────────────────────
echo "[2/3] Building..."
cd "$WS"
rosdep install --from-paths src --ignore-src -r -y
colcon build \
  --packages-select gpu_object_detection \
  --cmake-args -DCMAKE_BUILD_TYPE=Release

# ── Source the install overlay (disable -u again) ────────────────────────────
set +u
source "$WS/install/setup.bash"
set -u

# ── Launch ────────────────────────────────────────────────────────────────────
echo "[3/3] Launching detector node (Canny fallback mode)..."
echo ""
echo "  Subscribe topic : /camera/image_raw"
echo "  Publish topic   : /detector/image_annotated"
echo ""
echo "  To view output in another terminal:"
echo "    source $WS/install/setup.bash"
echo "    ros2 run rqt_image_view rqt_image_view /detector/image_annotated"
echo ""

ros2 launch gpu_object_detection detector.launch.py
