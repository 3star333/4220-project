#!/usr/bin/env bash
# =============================================================================
# download_yolo.sh
#
# Downloads YOLOv4 weights, config, and COCO class names into ~/yolo/
# Run this inside the VM if you want real YOLO detections.
# The Canny fallback works without any of these files.
# =============================================================================
set -euo pipefail

YOLO_DIR="$HOME/yolo"
mkdir -p "$YOLO_DIR"
cd "$YOLO_DIR"

echo "Downloading YOLOv4 config..."
curl -L -o yolov4.cfg \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg"

echo "Downloading COCO class names..."
curl -L -o coco.names \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"

echo "Downloading YOLOv4 weights (~245 MB) — this may take a while..."
curl -L -o yolov4.weights \
  "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

echo ""
echo "Done! Files saved to $YOLO_DIR:"
ls -lh "$YOLO_DIR"
echo ""
echo "Launch with:"
echo "  ros2 launch gpu_object_detection detector.launch.py \\"
echo "      model_cfg:=$YOLO_DIR/yolov4.cfg \\"
echo "      model_weights:=$YOLO_DIR/yolov4.weights \\"
echo "      class_names:=$YOLO_DIR/coco.names"
