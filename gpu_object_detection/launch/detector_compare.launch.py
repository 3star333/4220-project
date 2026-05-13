"""
detector_compare.launch.py

CPU vs CUDA side-by-side comparison demo.

Launches TWO detector_node instances against the same input topic:
  - detector_cpu  → /detector_cpu/image_annotated   (use_cuda=false)
  - detector_cuda → /detector_cuda/image_annotated  (use_cuda=true, auto-falls-back)

Optionally launches RViz2 with a pre-configured side-by-side layout.

──────────────────────────────────────────────────────────────
QUICK START (Canny fallback, no model files required)
──────────────────────────────────────────────────────────────

  # Play your bag in another terminal first, then:
  ros2 launch gpu_object_detection detector_compare.launch.py \
      input_topic:=/image_raw

──────────────────────────────────────────────────────────────
YOLO comparison
──────────────────────────────────────────────────────────────

  ros2 launch gpu_object_detection detector_compare.launch.py \
      input_topic:=/image_raw \
      model_cfg:=$HOME/yolo/yolov4.cfg \
      model_weights:=$HOME/yolo/yolov4.weights \
      class_names:=$HOME/yolo/coco.names

──────────────────────────────────────────────────────────────
LAUNCH WITH RViz side-by-side view
──────────────────────────────────────────────────────────────

  ros2 launch gpu_object_detection detector_compare.launch.py \
      input_topic:=/image_raw \
      launch_rviz:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    pkg_share = get_package_share_directory("gpu_object_detection")
    rviz_cfg  = os.path.join(pkg_share, "config", "compare.rviz")

    # ── Launch arguments ───────────────────────────────────────────────────
    args = [
        # ── Shared input ───────────────────────────────────────────────────
        DeclareLaunchArgument(
            "input_topic",
            default_value="/image_raw",
            description="Shared input image topic for both detectors "
                        "(must match your bag or camera driver output)"),

        # ── Model files (leave empty for Canny fallback) ───────────────────
        DeclareLaunchArgument(
            "model_cfg",
            default_value="",
            description="Path to YOLO .cfg  (empty → Canny fallback)"),

        DeclareLaunchArgument(
            "model_weights",
            default_value="",
            description="Path to YOLO .weights"),

        DeclareLaunchArgument(
            "class_names",
            default_value="",
            description="Path to coco.names"),

        # ── Detection tuning ───────────────────────────────────────────────
        DeclareLaunchArgument(
            "conf_thresh",
            default_value="0.5",
            description="Confidence threshold"),

        DeclareLaunchArgument(
            "nms_thresh",
            default_value="0.4",
            description="NMS IoU threshold"),

        DeclareLaunchArgument(
            "input_width",
            default_value="416",
            description="DNN blob width (use 320 for faster CPU inference)"),

        DeclareLaunchArgument(
            "input_height",
            default_value="416",
            description="DNN blob height"),

        DeclareLaunchArgument(
            "fps_window",
            default_value="30",
            description="Rolling FPS window size"),

        # ── RViz ──────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            "launch_rviz",
            default_value="false",
            description="Set true to automatically open RViz2 with the "
                        "side-by-side comparison layout"),
    ]

    # ── Parameters shared by both nodes ───────────────────────────────────
    shared_params = {
        "input_topic":   LaunchConfiguration("input_topic"),
        "model_cfg":     LaunchConfiguration("model_cfg"),
        "model_weights": LaunchConfiguration("model_weights"),
        "class_names":   LaunchConfiguration("class_names"),
        "conf_thresh":   LaunchConfiguration("conf_thresh"),
        "nms_thresh":    LaunchConfiguration("nms_thresh"),
        "input_width":   LaunchConfiguration("input_width"),
        "input_height":  LaunchConfiguration("input_height"),
        "fps_window":    LaunchConfiguration("fps_window"),
    }

    # ── CPU detector ──────────────────────────────────────────────────────
    cpu_node = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name="detector_cpu",
        namespace="detector_cpu",
        output="screen",
        parameters=[{
            **shared_params,
            "output_topic": "/detector_cpu/image_annotated",
            "use_cuda":     False,
        }],
    )

    # ── CUDA detector ─────────────────────────────────────────────────────
    # Automatically falls back to CPU if CUDA was not compiled in.
    cuda_node = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name="detector_cuda",
        namespace="detector_cuda",
        output="screen",
        parameters=[{
            **shared_params,
            "output_topic": "/detector_cuda/image_annotated",
            "use_cuda":     True,
        }],
    )

    # ── RViz2 (optional) ──────────────────────────────────────────────────
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_cfg],
        output="screen",
        condition=IfCondition(LaunchConfiguration("launch_rviz")),
    )

    return LaunchDescription(args + [cpu_node, cuda_node, rviz_node])
