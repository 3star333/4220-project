"""
detector.launch.py

Launches a single detector_node instance with configurable parameters.
All arguments can be overridden from the command line.

Single-instance examples
────────────────────────
  # Canny fallback (no model files needed):
  ros2 launch gpu_object_detection detector.launch.py

  # YOLO on CPU:
  ros2 launch gpu_object_detection detector.launch.py \
      model_cfg:=/path/to/yolov4.cfg \
      model_weights:=/path/to/yolov4.weights \
      class_names:=/path/to/coco.names

  # CUDA backend with a custom node name and output topic:
  ros2 launch gpu_object_detection detector.launch.py \
      node_name:=detector_cuda \
      output_topic:=/detector_cuda/image_annotated \
      use_cuda:=true

For the side-by-side CPU vs CUDA demo use dual_detector.launch.py instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Declare overridable launch arguments ───────────────────────────────
    args = [
        # ── Instance identity ──────────────────────────────────────────────
        DeclareLaunchArgument("node_name",
            default_value="detector_node",
            description="ROS2 node name (change to avoid collisions when "
                        "running two instances simultaneously)"),

        # ── Topics ─────────────────────────────────────────────────────────
        DeclareLaunchArgument("input_topic",
            default_value="/camera/image_raw",
            description="Input image topic (shared between instances)"),

        DeclareLaunchArgument("output_topic",
            default_value="/detector/image_annotated",
            description="Annotated output image topic "
                        "(must be unique per instance)"),

        # ── Model paths ────────────────────────────────────────────────────
        DeclareLaunchArgument("model_cfg",
            default_value="",
            description="Path to YOLO .cfg  (empty = use Canny fallback)"),

        DeclareLaunchArgument("model_weights",
            default_value="",
            description="Path to YOLO .weights"),

        DeclareLaunchArgument("class_names",
            default_value="",
            description="Path to coco.names  (one label per line)"),

        # ── Detection settings ─────────────────────────────────────────────
        DeclareLaunchArgument("conf_thresh",
            default_value="0.5",
            description="Detection confidence threshold"),

        DeclareLaunchArgument("nms_thresh",
            default_value="0.4",
            description="Non-Maximum Suppression IoU threshold"),

        DeclareLaunchArgument("input_width",
            default_value="416",
            description="DNN blob width"),

        DeclareLaunchArgument("input_height",
            default_value="416",
            description="DNN blob height"),

        # ── Diagnostics ────────────────────────────────────────────────────
        DeclareLaunchArgument("fps_window",
            default_value="30",
            description="Number of frames to average for FPS calculation"),

        # ── Backend ────────────────────────────────────────────────────────
        DeclareLaunchArgument("use_cuda",
            default_value="false",
            description="Enable CUDA backend (falls back to CPU if unavailable)"),
    ]

    # ── Node ───────────────────────────────────────────────────────────────
    detector = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name=LaunchConfiguration("node_name"),
        output="screen",
        parameters=[{
            "input_topic":   LaunchConfiguration("input_topic"),
            "output_topic":  LaunchConfiguration("output_topic"),
            "model_cfg":     LaunchConfiguration("model_cfg"),
            "model_weights": LaunchConfiguration("model_weights"),
            "class_names":   LaunchConfiguration("class_names"),
            "conf_thresh":   LaunchConfiguration("conf_thresh"),
            "nms_thresh":    LaunchConfiguration("nms_thresh"),
            "input_width":   LaunchConfiguration("input_width"),
            "input_height":  LaunchConfiguration("input_height"),
            "fps_window":    LaunchConfiguration("fps_window"),
            "use_cuda":      LaunchConfiguration("use_cuda"),
        }],
    )

    return LaunchDescription(args + [detector])
