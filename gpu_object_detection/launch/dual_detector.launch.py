"""
dual_detector.launch.py

Launches TWO detector_node instances from the same executable against the
same input topic, for a side-by-side CPU vs CUDA performance comparison.

Topic layout
────────────
  /camera/image_raw           ← shared input (bag replay or live camera)
  /detector_cpu/image_annotated   ← CPU pipeline output
  /detector_cuda/image_annotated  ← CUDA pipeline output (falls back to CPU
                                     if CUDA was not compiled in)

Usage
─────
  # Canny fallback comparison (no model files needed):
  ros2 launch gpu_object_detection dual_detector.launch.py

  # YOLO CPU vs CUDA comparison:
  ros2 launch gpu_object_detection dual_detector.launch.py \
      model_cfg:=/path/to/yolov4.cfg \
      model_weights:=/path/to/yolov4.weights \
      class_names:=/path/to/coco.names

  # Custom shared input topic (e.g. from a bag file):
  ros2 launch gpu_object_detection dual_detector.launch.py \
      input_topic:=/your/bag/image_topic

View both outputs in RViz or rqt_image_view:
  ros2 run rqt_image_view rqt_image_view /detector_cpu/image_annotated
  ros2 run rqt_image_view rqt_image_view /detector_cuda/image_annotated
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Shared launch arguments ────────────────────────────────────────────
    args = [
        # Both nodes subscribe to this same topic
        DeclareLaunchArgument("input_topic",
            default_value="/camera/image_raw",
            description="Shared input image topic"),

        # Model paths (identical for both nodes so results are comparable)
        DeclareLaunchArgument("model_cfg",
            default_value="",
            description="Path to YOLO .cfg  (empty = use Canny fallback)"),

        DeclareLaunchArgument("model_weights",
            default_value="",
            description="Path to YOLO .weights"),

        DeclareLaunchArgument("class_names",
            default_value="",
            description="Path to coco.names  (one label per line)"),

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

        DeclareLaunchArgument("fps_window",
            default_value="30",
            description="Rolling FPS window size"),
    ]

    # ── Shared model parameters (same for both nodes) ──────────────────────
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

    # ── CPU detector instance ──────────────────────────────────────────────
    cpu_node = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name="detector_cpu",
        output="screen",
        parameters=[{
            **shared_params,
            "output_topic": "/detector_cpu/image_annotated",
            "use_cuda":     "false",
        }],
    )

    # ── CUDA detector instance ─────────────────────────────────────────────
    # Falls back to CPU pipeline automatically if CUDA was not compiled in.
    cuda_node = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name="detector_cuda",
        output="screen",
        parameters=[{
            **shared_params,
            "output_topic": "/detector_cuda/image_annotated",
            "use_cuda":     "true",
        }],
    )

    return LaunchDescription(args + [cpu_node, cuda_node])
