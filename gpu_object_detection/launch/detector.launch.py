"""
detector.launch.py

Launches the detector_node with parameters.
All parameters can be overridden from the command line, e.g.:

  ros2 launch gpu_object_detection detector.launch.py \
      model_cfg:=/path/to/yolov4.cfg \
      model_weights:=/path/to/yolov4.weights \
      class_names:=/path/to/coco.names \
      input_topic:=/camera/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ── Declare overridable launch arguments ───────────────────────────────
    args = [
        DeclareLaunchArgument("input_topic",
            default_value="/camera/image_raw",
            description="Input image topic"),

        DeclareLaunchArgument("output_topic",
            default_value="/detector/image_annotated",
            description="Annotated output image topic"),

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
            description="Number of frames to average for FPS calculation"),
    ]

    # ── Node ───────────────────────────────────────────────────────────────
    detector = Node(
        package="gpu_object_detection",
        executable="detector_node",
        name="detector_node",
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
        }],
        # Remappings allow zero-config use with any camera driver
        remappings=[
            ("/camera/image_raw", LaunchConfiguration("input_topic")),
        ],
    )

    return LaunchDescription(args + [detector])
