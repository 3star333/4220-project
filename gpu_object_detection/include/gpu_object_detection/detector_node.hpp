#pragma once

/**
 * detector_node.hpp
 *
 * ROS2 node that:
 *   • Subscribes to a sensor_msgs/msg/Image topic
 *   • Converts frames via cv_bridge
 *   • Passes each frame through the DetectionPipeline
 *   • Publishes the annotated image to an output topic
 *   • Logs per-frame latency and a rolling FPS average
 *
 * The node owns an IPipeline* so the concrete backend (CPU or CUDA) is fully
 * swappable at construction time – the node itself never changes.
 */

#include "gpu_object_detection/detection_pipeline.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>

#include <memory>
#include <deque>

namespace gpu_od
{

class DetectorNode : public rclcpp::Node
{
public:
  explicit DetectorNode(const rclcpp::NodeOptions & options =
                          rclcpp::NodeOptions());

private:
  // ── ROS2 callbacks ──────────────────────────────────────────────────────────
  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);

  // ── Parameters ──────────────────────────────────────────────────────────────
  void declare_and_load_params();

  // ── FPS rolling average ─────────────────────────────────────────────────────
  void update_fps(double frame_ms);
  double rolling_fps() const;

  // ── Pipeline backend (CPU now, CUDA later) ──────────────────────────────────
  std::unique_ptr<IPipeline> pipeline_;

  // ── Image transport ─────────────────────────────────────────────────────────
  image_transport::Subscriber image_sub_;
  image_transport::Publisher  image_pub_;

  // ── Params (loaded from ROS2 parameter server) ───────────────────────────────
  std::string p_input_topic_;
  std::string p_output_topic_;
  std::string p_model_cfg_;
  std::string p_model_weights_;
  std::string p_class_names_;
  int         p_input_w_{416};
  int         p_input_h_{416};
  double      p_conf_thresh_{0.5};
  double      p_nms_thresh_{0.4};
  int         p_fps_window_{30};   // rolling FPS window size

  // ── Diagnostics ──────────────────────────────────────────────────────────────
  std::deque<double> frame_times_ms_;   // ring buffer for FPS calculation
  uint64_t           frame_count_{0};
};

}  // namespace gpu_od
