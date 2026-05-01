/**
 * detector_node.cpp
 *
 * ROS2 node implementation.
 *
 * Responsibilities
 * ────────────────
 *  1. Declare + load ROS2 parameters (topic names, model paths, thresholds).
 *  2. Instantiate the concrete pipeline (CpuPipeline here; swap for CudaPipeline later).
 *  3. Subscribe to the input image topic via image_transport.
 *  4. On each frame: convert → pipeline::run() → publish annotated image.
 *  5. Log per-frame timing and rolling FPS every 30 frames.
 */

#include "gpu_object_detection/detector_node.hpp"

#include <cv_bridge/cv_bridge.h>
#include <rclcpp/rclcpp.hpp>

#include <chrono>

namespace gpu_od
{

// ─────────────────────────────────────────────────────────────────────────────
// Constructor
// ─────────────────────────────────────────────────────────────────────────────

DetectorNode::DetectorNode(const rclcpp::NodeOptions & options)
: Node("detector_node", options)
{
  declare_and_load_params();

  // ── Build pipeline backend ────────────────────────────────────────────────
  //    CUDA extension: replace this line with:
  //      pipeline_ = std::make_unique<CudaPipeline>(...);
  //    Everything below this point stays untouched.
  pipeline_ = std::make_unique<CpuPipeline>(
    p_model_cfg_,
    p_model_weights_,
    p_class_names_,
    p_input_w_,
    p_input_h_,
    static_cast<float>(p_conf_thresh_),
    static_cast<float>(p_nms_thresh_)
  );

  RCLCPP_INFO(get_logger(), "Pipeline backend: %s", pipeline_->name().c_str());

  // ── image_transport setup ─────────────────────────────────────────────────
  auto it = image_transport::create_image_transport(
    std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {}));

  image_sub_ = it.subscribe(
    p_input_topic_, 1,
    std::bind(&DetectorNode::image_callback, this, std::placeholders::_1));

  image_pub_ = it.advertise(p_output_topic_, 1);

  RCLCPP_INFO(get_logger(),
              "DetectorNode ready\n  input  : %s\n  output : %s",
              p_input_topic_.c_str(), p_output_topic_.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
// Parameters
// ─────────────────────────────────────────────────────────────────────────────

void DetectorNode::declare_and_load_params()
{
  // Topic names
  declare_parameter("input_topic",   "/camera/image_raw");
  declare_parameter("output_topic",  "/detector/image_annotated");

  // YOLO model paths (leave empty to use Canny fallback)
  declare_parameter("model_cfg",     "");
  declare_parameter("model_weights", "");
  declare_parameter("class_names",   "");

  // DNN / detection settings
  declare_parameter("input_width",   416);
  declare_parameter("input_height",  416);
  declare_parameter("conf_thresh",   0.5);
  declare_parameter("nms_thresh",    0.4);

  // Diagnostics
  declare_parameter("fps_window",    30);

  // Load
  p_input_topic_   = get_parameter("input_topic").as_string();
  p_output_topic_  = get_parameter("output_topic").as_string();
  p_model_cfg_     = get_parameter("model_cfg").as_string();
  p_model_weights_ = get_parameter("model_weights").as_string();
  p_class_names_   = get_parameter("class_names").as_string();
  p_input_w_       = get_parameter("input_width").as_int();
  p_input_h_       = get_parameter("input_height").as_int();
  p_conf_thresh_   = get_parameter("conf_thresh").as_double();
  p_nms_thresh_    = get_parameter("nms_thresh").as_double();
  p_fps_window_    = get_parameter("fps_window").as_int();
}

// ─────────────────────────────────────────────────────────────────────────────
// Image callback  (called once per received frame)
// ─────────────────────────────────────────────────────────────────────────────

void DetectorNode::image_callback(
  const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  // ── 1. Convert ROS2 Image → OpenCV Mat ───────────────────────────────────
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat & frame = cv_ptr->image;
  if (frame.empty()) {
    RCLCPP_WARN(get_logger(), "Received empty frame, skipping.");
    return;
  }

  // ── 2. Run pipeline (preprocess → infer → postprocess) ────────────────────
  PipelineTiming timing;
  const auto detections = pipeline_->run(frame, timing);

  // ── 3. Update rolling FPS ─────────────────────────────────────────────────
  update_fps(timing.total_ms);
  ++frame_count_;

  // ── 4. Log every fps_window_ frames ───────────────────────────────────────
  if (frame_count_ % static_cast<uint64_t>(p_fps_window_) == 0) {
    RCLCPP_INFO(get_logger(),
                "[Frame %lu] detections=%zu | pre=%.1fms inf=%.1fms "
                "post=%.1fms total=%.1fms | rolling_fps=%.1f",
                frame_count_,
                detections.size(),
                timing.preprocess_ms,
                timing.inference_ms,
                timing.postprocess_ms,
                timing.total_ms,
                rolling_fps());
  }

  // ── 5. Publish annotated image ────────────────────────────────────────────
  cv_ptr->header = msg->header;   // preserve original timestamp + frame_id
  image_pub_.publish(cv_ptr->toImageMsg());
}

// ─────────────────────────────────────────────────────────────────────────────
// Rolling FPS helpers
// ─────────────────────────────────────────────────────────────────────────────

void DetectorNode::update_fps(double frame_ms)
{
  frame_times_ms_.push_back(frame_ms);
  while (static_cast<int>(frame_times_ms_.size()) > p_fps_window_) {
    frame_times_ms_.pop_front();
  }
}

double DetectorNode::rolling_fps() const
{
  if (frame_times_ms_.empty()) { return 0.0; }
  const double avg_ms =
    std::accumulate(frame_times_ms_.begin(), frame_times_ms_.end(), 0.0)
    / static_cast<double>(frame_times_ms_.size());
  return (avg_ms > 0.0) ? (1000.0 / avg_ms) : 0.0;
}

}  // namespace gpu_od
