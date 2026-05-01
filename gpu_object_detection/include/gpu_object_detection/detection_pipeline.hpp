#pragma once

/**
 * detection_pipeline.hpp
 *
 * Defines the DetectionPipeline class responsible for all image processing.
 * This is intentionally decoupled from ROS2 so that the processing logic
 * can later be swapped for a CUDA-accelerated version without touching the node.
 *
 * Stage map:
 *   preprocess()  → resize + colour conversion
 *   detect()      → CPU inference (OpenCV DNN / YOLO or Canny fallback)
 *   postprocess() → draw bounding boxes / overlay results
 *
 * To add CUDA later:
 *   1. Create detection_pipeline_cuda.hpp/.cu that inherits IPipeline.
 *   2. Replace the concrete type in the node via a compile-time flag or plugin.
 */

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>
#include <chrono>

namespace gpu_od
{

// ── Detection result ─────────────────────────────────────────────────────────
struct Detection
{
  int         class_id{-1};
  std::string label{"unknown"};
  float       confidence{0.0f};
  cv::Rect    bbox{};
};

// ── Timing snapshot ───────────────────────────────────────────────────────────
struct PipelineTiming
{
  double preprocess_ms{0.0};
  double inference_ms{0.0};
  double postprocess_ms{0.0};
  double total_ms{0.0};
};

// ── Abstract interface (swap CPU ↔ CUDA by implementing this) ─────────────────
class IPipeline
{
public:
  virtual ~IPipeline() = default;

  /**
   * Run the full pipeline on one frame.
   * @param frame  BGR frame from cv_bridge (modified in-place with overlays)
   * @param timing Output timing breakdown
   * @return       List of detections found in this frame
   */
  virtual std::vector<Detection> run(cv::Mat & frame,
                                     PipelineTiming & timing) = 0;

  /// Human-readable name shown in ROS2 log messages
  virtual std::string name() const = 0;
};

// ── CPU pipeline ──────────────────────────────────────────────────────────────
class CpuPipeline : public IPipeline
{
public:
  /**
   * @param model_cfg    Path to YOLO .cfg file  (empty → use Canny fallback)
   * @param model_weights Path to YOLO .weights  (empty → use Canny fallback)
   * @param class_names  Path to coco.names      (empty → use generic labels)
   * @param input_width  DNN blob width
   * @param input_height DNN blob height
   * @param conf_thresh  Confidence threshold
   * @param nms_thresh   NMS threshold
   */
  CpuPipeline(const std::string & model_cfg      = "",
              const std::string & model_weights   = "",
              const std::string & class_names     = "",
              int   input_width                   = 416,
              int   input_height                  = 416,
              float conf_thresh                   = 0.5f,
              float nms_thresh                    = 0.4f);

  std::vector<Detection> run(cv::Mat & frame,
                             PipelineTiming & timing) override;

  std::string name() const override { return "CpuPipeline"; }

private:
  // ── Stages ──────────────────────────────────────────────────────────────────
  cv::Mat                preprocess(const cv::Mat & frame);
  std::vector<Detection> infer_yolo(const cv::Mat & blob,
                                    const cv::Mat & original_frame);
  std::vector<Detection> infer_canny(const cv::Mat & frame);
  void                   postprocess(cv::Mat & frame,
                                     const std::vector<Detection> & dets,
                                     double total_ms);

  // ── Helpers ─────────────────────────────────────────────────────────────────
  std::vector<std::string>  get_output_layer_names();
  static std::string        elapsed_str(double ms);

  // ── State ────────────────────────────────────────────────────────────────────
  cv::dnn::Net              net_;
  std::vector<std::string>  class_names_;
  bool                      use_yolo_{false};

  int   input_w_;
  int   input_h_;
  float conf_thresh_;
  float nms_thresh_;
};

}  // namespace gpu_od
