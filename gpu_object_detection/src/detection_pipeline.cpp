/**
 * detection_pipeline.cpp
 *
 * CPU implementation of the detection pipeline.
 *
 * Modes
 * ─────
 *  YOLO mode  – activated when valid .cfg + .weights paths are provided.
 *               Uses OpenCV DNN (CPU backend) for YOLOv3/v4 inference.
 *
 *  Canny mode – fallback when no model files are given.
 *               Runs Gaussian blur → Canny edge detection and wraps each
 *               significant contour in a "Detection" struct so the rest of the
 *               pipeline works identically.
 *
 * CUDA extension point
 * ─────────────────────
 *  Replace the bodies of preprocess(), infer_yolo() / infer_canny(), and
 *  postprocess() with CUDA kernel launches in a derived CudaPipeline class.
 *  The node code is completely unaware of which concrete pipeline is active.
 */

#include "gpu_object_detection/detection_pipeline.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gpu_od
{

// ─────────────────────────────────────────────────────────────────────────────
// Construction
// ─────────────────────────────────────────────────────────────────────────────

CpuPipeline::CpuPipeline(const std::string & model_cfg,
                         const std::string & model_weights,
                         const std::string & class_names_path,
                         int   input_width,
                         int   input_height,
                         float conf_thresh,
                         float nms_thresh)
: input_w_(input_width),
  input_h_(input_height),
  conf_thresh_(conf_thresh),
  nms_thresh_(nms_thresh)
{
  // ── Load YOLO model ──────────────────────────────────────────────────────
  if (!model_cfg.empty() && !model_weights.empty()) {
    net_ = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
    if (net_.empty()) {
      throw std::runtime_error("Failed to load YOLO model from:\n  cfg:     "
                               + model_cfg + "\n  weights: " + model_weights);
    }
    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    use_yolo_ = true;
  }

  // ── Load class names ─────────────────────────────────────────────────────
  if (!class_names_path.empty()) {
    std::ifstream ifs(class_names_path);
    if (!ifs.is_open()) {
      throw std::runtime_error("Cannot open class names file: " + class_names_path);
    }
    std::string line;
    while (std::getline(ifs, line)) {
      if (!line.empty()) { class_names_.push_back(line); }
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public entry-point
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> CpuPipeline::run(cv::Mat & frame,
                                        PipelineTiming & timing)
{
  using Clock = std::chrono::steady_clock;

  const auto t_total_start = Clock::now();

  // ── Stage 1: Preprocess ──────────────────────────────────────────────────
  const auto t0 = Clock::now();
  cv::Mat blob = preprocess(frame);               // returns DNN blob or resized mat
  timing.preprocess_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  // ── Stage 2: Inference ───────────────────────────────────────────────────
  const auto t1 = Clock::now();
  std::vector<Detection> detections;
  if (use_yolo_) {
    detections = infer_yolo(blob, frame);
  } else {
    detections = infer_canny(frame);              // Canny works on the raw frame
  }
  timing.inference_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

  // ── Stage 3: Postprocess / overlay ───────────────────────────────────────
  timing.total_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();

  const auto t2 = Clock::now();
  postprocess(frame, detections, timing.total_ms);
  timing.postprocess_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

  // Recalculate total including overlay
  timing.total_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();

  return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 1 – Preprocess
// CUDA extension: upload frame to GPU here (cv::cuda::GpuMat), run resize /
//                 colour conversion as CUDA kernels, then create the DNN blob
//                 on-device with cv::cuda::dnn::blobFromImage().
// ─────────────────────────────────────────────────────────────────────────────

cv::Mat CpuPipeline::preprocess(const cv::Mat & frame)
{
  if (!use_yolo_) {
    // Canny path: just return original (we work on the frame directly)
    return frame;
  }

  // Create a 4-D blob from the frame for the DNN
  cv::Mat blob;
  cv::dnn::blobFromImage(frame,
                         blob,
                         1.0 / 255.0,              // scale
                         cv::Size(input_w_, input_h_),
                         cv::Scalar(0, 0, 0),
                         true,                      // swapRB (BGR→RGB)
                         false);                    // crop
  return blob;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2a – YOLO inference (OpenCV DNN, CPU)
// CUDA extension: net_.setPreferableBackend(DNN_BACKEND_CUDA);
//                 net_.setPreferableTarget(DNN_TARGET_CUDA);
//                 Everything else stays the same.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> CpuPipeline::infer_yolo(const cv::Mat & blob,
                                                const cv::Mat & original_frame)
{
  net_.setInput(blob);

  std::vector<cv::Mat> outs;
  net_.forward(outs, get_output_layer_names());

  const int frame_w = original_frame.cols;
  const int frame_h = original_frame.rows;

  std::vector<int>     class_ids;
  std::vector<float>   confidences;
  std::vector<cv::Rect> boxes;

  for (const auto & out : outs) {
    const auto * data = reinterpret_cast<const float *>(out.data);
    for (int i = 0; i < out.rows; ++i, data += out.cols) {
      // cols layout: [cx, cy, w, h, obj_conf, class_0, class_1, ...]
      const cv::Mat scores(1, out.cols - 5, CV_32F,
                           const_cast<float *>(data + 5));
      cv::Point class_id_pt;
      double    max_val{0.0};
      cv::minMaxLoc(scores, nullptr, &max_val, nullptr, &class_id_pt);

      const float confidence = static_cast<float>(max_val);
      if (confidence < conf_thresh_) { continue; }

      const int cx = static_cast<int>(data[0] * frame_w);
      const int cy = static_cast<int>(data[1] * frame_h);
      const int bw = static_cast<int>(data[2] * frame_w);
      const int bh = static_cast<int>(data[3] * frame_h);

      boxes.push_back({cx - bw / 2, cy - bh / 2, bw, bh});
      confidences.push_back(confidence);
      class_ids.push_back(class_id_pt.x);
    }
  }

  // Non-Maximum Suppression
  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, nms_indices);

  std::vector<Detection> detections;
  detections.reserve(nms_indices.size());

  for (int idx : nms_indices) {
    Detection det;
    det.class_id   = class_ids[idx];
    det.confidence = confidences[idx];
    det.bbox       = boxes[idx] & cv::Rect(0, 0, frame_w, frame_h); // clamp
    det.label      = (det.class_id < static_cast<int>(class_names_.size()))
                       ? class_names_[det.class_id]
                       : ("class_" + std::to_string(det.class_id));
    detections.push_back(det);
  }
  return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 2b – Canny edge detection (placeholder fallback)
// CUDA extension: replace with cv::cuda::createCannyEdgeDetector() which runs
//                 entirely on GPU memory with no CPU-GPU copies mid-pipeline.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> CpuPipeline::infer_canny(const cv::Mat & frame)
{
  cv::Mat gray, blurred, edges;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
  cv::Canny(blurred, edges, 50, 150);

  // Find contours and treat each significant one as a "detection"
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<Detection> detections;
  for (const auto & contour : contours) {
    if (cv::contourArea(contour) < 500.0) { continue; }  // noise filter

    Detection det;
    det.class_id   = 0;
    det.label      = "edge_region";
    det.confidence = 1.0f;
    det.bbox       = cv::boundingRect(contour);
    detections.push_back(det);
  }
  return detections;
}

// ─────────────────────────────────────────────────────────────────────────────
// Stage 3 – Postprocess / draw overlay
// CUDA extension: use cv::cuda::GpuMat-based drawing utilities or download
//                 results to CPU only here for final rendering.
// ─────────────────────────────────────────────────────────────────────────────

void CpuPipeline::postprocess(cv::Mat & frame,
                               const std::vector<Detection> & dets,
                               double total_ms)
{
  // Colour palette (cycles through 10 colours)
  static const std::vector<cv::Scalar> palette = {
    {  0, 255,   0}, {255,   0,   0}, {  0,   0, 255},
    {255, 255,   0}, {  0, 255, 255}, {255,   0, 255},
    {128, 255,   0}, {  0, 128, 255}, {255, 128,   0},
    {128,   0, 255}
  };

  for (const auto & det : dets) {
    const cv::Scalar & colour = palette[det.class_id % palette.size()];

    // Bounding box
    cv::rectangle(frame, det.bbox, colour, 2);

    // Label background + text
    const std::string label_str =
      det.label + " " + cv::format("%.2f", det.confidence);
    int baseline = 0;
    const cv::Size text_size =
      cv::getTextSize(label_str, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

    const cv::Point label_tl{det.bbox.x,
                             std::max(det.bbox.y - text_size.height - 4, 0)};
    cv::rectangle(frame,
                  label_tl,
                  label_tl + cv::Point(text_size.width, text_size.height + baseline + 2),
                  colour, cv::FILLED);
    cv::putText(frame, label_str,
                label_tl + cv::Point(0, text_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }

  // ── HUD: timing + FPS ────────────────────────────────────────────────────
  const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;
  const std::string hud =
    cv::format("Pipeline: %s | Latency: %.1f ms | FPS: %.1f",
               use_yolo_ ? "YOLO-CPU" : "Canny-CPU",
               total_ms, fps);

  cv::rectangle(frame,
                cv::Point(0, 0),
                cv::Point(frame.cols, 22),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, hud,
              cv::Point(5, 16),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 80), 1, cv::LINE_AA);

  // Detection count
  cv::putText(frame,
              "Detections: " + std::to_string(dets.size()),
              cv::Point(5, frame.rows - 8),
              cv::FONT_HERSHEY_SIMPLEX, 0.45,
              cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

std::vector<std::string> CpuPipeline::get_output_layer_names()
{
  static std::vector<std::string> names;
  if (names.empty()) {
    const auto & layer_names = net_.getLayerNames();
    const auto   unconnected = net_.getUnconnectedOutLayers();
    names.resize(unconnected.size());
    for (std::size_t i = 0; i < unconnected.size(); ++i) {
      names[i] = layer_names[unconnected[i] - 1];
    }
  }
  return names;
}

std::string CpuPipeline::elapsed_str(double ms)
{
  std::ostringstream oss;
  oss << std::fixed;
  oss.precision(2);
  oss << ms << " ms";
  return oss.str();
}

}  // namespace gpu_od
