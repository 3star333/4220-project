#include "gpu_object_detection/detection_pipeline_cuda.hpp"

#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpu_od
{

CudaPipeline::CudaPipeline(const std::string & model_cfg,
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
  if (!model_cfg.empty() && !model_weights.empty()) {
    net_ = cv::dnn::readNetFromDarknet(model_cfg, model_weights);
    if (net_.empty()) {
      throw std::runtime_error("Failed to load YOLO model from:\n  cfg:     "
                               + model_cfg + "\n  weights: " + model_weights);
    }

    try {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } catch (const cv::Exception & e) {
      throw std::runtime_error(
              "CUDA backend requested but OpenCV DNN CUDA is not available: "
              + std::string(e.what()));
    }
    use_yolo_ = true;
  }

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

std::vector<Detection> CudaPipeline::run(cv::Mat & frame,
                                         PipelineTiming & timing)
{
  using Clock = std::chrono::steady_clock;

  const auto t_total_start = Clock::now();

  const auto t0 = Clock::now();
  cv::Mat blob = preprocess(frame);
  timing.preprocess_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

  const auto t1 = Clock::now();
  std::vector<Detection> detections;
  if (use_yolo_) {
    detections = infer_yolo(blob, frame);
  } else {
    detections = infer_canny(frame);
  }
  timing.inference_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t1).count();

  timing.total_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();

  const auto t2 = Clock::now();
  postprocess(frame, detections, timing.total_ms);
  timing.postprocess_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t2).count();

  timing.total_ms =
    std::chrono::duration<double, std::milli>(Clock::now() - t_total_start).count();

  return detections;
}

cv::Mat CudaPipeline::preprocess(const cv::Mat & frame)
{
  if (!use_yolo_) {
    return frame;
  }

  cv::cuda::GpuMat gpu_bgr;
  gpu_bgr.upload(frame);

  cv::cuda::GpuMat gpu_resized;
  cv::cuda::resize(gpu_bgr, gpu_resized, cv::Size(input_w_, input_h_));

  cv::cuda::GpuMat gpu_rgb;
  cv::cuda::cvtColor(gpu_resized, gpu_rgb, cv::COLOR_BGR2RGB);

  // OpenCV DNN does not expose a stable cross-version API to build a blob
  // directly from cv::cuda::GpuMat, so we download the GPU-preprocessed frame
  // once and build the final 4-D blob on CPU.
  cv::Mat rgb_cpu;
  gpu_rgb.download(rgb_cpu);

  cv::Mat blob;
  cv::dnn::blobFromImage(rgb_cpu,
                         blob,
                         1.0 / 255.0,
                         cv::Size(input_w_, input_h_),
                         cv::Scalar(0, 0, 0),
                         false,
                         false);
  return blob;
}

std::vector<Detection> CudaPipeline::infer_yolo(const cv::Mat & blob,
                                                const cv::Mat & original_frame)
{
  net_.setInput(blob);

  std::vector<cv::Mat> outs;
  try {
    net_.forward(outs, get_output_layer_names());
  } catch (const cv::Exception & e) {
    throw std::runtime_error(
            "YOLO CUDA inference failed. Ensure OpenCV is built with DNN CUDA "
            "support and a compatible CUDA runtime is installed: "
            + std::string(e.what()));
  }

  const int frame_w = original_frame.cols;
  const int frame_h = original_frame.rows;

  std::vector<int> class_ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;

  for (const auto & out : outs) {
    const auto * data = reinterpret_cast<const float *>(out.data);
    for (int i = 0; i < out.rows; ++i, data += out.cols) {
      const cv::Mat scores(1, out.cols - 5, CV_32F,
                           const_cast<float *>(data + 5));
      cv::Point class_id_pt;
      double max_val{0.0};
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

  std::vector<int> nms_indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, nms_indices);

  std::vector<Detection> detections;
  detections.reserve(nms_indices.size());

  for (int idx : nms_indices) {
    Detection det;
    det.class_id   = class_ids[idx];
    det.confidence = confidences[idx];
    det.bbox       = boxes[idx] & cv::Rect(0, 0, frame_w, frame_h);
    det.label      = (det.class_id < static_cast<int>(class_names_.size()))
                     ? class_names_[det.class_id]
                     : ("class_" + std::to_string(det.class_id));
    detections.push_back(det);
  }
  return detections;
}

std::vector<Detection> CudaPipeline::infer_canny(const cv::Mat & frame)
{
  cv::Mat gray, blurred, edges;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.4);
  cv::Canny(blurred, edges, 50, 150);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  std::vector<Detection> detections;
  for (const auto & contour : contours) {
    if (cv::contourArea(contour) < 500.0) { continue; }

    Detection det;
    det.class_id   = 0;
    det.label      = "edge_region";
    det.confidence = 1.0f;
    det.bbox       = cv::boundingRect(contour);
    detections.push_back(det);
  }
  return detections;
}

void CudaPipeline::postprocess(cv::Mat & frame,
                               const std::vector<Detection> & dets,
                               double total_ms)
{
  static const std::vector<cv::Scalar> palette = {
    {  0, 255,   0}, {255,   0,   0}, {  0,   0, 255},
    {255, 255,   0}, {  0, 255, 255}, {255,   0, 255},
    {128, 255,   0}, {  0, 128, 255}, {255, 128,   0},
    {128,   0, 255}
  };

  for (const auto & det : dets) {
    const cv::Scalar & colour = palette[det.class_id % palette.size()];
    cv::rectangle(frame, det.bbox, colour, 2);

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

  const double fps = (total_ms > 0.0) ? (1000.0 / total_ms) : 0.0;
  const std::string hud =
    cv::format("Pipeline: %s | Latency: %.1f ms | FPS: %.1f",
               use_yolo_ ? "YOLO-CUDA" : "Canny-CUDA",
               total_ms, fps);

  cv::rectangle(frame,
                cv::Point(0, 0),
                cv::Point(frame.cols, 22),
                cv::Scalar(0, 0, 0), cv::FILLED);
  cv::putText(frame, hud,
              cv::Point(5, 16),
              cv::FONT_HERSHEY_SIMPLEX, 0.5,
              cv::Scalar(0, 255, 80), 1, cv::LINE_AA);

  cv::putText(frame,
              "Detections: " + std::to_string(dets.size()),
              cv::Point(5, frame.rows - 8),
              cv::FONT_HERSHEY_SIMPLEX, 0.45,
              cv::Scalar(0, 200, 255), 1, cv::LINE_AA);
}

std::vector<std::string> CudaPipeline::get_output_layer_names()
{
  static std::vector<std::string> names;
  if (names.empty()) {
    const auto & layer_names = net_.getLayerNames();
    const auto unconnected = net_.getUnconnectedOutLayers();
    names.resize(unconnected.size());
    for (std::size_t i = 0; i < unconnected.size(); ++i) {
      names[i] = layer_names[unconnected[i] - 1];
    }
  }
  return names;
}

}  // namespace gpu_od
