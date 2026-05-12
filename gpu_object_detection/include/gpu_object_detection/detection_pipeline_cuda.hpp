#pragma once

#include "gpu_object_detection/detection_pipeline.hpp"

namespace gpu_od
{

class CudaPipeline : public IPipeline
{
public:
  CudaPipeline(const std::string & model_cfg      = "",
               const std::string & model_weights   = "",
               const std::string & class_names     = "",
               int   input_width                   = 416,
               int   input_height                  = 416,
               float conf_thresh                   = 0.5f,
               float nms_thresh                    = 0.4f);

  std::vector<Detection> run(cv::Mat & frame,
                             PipelineTiming & timing) override;

  std::string name() const override { return "CudaPipeline"; }

private:
  cv::Mat                preprocess(const cv::Mat & frame);
  std::vector<Detection> infer_yolo(const cv::Mat & blob,
                                    const cv::Mat & original_frame);
  std::vector<Detection> infer_canny(const cv::Mat & frame);
  void                   postprocess(cv::Mat & frame,
                                     const std::vector<Detection> & dets,
                                     double total_ms);
  std::vector<std::string> get_output_layer_names();

  cv::dnn::Net             net_;
  std::vector<std::string> class_names_;
  bool                     use_yolo_{false};

  int   input_w_;
  int   input_h_;
  float conf_thresh_;
  float nms_thresh_;
};

}  // namespace gpu_od
