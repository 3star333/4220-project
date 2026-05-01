/**
 * main.cpp
 *
 * Entry point. Spins the DetectorNode as a single-threaded executor.
 * For higher throughput later you can switch to MultiThreadedExecutor and
 * annotate the callback with the REENTRANT mutually exclusive callback group.
 */

#include "gpu_object_detection/detector_node.hpp"
#include <rclcpp/rclcpp.hpp>

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<gpu_od::DetectorNode>();

  // Single-threaded spin – no data races possible here.
  // CUDA extension: swap for rclcpp::executors::MultiThreadedExecutor
  //                 and add a callback group for the image subscriber.
  rclcpp::spin(node);

  rclcpp::shutdown();
  return 0;
}
