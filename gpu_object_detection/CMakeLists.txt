cmake_minimum_required(VERSION 3.16)
project(gpu_object_detection)

# ── Compiler settings ──────────────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Find dependencies ──────────────────────────────────────────────────────────
find_package(ament_cmake       REQUIRED)
find_package(rclcpp            REQUIRED)
find_package(sensor_msgs       REQUIRED)
find_package(image_transport   REQUIRED)
find_package(cv_bridge         REQUIRED)
find_package(OpenCV            REQUIRED COMPONENTS core imgproc dnn highgui)

# ── Pipeline library (decoupled from ROS2) ────────────────────────────────────
# Building as a static lib keeps the CUDA swap-in clean:
#   - Later add a gpu_object_detection_cuda target with the same interface.
add_library(detection_pipeline STATIC
  src/detection_pipeline.cpp
)

target_include_directories(detection_pipeline
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(detection_pipeline
  PUBLIC  ${OpenCV_LIBS}
)

ament_target_dependencies(detection_pipeline
  rclcpp      # Only needed if pipeline later uses rclcpp::Logger
)

# ── ROS2 node executable ───────────────────────────────────────────────────────
add_executable(detector_node
  src/main.cpp
  src/detector_node.cpp
)

target_include_directories(detector_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(detector_node
  detection_pipeline
)

ament_target_dependencies(detector_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
)

# ── Install ────────────────────────────────────────────────────────────────────
install(TARGETS
  detection_pipeline
  detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ──────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
