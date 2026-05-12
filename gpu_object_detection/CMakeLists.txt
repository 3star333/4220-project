cmake_minimum_required(VERSION 3.16)
project(gpu_object_detection)

# ── Compiler settings ─────────────────────────────────────────────────────────
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# ── Find dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake       REQUIRED)
find_package(rclcpp            REQUIRED)
find_package(sensor_msgs       REQUIRED)
find_package(image_transport   REQUIRED)
find_package(cv_bridge         REQUIRED)
find_package(OpenCV            REQUIRED COMPONENTS core imgproc dnn highgui)
find_package(CUDAToolkit       QUIET)
include(CheckLanguage)

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

# NOTE: use plain (non-keyword) signature so it is compatible with the plain
#       signature that ament_target_dependencies uses internally on the same target.
target_link_libraries(detection_pipeline ${OpenCV_LIBS})

ament_target_dependencies(detection_pipeline
  rclcpp      # Only needed if pipeline later uses rclcpp::Logger
)

set(GPU_OD_BUILD_CUDA_PIPELINE OFF)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

if(CMAKE_CUDA_COMPILER)
  find_library(OPENCV_CUDAIMGPROC_LIB NAMES opencv_cudaimgproc)
  find_library(OPENCV_CUDAWARPING_LIB NAMES opencv_cudawarping)

  if(OPENCV_CUDAIMGPROC_LIB AND OPENCV_CUDAWARPING_LIB)
    add_library(detection_pipeline_cuda STATIC
      src/detection_pipeline_cuda.cu
    )

    set_target_properties(detection_pipeline_cuda PROPERTIES
      CUDA_STANDARD 17
      CUDA_STANDARD_REQUIRED ON
    )

    target_include_directories(detection_pipeline_cuda
      PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )

    target_link_libraries(detection_pipeline_cuda
      ${OpenCV_LIBS}
      ${OPENCV_CUDAIMGPROC_LIB}
      ${OPENCV_CUDAWARPING_LIB}
    )

    if(CUDAToolkit_FOUND)
      target_link_libraries(detection_pipeline_cuda CUDA::cudart)
    endif()

    set(GPU_OD_BUILD_CUDA_PIPELINE ON)
    message(STATUS "Building CUDA pipeline backend")
  else()
    message(STATUS "OpenCV CUDA modules not found (cudaimgproc/cudawarping); CUDA pipeline disabled")
  endif()
else()
  message(STATUS "CUDA compiler not available; CUDA pipeline disabled")
endif()

# ── ROS2 node executable ──────────────────────────────────────────────────────
add_executable(detector_node
  src/main.cpp
  src/detector_node.cpp
)

target_include_directories(detector_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(detector_node detection_pipeline)
if(GPU_OD_BUILD_CUDA_PIPELINE)
  target_link_libraries(detector_node detection_pipeline_cuda)
  target_compile_definitions(detector_node PRIVATE GPU_OD_HAS_CUDA_PIPELINE=1)
endif()

ament_target_dependencies(detector_node
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS
  detection_pipeline
  detector_node
  DESTINATION lib/${PROJECT_NAME}
)

if(GPU_OD_BUILD_CUDA_PIPELINE)
  install(TARGETS detection_pipeline_cuda
    DESTINATION lib/${PROJECT_NAME}
  )
endif()

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
