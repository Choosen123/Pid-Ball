cmake_minimum_required(VERSION 3.5)
project(pid_ball LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
set(ONNX_RUNTIME_PATH "/opt/onnxruntime")

include_directories(${CMAKE_SOURCE_DIR}/serialport)

add_executable(pid_ball main.cpp serialport/serialport.cpp)

target_include_directories(pid_ball PUBLIC
    "${OpenCV_INCLUDE_DIRS}"
    "${ONNX_RUNTIME_PATH}/include")
target_link_directories(pid_ball PUBLIC
    "${ONNX_RUNTIME_PATH}/lib")
target_link_libraries(pid_ball PUBLIC
    ${OpenCV_LIBS}
    "onnxruntime")

