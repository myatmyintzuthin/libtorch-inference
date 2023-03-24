cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(libtorch)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

# Argparser
include(FetchContent)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(libtorch inference.cpp)

target_link_libraries(libtorch "${OpenCV_LIBS}" "${TORCH_LIBRARIES}" argparse)

set_property(TARGET libtorch PROPERTY CXX_STANDARD 14)