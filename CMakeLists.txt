cmake_minimum_required(VERSION 2.8.3)
project(shape-detection)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

find_package(Caffe REQUIRED)
find_package(TinyXML REQUIRED)
find_package(CUDA 8.0 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)
add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY
message("Caffe_INCLUDE_DIRS is ${Caffe_INCLUDE_DIRS}")	
option(-std "std" c++11)


message(${CMAKE_CURRENT_LIST_DIR})
message(${Caffe_INCLUDE_DIRS})
include_directories(include ${Caffe_INCLUDE_DIRS})

# CLASSIFIER LIBS

set(GLOG_LIB glog)
add_library(cylinder_classifier src/cylinder_classifier.cpp)
target_link_libraries(cylinder_classifier ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})

