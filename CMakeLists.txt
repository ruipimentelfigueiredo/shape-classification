cmake_minimum_required(VERSION 2.8.3)
project(shape-classification)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wno-sign-compare")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

find_package(Caffe REQUIRED)
find_package(Glog REQUIRED)

#set(CMAKE_PREFIX_PATH "/usr/local")
#find_package(OpenCV 3.4 REQUIRED)
find_package(OpenCV REQUIRED)
#set(OpenCV_INCLUDE_DIRS "/usr/local/include")

find_package(Boost COMPONENTS system filesystem regex REQUIRED)

MESSAGE(${OpenCV_VERSION})
include_directories(include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS})  

#add_library(object_detector src/cpp/object_detection.cpp)
#target_link_libraries(object_detector ${Boost_LIBRARIES} ${OpenCV_LIBS})
add_library(shape_classifier src/cpp/network.cpp src/cpp/cylinder_classifier.cpp)
target_link_libraries (shape_classifier ${Boost_LIBRARIES} ${Caffe_LIBS} -lglog)



