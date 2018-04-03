cmake_minimum_required(VERSION 2.8.3)
project(shape-detection)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

#set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

#find_package(Caffe REQUIRED)
#find_package(TinyXML REQUIRED)
#find_package(CUDA REQUIRED)

#set(CMAKE_PREFIX_PATH "/usr/local")
find_package(OpenCV 3.4 REQUIRED)
#set(OpenCV_INCLUDE_DIRS "/usr/local/include")

find_package(Boost COMPONENTS system filesystem regex REQUIRED)

MESSAGE(${OpenCV_INCLUDE_DIRS})
include_directories(include ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})  #${Caffe_INCLUDE_DIRS}
link_directories(${OpenCV_LIB_DIR})

add_library(object_detector src/cpp/object_detection.cpp)
target_link_libraries(object_detector ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})

# CLASSIFIER LIBS

#set(GLOG_LIB glog)
#add_library(cylinder_classifier src/cylinder_classifier.cpp)

#target_link_libraries(cylinder_classifier ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})



