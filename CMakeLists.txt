cmake_minimum_required(VERSION 2.8.3)
project(shape-detection)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_LIST_DIR}/cmake/")

#find_package(Caffe REQUIRED)
find_package(TinyXML REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS system filesystem regex REQUIRED)



#add_definitions(${Caffe_DEFINITIONS})    # ex. -DCPU_ONLY


#message(${Caffe_INCLUDE_DIRS})
<<<<<<< HEAD
#message("RUUUUUUUUUUUUUUUUUI")
#include_directories(include ${Caffe_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
=======

include_directories(include ${CUDA_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})  #${Caffe_INCLUDE_DIRS}
>>>>>>> temporary-detached


# CLASSIFIER LIBS

#set(GLOG_LIB glog)
#add_library(cylinder_classifier src/cylinder_classifier.cpp)
<<<<<<< HEAD
#target_link_libraries(cylinder_classifier ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS})
=======
#target_link_libraries(cylinder_classifier ${Boost_LIBRARIES} ${GLOG_LIB}  ${OpenCV_LIBS}) #${Caffe_LIBRARIES}
>>>>>>> temporary-detached


