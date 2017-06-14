# Caffe package
unset(Caffe_FOUND)

###Set the variable Caffe_DIR as the root of your caffe directory
get_filename_component(Caffe_DIR ${CMAKE_CURRENT_LIST_DIR} PATH)
set(Caffe_DIR ${Caffe_DIR}/lib/caffe/build/install)
#set(Caffe_DIR ${PARENT_DIR}/lib/caffe/build/install)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp caffe/vision_layers.hpp
  HINTS
  ${Caffe_DIR}/include)



find_library(Caffe_LIBRARIES NAMES caffe
  HINTS
  ${Caffe_DIR}/lib)


if(Caffe_LIBRARIES AND Caffe_INCLUDE_DIRS)
    set(Caffe_FOUND 1)
endif()
