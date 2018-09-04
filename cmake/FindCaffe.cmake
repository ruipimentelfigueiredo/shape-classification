
# Caffe package for CNN Triplet training
unset(Caffe_FOUND)

find_path(Caffe_INCLUDE_DIRS NAMES caffe/caffe.hpp caffe/common.hpp caffe/net.hpp caffe/proto/caffe.pb.h caffe/util/io.hpp
  HINTS
  /home/vislab/caffe/build/install/include/) #example: /home/jonhdoe/caffe

find_library(Caffe_LIBS NAMES caffe
  HINTS
  /home/vislab/caffe/build/lib)

if(Caffe_LIBS AND Caffe_INCLUDE_DIR)
    set(Caffe_FOUND 1)
endif()
