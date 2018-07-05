#!/bin/bash

##### Functions

function get_networks
{
	mkdir base_networks
        cd base_networks/
	# SqueezeNet
	mkdir squeezenet
	wget -P ./squeezenet https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
	#wget -P ./squeezenet https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/solver.prototxt
	#wget -P ./squeezenet https://raw.githubusercontent.com/DeepScale/SqueezeNet/master/SqueezeNet_v1.1/train_val.prototxt
	# AlexNet
	#mkdir alexnet
	#wget -P ./alexnet http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel
	#wget -P ./alexnet https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/solver.prototxt
	#wget -P ./alexnet https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_alexnet/train_val.prototxt
}



# Get network models
get_networks

