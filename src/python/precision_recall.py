#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:49:38 2018

@author: atabak
"""

from os import path as op
import os
import inspect
import argparse
import glob
import cv2
import caffe
import numpy as np
from caffe.proto import caffe_pb2

from itertools import cycle

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from matplotlib import pyplot as plt
import sys
caffe.set_mode_gpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

fontsize=20

file_path = os.getcwd()
weights_path = op.join(file_path, 'train_iter_201.caffemodel')
output_file = op.join(file_path, 'p-r.pdf')
dataset_path = op.join(file_path, 'dataset-path')
deploy_path =  op.join(file_path, 'base_networks', 'squeezenet', 'deploy.prototxt')
mean_path =    op.join(file_path, 'mean.binaryproto')

parser = argparse.ArgumentParser()
parser.add_argument('-w', 
                    '--weights-path', 
                    default=weights_path, 
                    type=str, 
                    help="Path to the trained weights")
parser.add_argument('-b', 
                    '--output-file', 
                    default=output_file, 
                    type=str, 
                    help="Directory of baseline classifications")
parser.add_argument('-p', 
                    '--deploy-prototxt', 
                    default=deploy_path, 
                    type=str, 
                    help="Path to deploy prototxt")
parser.add_argument('-d', 
                    '--data-path', 
                    default=dataset_path, 
                    type=str, 
                    help="Path to dataset")
parser.add_argument('-fc', 
                    '--fitting-cylinders-path', 
                    default=None, 
                    type=str, 
                    help="Path to fitting logs folder")
parser.add_argument('-fo', 
                    '--fitting-others-paths', 
                    default=[], 
                    #type=list, 
                    help="Paths to fitting logs folder",
                    nargs='+')
parser.add_argument('-m', 
                    '--mean-binaryproto', 
                    default=mean_path, 
                    type=str, 
                    help="Path to the mean file",
                    required=True)

args=parser.parse_args()
base_dataset=args.data_path
fitting_cylinders_path=args.fitting_cylinders_path
fitting_others_paths=args.fitting_others_paths
mean_path=args.mean_binaryproto
output_file=args.output_file

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open(mean_path) as f:
    mean_blob.ParseFromString(f.read())
    mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))

#Read model architecture and trained model's weights
net = caffe.Net(args.deploy_prototxt, 
                caffe.TEST, 
                weights=args.weights_path)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in \
                  glob.glob(op.join(args.data_path,'')+'*jpg')] 

#Making predictions
y_score = list()
y_test = list()
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)

    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob'][0].squeeze().copy()
    new_prob = np.zeros([1,2])
    new_prob[0,0]=pred_probas[0]
    new_prob[0,1]=pred_probas[1:].sum()
    y_score.append(new_prob)
    if 'cylinder.' in img_path.split('/')[-1][:-4]:
      y_test.append(np.array([1,0]))
    else:
      y_test.append(np.array([0,1]))

y_test = np.asarray(y_test)
y_score = np.asarray(y_score).squeeze()
##
##
n_classes = 1
###
precision = dict()
recall = dict()
average_precision = dict()
for i in range(n_classes):
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")

precision_2 = dict()
recall_2 = dict()
average_precision_2 = dict()

second_y_score_cyl=np.array([]).reshape(-1,2)
second_y_test_cyl=np.array([]).reshape(-1,2)

# fitting parse cylinder scores
filename='cluster_fitting_score.txt'
for base_root, base_dirs, base_files in os.walk(fitting_cylinders_path):
    for directory in base_dirs:
        path_=fitting_cylinders_path+directory+'/results/'

        if directory=='results':
            continue

        # cylinders
        second_y_score_cyl_temp=np.genfromtxt(op.join(path_,filename)).reshape(-1,1)
        second_y_score_cyl_temp=np.hstack((second_y_score_cyl_temp, 1.0-second_y_score_cyl_temp))
        second_y_test_cyl_temp=np.ones_like(second_y_score_cyl_temp)*np.asarray([1,0])

        second_y_score_cyl=np.append(second_y_score_cyl.reshape(-1,2),second_y_score_cyl_temp,axis=0)
        second_y_test_cyl=np.append(second_y_test_cyl.reshape(-1,2),second_y_test_cyl_temp,axis=0)


second_y_score_obj=np.array([]).reshape(-1,2)
second_y_test_obj=np.array([]).reshape(-1,2)

# fitting parse other scores
for fitting_others_path in fitting_others_paths[0].split(","):
    for base_root, base_dirs, base_files in os.walk(fitting_others_path):
        for directory in base_dirs:
            path_=fitting_others_path+directory+'/results/'

            if directory=='results':
                continue

            # others
            second_y_score_obj_temp=np.genfromtxt(op.join(path_,filename)).reshape(-1,1)
            second_y_score_obj_temp=np.hstack((1.0-second_y_score_obj_temp, second_y_score_obj_temp))
            second_y_test_obj_temp=np.ones_like(second_y_score_obj_temp)*np.asarray([0,1])

            second_y_score_obj=np.append(second_y_score_obj.reshape(-1,2),second_y_score_obj_temp,axis=0)
            second_y_test_obj=np.append(second_y_test_obj.reshape(-1,2),second_y_test_obj_temp,axis=0)


second_y_test = np.vstack((second_y_test_cyl, second_y_test_obj))
second_y_score = np.vstack((second_y_score_cyl, second_y_score_obj))


temp_precision_2=dict()
temp_average_precision_2=dict()
temp_recall_2=dict()

for i in range(n_classes):
    temp_precision_2[i], temp_recall_2[i], _ = precision_recall_curve(second_y_test[:, i].astype(int), second_y_score[:, i])
    temp_average_precision_2[i] = average_precision_score(second_y_test[:, i].astype(int), second_y_score[:, i])

    precision_2.update(temp_precision_2)
    recall_2.update(temp_recall_2)
    average_precision_2.update(temp_average_precision_2)

# Compute micro-average ROC curve and ROC area
#precision_2["micro"], recall_2["micro"], _ = precision_recall_curve(second_y_test.ravel(), second_y_score.ravel())
#average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
# Plot Precision-Recall curve
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
plt.clf()
fig, ax1 = plt.subplots()
ax1.plot(recall[0], precision[0], lw=lw, color='blue', label='classifier AUC={0:0.2f}'.format(average_precision[0]))
ax1.plot(recall_2[0], precision_2[0], lw=lw, color='orange', label='baseline AUC={0:0.2f}'.format(average_precision_2[0]))
ax1.set_xlabel('Recall',fontsize=fontsize)
ax1.set_ylabel('Precision',fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)
ax1.set_ylim([0.0, 1.05])
ax1.set_xlim([0.0, 1.0])
#plt.title('Precision-recall curve of class cylinders')
ax1.legend(loc="lower left")
plt.tight_layout()
plt.savefig(op.join(op.dirname(args.output_file),'p-r.pdf'), format='pdf')

