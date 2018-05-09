#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  9 14:49:38 2018

@author: atabak
"""


from os import path as op
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


caffe.set_mode_gpu()

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227


file_path = inspect.stack()[0][1]
repository_path = op.dirname(op.dirname(op.dirname(file_path)))
weights_path = op.join(repository_path, 'train_iter_201.caffemodel')
baseline_dir = op.join(repository_path, 'inference_data', 'baseline')
dataset_path = op.join(repository_path, 'dataset')
deploy_path = op.join(repository_path, 'base_networks', 'squeezenet', 'deploy.prototxt')
mean_path = op.join(repository_path, 'dataset', 'mean.binaryproto')
parser = argparse.ArgumentParser()
parser.add_argument('-w', 
                    '--weights-path', 
                    default=weights_path, 
                    type=str, 
                    help="Path to the trained weights")
parser.add_argument('-b', 
                    '--baseline-dir', 
                    default=baseline_dir, 
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
parser.add_argument('-m', 
                    '--mean-binaryproto', 
                    default=mean_path, 
                    type=str, 
                    help="Path to the mean file")
args = parser.parse_args()
base_dataset=args.data_path

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
with open(args.mean_binaryproto) as f:
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
                  glob.glob(op.join(args.data_path,'test/')+'*jpg')]
    

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
    precision[i], recall[i], _ = precision_recall_curve(y_test[:, i],
                                                        y_score[:, i])
    average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(),
    y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")



second_y_score_cyl = np.genfromtxt(op.join(args.baseline_dir,'fitting_quality_cylinders_1.txt')).reshape(-1,1)
second_y_score_cyl = np.hstack((second_y_score_cyl, 1.0-second_y_score_cyl))
second_y_test_cyl = np.ones_like(second_y_score_cyl)*np.asarray([1,0])

second_y_score_obj = np.genfromtxt(op.join(args.baseline_dir,'fitting_quality_others_1.txt')).reshape(-1,1)
second_y_score_obj = np.hstack((second_y_score_obj, 1.0 - second_y_score_obj))
second_y_test_obj = np.ones_like(second_y_score_obj)*np.asarray([0,1])

second_y_test = np.vstack((second_y_test_cyl, second_y_test_obj))
second_y_score = np.vstack((second_y_score_cyl, second_y_score_obj))
#
#
#
precision_2 = dict()
recall_2 = dict()
average_precision_2 = dict()
for i in range(n_classes):
    precision_2[i], recall_2[i], _ = precision_recall_curve(second_y_test[:, i],
                                                        second_y_score[:, i])
    average_precision_2[i] = average_precision_score(second_y_test[:, i], second_y_score[:, i])

# Compute micro-average ROC curve and ROC area
precision_2["micro"], recall_2["micro"], _ = precision_recall_curve(second_y_test.ravel(),
    second_y_score.ravel())
average_precision["micro"] = average_precision_score(y_test, y_score,
                                                     average="micro")
# Plot Precision-Recall curve
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
lw = 2
plt.clf()
plt.plot(recall[0], precision[0], lw=lw, color='cornflowerblue', 
         label='Classifier AUC={0:0.2f}'.format(average_precision[0]))
plt.plot(recall_2[0], precision_2[0], lw=lw, color='darkorange', 
         label='baseline AUC={0:0.2f}'.format(average_precision_2[0]))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-recall curve of class cylinders')
plt.legend(loc="lower left")
plt.savefig(op.join(op.dirname(args.baseline_dir),'p-r.pdf'), format='pdf')