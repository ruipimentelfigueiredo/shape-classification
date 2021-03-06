'''
Title           :create_lmdb.py
Description     :This script divides the training images into 2 sets and stores them in lmdb databases for training and validation.
Author          :Adil Moujahid
Date Created    :20160619
Date Modified   :20160625
version         :0.2
usage           :python create_lmdb.py
python_version  :2.7.11
'''

import os
import glob
import random
import numpy as np

import cv2

from caffe.proto import caffe_pb2
import lmdb
import inspect
import argparse

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

file_path = inspect.stack()[0][1]
repository_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
dataset_path = os.path.join(repository_path, 'dataset')
parser = argparse.ArgumentParser()
parser.add_argument('-d', 
                    '--data-path', 
                    default=dataset_path, 
                    type=str, 
                    help="Path to dataset")
parser.add_argument('-l', 
                    '--lmdb-path', 
                    default=None, 
                    type=str, 
                    help="LMDB save location")
args = parser.parse_args()
base_dataset=args.data_path
base=args.lmdb_path

if base is None:
  base = os.path.join(dataset_path, 'lmdb')
if not os.path.exists(base):
  os.makedirs(base)

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


def make_datum(img, label):
    #image is numpy.ndarray format. BGR instead of RGB
    return caffe_pb2.Datum(
        channels=3,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
        label=label,
        data=np.rollaxis(img, 2).tostring())



#train_lmdb = base + '/train_lmdb'
#validation_lmdb = base + '/validation_lmdb'
train_lmdb = os.path.join(base,'train_lmdb')
validation_lmdb = os.path.join(base, 'validation_lmdb')


if not os.path.exists(train_lmdb):
  os.makedirs(train_lmdb)
if not os.path.exists(validation_lmdb):
  os.makedirs(validation_lmdb)

#os.system('rm -rf  ' + train_lmdb)
#os.system('rm -rf  ' + validation_lmdb)

print base_dataset

#train_data = [img for img in glob.glob("../../dataset_v3/train/*jpg")]
train_data = [img for img in glob.glob(base_dataset + '/train/*jpg')]

#test_data = [img for img in glob.glob("../input/test1/*jpg")]

#Shuffle train_data
random.shuffle(train_data)

print 'Creating train_lmdb'

in_db = lmdb.open(train_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx %  10 == 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'cylinder.' in img_path:
            label = 0
        elif 'sphere.' in img_path:
            label = 1
        else:
            label = 2


        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()
print 'Done'

print '\nCreating validation_lmdb'

in_db = lmdb.open(validation_lmdb, map_size=int(1e12))
with in_db.begin(write=True) as in_txn:
    for in_idx, img_path in enumerate(train_data):
        if in_idx % 10 != 0:
            continue
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
        if 'cylinder.' in img_path:
            label = 0
        elif 'sphere.' in img_path:
            label = 1
        else:
            label = 2

        datum = make_datum(img, label)
        in_txn.put('{:0>5d}'.format(in_idx), datum.SerializeToString())
        print '{:0>5d}'.format(in_idx) + ':' + img_path
in_db.close()

print '\nFinished processing all images'
