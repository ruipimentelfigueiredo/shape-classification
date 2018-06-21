#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 20:19:54 2017

@author: atabak
"""

import os
import cv2
import numpy as np
import inspect
import argparse
file_path = inspect.stack()[0][1]
repository_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
dataset_path = os.path.join(repository_path, 'dataset')
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data-path', default=dataset_path, type=str, help="Path to dataset")
args = parser.parse_args()
base_dataset=args.data_path

objs = os.listdir(os.path.join(base_dataset,''))
imgs = {'cylinder':list(), 'sphere':list(), 'other':list()}
print 'creating list of images'
for obj in objs:
  idxs_path = os.path.join(base_dataset, '', obj)
  idxs = os.listdir(idxs_path)
  for idx in idxs:
    imgs_path = os.path.join(idxs_path, idx)
    for img in os.listdir(imgs_path):
      imgs[obj].append(cv2.imread(os.path.join(imgs_path, img)))
print 'done'      
      
def write_imgs(label='cylinder'):
  if not os.path.exists(os.path.join(base_dataset, 'train')):
    os.makedirs(os.path.join(base_dataset, 'train'))
  if not os.path.exists(os.path.join(base_dataset, 'test')):
    os.makedirs(os.path.join(base_dataset, 'test'))
  counter_test = 0
  counter_train = 0
  print 'creating test and train data for class labeled ' + label
  def select_fate(img, counter_train, counter_test):
    rnd = np.random.rand()
    if rnd < 0.1:
      cv2.imwrite(os.path.join(base_dataset, 'test', label+'.'+str(counter_test)+'.jpg'), img)
      counter_test+=1
    else:
      cv2.imwrite(os.path.join(base_dataset, 'train', label+'.'+str(counter_train)+'.jpg'), img)
      counter_train+=1
    return counter_train, counter_test
  imgs_list=imgs[label]
  for img in imgs_list:
    counter_train, counter_test = select_fate(img, counter_train, counter_test)
    img  = cv2.flip(img,1)
    counter_train, counter_test = select_fate(img, counter_train, counter_test)
    img = cv2.flip(img,0)
    counter_train, counter_test = select_fate(img, counter_train, counter_test)
    img = cv2.flip(img,1)
    counter_train, counter_test = select_fate(img, counter_train, counter_test)
  print 'done'
    
write_imgs('cylinder')
write_imgs('sphere')
write_imgs('other')
      
      
