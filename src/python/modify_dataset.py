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

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

file_path = inspect.stack()[0][1]
repository_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
dataset_path = os.path.join(repository_path, 'dataset')
output_path = os.path.join(repository_path, 'output')

first_type='train'
second_type='validation'
test_prob_sample=0.1
single_obj_class=True
mode='train'
#augment=True
parser = argparse.ArgumentParser()
parser.register('type','bool',str2bool) # add type keyword to registries
parser.add_argument('-d', '--data-path', default=dataset_path, type=str, help="Path to dataset")
parser.add_argument('-o', '--output-path', default=output_path, type=str, help="Path to dataset")

parser.add_argument('-sc', '--single-class-binary', default=single_obj_class, type=str2bool, help="Single object class (e.g. cylinder vs all)")
parser.add_argument('-ft', '--first-dname', default=first_type, type=str, help="first partition dataset name")
parser.add_argument('-st', '--second-dname', default=second_type, type=str, help="first partition dataset name")
parser.add_argument('-sp', '--second-prob', default=test_prob_sample, type=float, help="second partition dataset probability")
parser.add_argument('-aug', '--augment', default=True, type=str2bool, help="augment the dataset")
parser.add_argument('-m', '--mode', default=mode, type=str, help="mode", required=True)

args = parser.parse_args()
base_dataset=args.data_path
base_output=args.output_path
single_obj_class=args.single_class_binary
test_prob_sample=args.second_prob
first_dname=args.first_dname
second_dname=args.second_dname
test_prob_sample=args.second_prob
augment=args.augment
mode=args.mode

objs = os.listdir(os.path.join(base_dataset,''))
imgs = {'cylinder':list(), 'sphere':list(),'box':list()}

print 'creating list of images'
for obj in objs:
  idxs_path = os.path.join(base_dataset, '', obj)
  idxs = os.listdir(idxs_path)
  for idx in idxs:
    try:
      imgs_path = os.path.join(idxs_path, idx)
    except OSError:
      continue
    #listdir=os.listdir(imgs_path)
    for img in os.listdir(imgs_path):
      try:
        imgs[obj].append(cv2.imread(os.path.join(imgs_path, img)))
      except KeyError:
        continue
print 'done'      
     
def write_imgs(label='cylinder',mode='train'):
  if mode=='train':
    if not os.path.exists(os.path.join(base_output, first_dname)):
      os.makedirs(os.path.join(base_output, first_dname))
    if not os.path.exists(os.path.join(base_output, second_dname)):
      os.makedirs(os.path.join(base_output, second_dname))
    counter_test = 0
    counter_train = 0
    print 'creating validation and train data for class labeled ' + label
    def select_fate(img, counter_train, counter_test):
      rnd = np.random.rand()
      if rnd < test_prob_sample:
        cv2.imwrite(os.path.join(base_output, second_dname, label+'.'+str(counter_test)+'.jpg'), img)
        counter_test+=1
      else:
        cv2.imwrite(os.path.join(base_output, first_dname, label+'.'+str(counter_train)+'.jpg'), img)
        counter_train+=1
      return counter_train, counter_test
    imgs_list=imgs[label]
    for img in imgs_list:
      counter_train, counter_test = select_fate(img, counter_train, counter_test)
      if augment:
        img  = cv2.flip(img,1)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
        img = cv2.flip(img,0)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
        img = cv2.flip(img,1)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
    print 'done'
  else:
    if not os.path.exists(os.path.join(base_output, first_dname)):
      os.makedirs(os.path.join(base_output, first_dname))
    counter_test = 0
    print 'creating test data for class labeled ' + label
    def select_fate(img, counter_test):
      cv2.imwrite(os.path.join(base_output, first_dname, label+'.'+str(counter_test)+'.jpg'), img)
      return counter_test
    imgs_list=imgs[label]
    for img in imgs_list:
      counter_test = select_fate(img, counter_test)
    print 'done'    


def write_imgs_single(label='cylinder',mode='train'):
  if mode=='train':
    if not os.path.exists(os.path.join(base_output, first_dname)):
      os.makedirs(os.path.join(base_output, first_dname))
    if not os.path.exists(os.path.join(base_output, second_dname)):
      os.makedirs(os.path.join(base_output, second_dname))
    counter_test = 0
    counter_train = 0
    print 'creating validation and train data for class labeled ' + label
    def select_fate(img, counter_train, counter_test):
      rnd = np.random.rand()
      if rnd < test_prob_sample:
        cv2.imwrite(os.path.join(base_output, second_dname, label+'.'+str(counter_test)+'.jpg'), img)
        counter_test+=1
      else:
        cv2.imwrite(os.path.join(base_output, first_dname, label+'.'+str(counter_train)+'.jpg'), img)
        counter_train+=1
      return counter_train, counter_test
    imgs_list=imgs[label]
    for img in imgs_list:
      counter_train, counter_test = select_fate(img, counter_train, counter_test)
      if augment:
        img  = cv2.flip(img,1)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
        img = cv2.flip(img,0)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
        img = cv2.flip(img,1)
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
    print 'done'
    print 'creating validation and train data for all the other labels '
    for obj in objs:
      if obj=='cylinder':
            continue
      def select_fate(img, counter_train, counter_test):
        rnd = np.random.rand()
        if rnd < test_prob_sample:
          cv2.imwrite(os.path.join(base_output, second_dname, 'other.'+str(counter_test)+'.jpg'), img)
          counter_test+=1
        else:
          cv2.imwrite(os.path.join(base_output, first_dname, 'other.'+str(counter_train)+'.jpg'), img)
          counter_train+=1
        return counter_train, counter_test
      try:
        imgs_list=imgs[obj]
      except KeyError:
        continue
      for img in imgs_list:
        counter_train, counter_test = select_fate(img, counter_train, counter_test)
        if augment:
          img  = cv2.flip(img,1)
          counter_train, counter_test = select_fate(img, counter_train, counter_test)
          img = cv2.flip(img,0)
          counter_train, counter_test = select_fate(img, counter_train, counter_test)
          img = cv2.flip(img,1)
          counter_train, counter_test = select_fate(img, counter_train, counter_test)
    print 'done'
  else:
    if not os.path.exists(os.path.join(base_output, first_dname)):
      os.makedirs(os.path.join(base_output, first_dname))
    counter_test = 0
    print 'creating test data for class labeled ' + label
    def select_fate(img, counter_test):
      cv2.imwrite(os.path.join(base_output, first_dname, label+'.'+str(counter_test)+'.jpg'), img)
      counter_test+=1
      return counter_test
    imgs_list=imgs[label]
    for img in imgs_list:
      counter_test = select_fate(img, counter_test)
    print 'done'
    print 'creating test data for all the other labels '
    for obj in objs:
      if obj=='cylinder':
            continue
      def select_fate(img, counter_test):
        cv2.imwrite(os.path.join(base_output, first_dname, 'other.'+str(counter_test)+'.jpg'), img)
        counter_test+=1
        return counter_test
      try:
        imgs_list=imgs[obj]
      except KeyError:
        continue
      for img in imgs_list:
        counter_test = select_fate(img, counter_test)
    print 'done'    

if single_obj_class:
  write_imgs_single('cylinder',mode)
else:
  write_imgs('cylinder')
  write_imgs('sphere')
  write_imgs('box')
