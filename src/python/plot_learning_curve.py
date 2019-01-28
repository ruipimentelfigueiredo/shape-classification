#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:41:52 2018

@author: atabak

Modified from:
  https://github.com/adilmoujahid/deeplearning-cats-dogs-tutorial/blob/master/code/plot_learning_curve.py
"""

import os
import subprocess
import pandas as pd

from matplotlib import pyplot as plt
import inspect
import argparse

#plt.style.use('ggplot')
fontsize=20
file_path = inspect.stack()[0][1]
repository_path = os.path.dirname(os.path.dirname(os.path.dirname(file_path)))
model_log_path = os.path.join(repository_path, '')
parser = argparse.ArgumentParser()
parser.add_argument('-c', 
                    '--caffe-path', 
                    required=True, 
                    type=str, 
                    help="Path to dataset")
parser.add_argument('-l', 
                    '--log-path', 
                    default=None, 
                    type=str, 
                    help="Path to model.log")

args = parser.parse_args()
caffe_path = args.caffe_path
log_path=args.log_path
if log_path is None:
  log_path = model_log_path

model_log_dir_path = os.path.dirname(log_path)
#os.chdir(model_log_dir_path)

model_path = log_path + 'model.log'
train_log_path = log_path + '.train'
test_log_path = log_path + '.test'

'''
Generating training and test logs
'''

train_log_path = model_path + '.train'
test_log_path = model_path + '.test'

#Parsing training/validation logs

command1 = os.path.join(caffe_path, 
                       'tools/', 
                       'extra/', 
                       'parse_log.sh') + ' '+ log_path+'model.log'
command2 = 'mv model.log.test '+log_path
command3 = 'mv model.log.train '+log_path

process = subprocess.Popen(command1, shell=True, stdout=subprocess.PIPE)
process.wait()
process = subprocess.Popen(command2, shell=True, stdout=subprocess.PIPE)
process.wait()
process = subprocess.Popen(command3, shell=True, stdout=subprocess.PIPE)
process.wait()

#Read training and test logs

train_log = pd.read_csv(train_log_path, delim_whitespace=True)
test_log = pd.read_csv(test_log_path, delim_whitespace=True)
test2_log = pd.read_csv(test_log_path, delim_whitespace=True)

'''
Making learning curve
'''
plt.clf()
fig, ax1 = plt.subplots()
#Plotting training and test losses
train_loss, = ax1.plot(train_log['#Iters'], train_log['TrainingLoss'], color='red',  alpha=.5)
test_loss, = ax1.plot(test_log['#Iters'], test_log['TestLoss'], color='green', linewidth=2)

ax1.set_ylim(ymin=0, ymax=1)
ax1.set_xlabel('Iterations', fontsize=fontsize)
ax1.set_ylabel('Loss', fontsize=fontsize)
ax1.tick_params(labelsize=fontsize)
ax1.set_axis_bgcolor('white')

#Plotting test accuracy
ax2 = ax1.twinx()
test_accuracy, = ax2.plot(test_log['#Iters'], test_log['TestAccuracy'], color='blue', linewidth=2)
ax2.set_ylim(ymin=0, ymax=1)
ax2.set_ylabel('Accuracy', fontsize=fontsize)
ax2.tick_params(labelsize=fontsize)

#Adding legend
plt.legend([train_loss, test_loss, test_accuracy], ['Training Loss', 'Validation Loss', 'Validation Accuracy'],  bbox_to_anchor=(1, 0.95))

#plt.title('Training Curve', fontsize=18)

#Saving learning curve
plt.tight_layout()
plt.savefig(log_path+'/model_learning_curve.pdf', format='pdf')

'''
Deleting training and test logs
'''

command = 'rm ' + train_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()

command = command = 'rm ' + test_log_path
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
process.wait()
