from __future__ import print_function, absolute_import, division

import numpy as np
import os
import tensorflow as tf
#from six.moves import urllib
from six.moves.urllib.request import urlretrieve
import sys
import inspect


class Detector(object):
  def __init__(self, NUM_CLASSES=3):
    self.dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    self.path = os.path.join(self.dirname,'detector.py')
    root = os.path.dirname(self.dirname)
    self.infer_path = os.path.join(root, 'inference_data')
    sys.path.append(os.path.join(root, 'lib', 'model'))
    
    from utils import label_map_util
    from utils import visualization_utils# as vis_util
    self.vis_util = visualization_utils
    
    self.graph_path = self.maybe_download()
    self.label_path = os.path.join(self.infer_path,'label_map.pbtxt')
    
    label_map = label_map_util.load_labelmap(self.label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    self.category_index = label_map_util.create_category_index(categories)
    
    detection_graph = tf.Graph()
    
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(self.graph_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    
    self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    self.sess = tf.Session(graph=detection_graph)
    
    
  def maybe_download(self, force=False):
    
    def download_progress_hook(count, blockSize, totalSize):
      """A hook to report the progress of a download. This is mostly intended for users with
      slow internet connections. Reports every 5% change in download progress.
      """
      global last_percent_reported
      percent = int(count * blockSize * 100 / totalSize)
    
      if last_percent_reported != percent:
        if percent % 5 == 0:
          sys.stdout.write("%s%%" % percent)
          sys.stdout.flush()
        else:
          sys.stdout.write(".")
          sys.stdout.flush()
          
        last_percent_reported = percent
    
    dest_filename = os.path.join(self.infer_path, 'frozen_inference_graph.pb')
    if force or not os.path.exists(dest_filename):
      print('Attempting to download: frozen_inference_graph.pb')
      link = 'https://mega.nz/#!GMBx3SID!N93BRx0Jyfy0exLd8S3lD4nW7x4VV6LlTazXPXW7gE4'
      urlretrieve(link, dest_filename, reporthook=download_progress_hook)
    return dest_filename
  
  def overlay_shapes(self, rgb_uint8_img):
    image_np_expanded = np.expand_dims(rgb_uint8_img, axis=0)
    (boxes, scores, classes, num) = self.sess.run(
                                                  [self.detection_boxes, 
                                                   self.detection_scores, 
                                                   self.detection_classes, 
                                                   self.num_detections],
                                                   feed_dict={self.image_tensor: image_np_expanded})
    self.vis_util.visualize_boxes_and_labels_on_image_array(
          rgb_uint8_img,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          self.category_index,
          use_normalized_coordinates=True,
          line_thickness=2)
    return rgb_uint8_img
  
  def close_sess(self):
    self.sess.close()
    
if __name__ == "__main__":
  import skimage
  from PIL import Image
  from matplotlib import pyplot as plt
  
  def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
  
  my_detector = Detector()
  img_path = "/home/atabak/Desktop/box.png"
  image = skimage.io.imread(img_path)
  image = skimage.transform.resize(image, [300,300], mode='reflect')
  skimage.io.imsave(img_path, image)
  image = Image.open(img_path)
  image_np = load_image_into_numpy_array(image)
  my_detector.overlay_shapes(image_np)
  plt.imshow(image_np)