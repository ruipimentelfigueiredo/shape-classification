from __future__ import print_function, absolute_import, division

import numpy as np
import os
import tensorflow as tf
import sys
import inspect
#from subprocess import call

class Detector(object):
  def __init__(self, NUM_CLASSES=3):
    self.dirname = os.path.dirname(os.path.abspath(inspect.stack()[0][1]))
    self.path = os.path.join(self.dirname,'detector.py')
    root = os.path.dirname(os.path.dirname(self.dirname))
    infer_path = os.path.join(root, 'inference_data')
    obj_detect_path = os.path.join(root, 'lib', 'models', 'research', 'object_detection')
    sys.path.append(obj_detect_path)
    sys.path.append(os.path.join(root, 'lib', 'models', 'research'))
    sys.path.append(os.path.join(root, 'lib', 'models', 'research', 'slim'))
    
   # os.chdir(os.path.join(root, 'lib', 'models', 'research/'))    
   # print(os.getcwd())
   # call(["protoc", "object_detection/protos/*.proto", "--python_out=."])
    
    from utils import label_map_util
    from utils import visualization_utils
    self.vis_util = visualization_utils
    
    self.graph_path = os.path.join(infer_path, 'frozen_inference_graph.pb')
    self.label_path = os.path.join(infer_path,'label_map.pbtxt')
    
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
  import skimage.io
  import skimage.transform
  from PIL import Image
  from matplotlib import pyplot as plt
  
  def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)
  
  
  my_detector = Detector()
  img_path = os.path.expanduser('~') + "/Desktop/box.png"
  image = skimage.io.imread(img_path)
  image = skimage.transform.resize(image, [300,300], mode='reflect')
  skimage.io.imsave(img_path, image)
  image = Image.open(img_path)
  image_np = load_image_into_numpy_array(image)
  my_detector.overlay_shapes(image_np)
  plt.imshow(image_np)
  plt.savefig('test2.png', format='png', bbox_inches='tight')
