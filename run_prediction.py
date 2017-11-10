# coding: utf-8

'''
This code performs the prediction using TensorFlow object detection API on the
basis of the code presented in the jupyter notebook at
https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
This code is supposed to run in the folder:
/models/research/object_detection/
'''

# # Imports

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from PIL import Image
import time

# ## Env setup
# Path to object detection library
sys.path.append("..")

# ## Object detection imports
from utils import label_map_util
from utils import visualization_utils as vis_util

# # Model preparation

# ## Variables
# Any model exported using the `export_inference_graph.py` tool can be loaded
#here simply by changing `PATH_TO_CKPT` to point to a new .pb file.

# model
MODEL_NAME = 'tree_detection_graph'
ROOT_PATH = '/home/madi/Projects/models/research/trees_recognition/training/'

# Path to frozen detection graph. This is the actual model that is
#used for the object detection.
PATH_TO_CKPT = ROOT_PATH + MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(ROOT_PATH, 'trees_detection.pbtxt')

NUM_CLASSES = 2

PATH_TO_TEST_IMAGES_DIR = '/home/madi/Projects/models/research/trees_recognition/images/pred/'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'aaa_pt604000-4399000.jpg' )]

# Size, in inches, of the output images
IMAGE_SIZE = (3470, 3470)

# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name = '')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network
#predicts `5`, we know that this corresponds to `airplane`.  Here we use internal
#utility functions, but anything that returns a dictionary mapping integers to
#appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, \
             max_num_classes = NUM_CLASSES, \
             use_display_name = True)
category_index = label_map_util.create_category_index(categories)

# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# # Detection
start_time = time.time()
with detection_graph.as_default():
  with tf.Session(graph = detection_graph) as sess:
    # Definite input and output Tensors for detection_graph
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    for image_path in TEST_IMAGE_PATHS:
      image = Image.open(image_path)
      # the array based representation of the image will be used later in order
      # to prepare the result image with boxes and labels on it.
      image_np = load_image_into_numpy_array(image)
      # Expand dimensions since the model expects images to have shape:
      # [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis = 0)
      # Actual detection.
      (boxes, scores, classes, num) = sess.run(
          [detection_boxes, detection_scores, detection_classes, num_detections],
          feed_dict = {image_tensor: image_np_expanded})

      # write file
      CSVfile = open(image_path.split(".")[0] + ".csv", "w")
      # note that y axis is reversed
      CSVfile.write("ID, ymax, xmin, ymin, xmax, class, score" + "\n")
      for item in range(0, int(num[0])):
          CSVfile.write("%s" % item)
          CSVfile.write(",")
          boxes[0][item].tofile(CSVfile, sep=",", format="%s")
          CSVfile.write(",")
          classes[0][item].tofile(CSVfile, sep=",", format="%s")
          CSVfile.write(",")
          scores[0][item].tofile(CSVfile, sep=",", format="%s")
          CSVfile.write("\n")
      CSVfile.close()

      elapsed_time = time.time() - start_time
      print "Time for detection and CSV writing ", elapsed_time

      # Visualization of the results of a detection.
      import matplotlib.pyplot as plt
      import matplotlib.image as mpimg
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

      imgplot = plt.imshow(image_np)
      plt.show()
