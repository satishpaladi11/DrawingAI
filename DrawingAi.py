
import os
import tarfile
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2

from skimage.measure import compare_ssim
import warnings
warnings.filterwarnings("ignore")

#%tensorflow_version 1.x
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    self.n_cpus=20

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(config=tf.ConfigProto(device_count={ "CPU": self.n_cpus },inter_op_parallelism_threads=self.n_cpus,intra_op_parallelism_threads=10),graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """

    
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_pascal_label_colormap():
  """Creates a label colormap used in PASCAL VOC segmentation benchmark.

  Returns:
    A Colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=int)
  ind = np.arange(256, dtype=int)

  for shift in reversed(range(8)):
    for channel in range(3):
      colormap[:, channel] |= ((ind >> channel) & 1) << shift
    ind >>= 3

  return colormap


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_pascal_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tv'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)


MODEL = DeepLabModel("/home/sathish/PycharmProjects/Yogapose/deeplab_model.tar.gz")
print('model loaded successfully!')

def mask_model(frame):
    # Changing opencv BGR format to pillow supported RGB format
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #converting image array to pillow image object
    image = Image.fromarray(img)
    # giving pillow image to model
    resized_im, seg_map = MODEL.run(image)
    #converting "resized_im" pillow object to numpy array
    resized_im=np.array(resized_im)
    #Changing pillow supported RGB format to opencv BGR format
    resized_im=cv2.cvtColor(resized_im,cv2.COLOR_RGB2BGR)
    #converting model generated segmentation map into a color_map
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    #resizing the image output and segmentation output in to same size
    seg_image = cv2.resize(seg_image, resized_im.shape[1::-1])
    #Detecting the persons color and removing remaining colors in segmentation
    lower = np.array([192,128,128], dtype = "uint8")
    upper = np.array([192,128,128], dtype = "uint8")
    mask = cv2.inRange(seg_image, lower, upper)
    seg_output = cv2.bitwise_and(resized_im,resized_im, mask=mask)
    return seg_output

def draw_effect(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_invert = cv2.bitwise_not(img_gray)
    img_smoothing = cv2.GaussianBlur(img_invert, (21, 21), sigmaX=100, sigmaY=100)
    draw = cv2.divide(img_gray, 255 - img_smoothing, scale=256)
    return draw


def maskdraw(frame):
    out = mask_model(frame)
    blank_image = np.zeros((out.shape[0],out.shape[1]), np.uint8)
    blank_image.fill(255)
    return draw_effect(out)+blank_image

iu=cv2.imread('sam.jpg')
mo=maskdraw(iu)
ms=np.hstack([mo])

blvimage = np.zeros((ms.shape[0], ms.shape[1]), np.uint8)
blvimage.fill(255)
blhimage = np.zeros((ms.shape[0], ms.shape[1]), np.uint8)
blhimage.fill(255)

pensize=10# size from 1 to 100
speed=100#Speed 1 to 100
for i in range(1,ms.shape[0]//pensize+1):
    vs = np.vstack([ms[:i*pensize,:],blvimage[:ms.shape[0]-i*pensize,:]])
    for j in range(1, ms.shape[1] // pensize+1):
        re = np.hstack([vs[:, :j*pensize], blhimage[:, j*pensize:]])
        cv2.imshow('Drawing AI', re)
        if(cv2.waitKey(101-speed)==ord('q')):
            break
    blhimage=vs

cv2.waitKey(10000)
