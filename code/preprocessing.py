import cv2
import skimage
import numpy as np
from skimage import img_as_float
from skimage.morphology import reconstruction
from scipy.ndimage import gaussian_filter
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage import feature, filters

def erosion(image, erosionType, kernelSize):
  # if len(image.shape)>2:
  #   raise "Shape mismatch"

  # kernelSize = 3 # should be odd
  # erosionType = cv2.MORPH_RECT
  # erosionType = cv2.MORPH_CROSS
  # erosionType = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(erosionType, (2*kernelSize + 1, 2*kernelSize+1), (kernelSize, kernelSize))
  erodedImage = cv2.erode(image, element)
  return erodedImage

def dilate(image, kernelSize, erosionType):
  # if len(image.shape)>2:
  #   raise "Shape mismatch"

  # kernelSize = 3 # should be odd
  # erosionType = cv2.MORPH_RECT
  # erosionType = cv2.MORPH_CROSS
  # erosionType = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(erosionType, (2*kernelSize + 1, 2*kernelSize+1), (kernelSize, kernelSize))
  dilatedImage = cv2.dilate(image, element)
  return dilatedImage

def openImage(image, kernelSize):
  # if len(image.shape)>2:
  #   raise "Shape mismatch"

  # kernelSize = 3 # should be odd
  # erosionType = cv2.MORPH_RECT
  # erosionType = cv2.MORPH_CROSS
  erosionType = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(erosionType, (2*kernelSize + 1, 2*kernelSize+1), (kernelSize, kernelSize))
  dst = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
  return dst

def closeImage(image, kernelSize):
  # if len(image.shape)>2:
  #   raise "Shape mismatch"

  kernelSize = 3 # should be odd
  # erosionType = cv2.MORPH_RECT
  # erosionType = cv2.MORPH_CROSS
  erosionType = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(erosionType, (2*kernelSize + 1, 2*kernelSize+1), (kernelSize, kernelSize))
  dst = cv2.morphologyEx(image, cv2.MORPH_CLOSE, element)
  return dst

def contraharmonic_mean(img, size, Q):
  num = np.power(img, Q + 1)
  denom = np.power(img, Q)
  kernel = np.full(size, 1.0)
  result = cv2.filter2D(num, -1, kernel) / cv2.filter2D(denom, -1, kernel)
  return result

def gradientImage(image, kernelSize):
  # if len(image.shape)>2:
  #   raise "Shape mismatch"
  # kernelSize = 3 # should be odd
  # erosionType = cv2.MORPH_RECT
  # erosionType = cv2.MORPH_CROSS
  erosionType = cv2.MORPH_ELLIPSE
  element = cv2.getStructuringElement(erosionType, (2*kernelSize + 1, 2*kernelSize+1), (kernelSize, kernelSize))
  dst = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, element)
  return dst

def clahe(gray_image):
  # create a CLAHE object (Arguments are optional).
  clahe = cv2.createCLAHE()
  cl1 = clahe.apply(gray_image)
  return cl1

def dilation(image):
  image = img_as_float(image)
  image = gaussian_filter(image, 3)
  # seed = np.copy(image)
  h = 0.9
  seed = image - h
  # seed[1:-1, 1:-1] = image.min()
  mask = image

  dilated = reconstruction(seed, mask, method='dilation')
  return dilated

def equalizeHist(image, globalHist=False):
  if globalHist:
    img = exposure.equalize_hist(image)
  else:
    # Equalization
    selem = disk(200)
    img = rank.equalize(image, selem=selem)
  return img

def adaptiveEq(image, limit=0.03):
  # Adaptive Equalization
  img_adapteq = exposure.equalize_adapthist(image, clip_limit=limit)
  return img_adapteq

def contrastStretching(image):
  # Contrast stretching
  p2, p98 = np.percentile(image, (25, 90))
  img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
  return img_rescale

def canny(image, sigma=1):
  edges = feature.canny(image, sigma=sigma)
  return edges

def sobel(image):
  edge_sobel = filters.sobel(image)
  return edge_sobel

# Not working
def laplace(image, kernelSize=3):
  h = 100
  mask = image - h
  dst = filters.laplace(image, kernelSize, mask)
  return dst

def sift(image):
  orb = cv2.ORB_create(nfeatures=1500)

  # find the keypoints and descriptors with SIFT
  kp, desc_a = orb.detectAndCompute(image, None)
  
  img=cv2.drawKeypoints(image,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  return img
  # kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # # initialize the bruteforce matcher
  # bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

  # # match.distance is a float between {0:100} - lower means more similar
  # matches = bf.match(desc_a, desc_b)
  # similar_regions = [i for i in matches if i.distance < 70]
  # if len(matches) == 0:
  #   return 0
  # return len(similar_regions) / len(matches)
def bitplane(img, bit):
  lst = []
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      lst.append(np.binary_repr(img[i][j] ,width=8))
  bit_img = (np.array([int(i[bit]) for i in lst],dtype = np.uint8) * 128).reshape(img.shape[0],img.shape[1])
  return bit_img