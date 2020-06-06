import numpy as np
import cv2
import skimage
import os
import matplotlib
import matplotlib.pyplot as plt
from preprocessing import *

# dataDir = "./chest-xray-pneumonia/chest_xray/"
# trainDir = dataDir + "train/"
# testDir = dataDir + "test/"
# valDir = dataDir + "val/"

# sampleNormalImageName = "NORMAL/IM-0115-0001.jpeg"
imagesDir = "./images/"
sampleNormalImageName = "20160928-140314-0.jpg"
sampleImage = cv2.imread(imagesDir+sampleNormalImageName)
# print(sampleImage.shape)
# plt.imshow(sampleImage, cmap='gray')
# plt.show()

grayImage = sampleImage
# grayImage = cv2.cvtColor(sampleImage, cv2.COLOR_BGR2GRAY)
plt.subplot(121)

plt.imshow(grayImage, cmap='gray')
plt.subplot(122)

# dst = clahe(grayImage)
# dst = contraharmonic_mean(grayImage, (3, 3), 0.5)

# dst = erosion(grayImage, cv2.MORPH_ELLIPSE, 1)
# dst = dilate(grayImage, 1, cv2.MORPH_ELLIPSE)
# dst = dilation(grayImage)


# o = openImage(grayImage, 1)
# c = closeImage(grayImage, 1)
# dst = c - o

# dst = gradientImage(, 3)

# dst = equalizeHist(grayImage, globalHist=False)
# dst = equalizeHist(grayImage, globalHist=True)

dst = adaptiveEq(grayImage, 0.03)

# dst = contrastStretching(grayImage)

# dst = canny(grayImage, 0.4)

# dst = bitplane(grayImage, 0) + bitplane(grayImage, 1) #+ bitplane(grayImage, 2) + bitplane(grayImage, 3) + bitplane(grayImage, 4)

plt.imshow(dst, cmap='gray')
plt.show()