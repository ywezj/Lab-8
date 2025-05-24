import cv2
import numpy as np

#matplotlib inline
from matplotlib import pyplot as plt
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

input_image = cv2.imread('variant-7.jpg')
norm = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

flipped_code_0 = cv2.flip(norm, 1)
flipped_code_1 = cv2.flip(flipped_code_0, 0) # horizontal flip
plt.imshow(flipped_code_1)