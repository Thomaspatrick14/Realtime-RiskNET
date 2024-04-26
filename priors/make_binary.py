import numpy as np
import os
import matplotlib.pyplot as plt
import PIL.Image
import cv2
from sys import platform
from pathlib import Path


prior = np.array(PIL.Image.open(Path("gta_prior_car.jpg")))
percentage = 0.5
threshold = int(percentage*255)

prior[prior > threshold] = 255
prior[prior <= threshold] = 0

img = PIL.Image.fromarray(prior)
img.save("gta_prior_car_binary.jpg")

plt.imshow(prior)
plt.show()

