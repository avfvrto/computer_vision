# homework 1 for EECE 7216: Computer vision
# Angelin Favorito

# Arts & special effects using basic mathematical operations

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pathlib import Path


# == Part 2: Subtraction == #

display_plots = True # can turn on/off to show plots

# Load images:
part1_fp = Path(r'homework01/files/partB/raw_images')
image_list = []

# In this step, I load all of the images taken of me walking. This was takine with a Basler camera
# attached to a stationary imaging rig, so no registration or alignment was eneded since all images
# were already aligned on a mount

for file in os.listdir(part1_fp):
    file_fp = part1_fp/Path(file)
    img_array = cv2.imread(file_fp, cv2.IMREAD_UNCHANGED)
    image_list.append(img_array)

diff_images = [] # initialize an array of all the difference images, one after another

# iterate through each image in the image list and turn into grayscale (although original images were already mono)
# here we subtract the current frame from the previous frame and append the resulting difference image
for i in range(1, len(image_list)):
    # Convert to grayscale
    prev_gray = cv2.cvtColor(image_list[i-1], cv2.COLOR_BGR2GRAY) if image_list[i-1].ndim == 3 else image_list[i-1]
    curr_gray = cv2.cvtColor(image_list[i], cv2.COLOR_BGR2GRAY) if image_list[i].ndim == 3 else image_list[i]
    
    # Compute absolute difference using cv2. This takes a subtraction of two consecutive frames to see what's
    # different bewteen them
    diff = cv2.absdiff(curr_gray, prev_gray)
    diff_images.append(diff)

# plot each image in the diff_images array to see the succession of differences
for i, diff in enumerate(diff_images, start=1):
    plt.figure()
    plt.imshow(diff, cmap='gray')
    plt.title(f"Difference Frame {i} → {i+1}")
    plt.axis('off')
    plt.show()

# The results shown in the image file outputs (Figures 1-4 PNGs) show images in black and white, where the
# lighter parts explain areas of the image with a larger difference from the previous image. In the image set
# I used, it can be seen that as i walk through the frame, the data is able to highlight me as I am the only
# moving feature in the frame and everything else is stationary and unmoving