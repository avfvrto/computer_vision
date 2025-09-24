# homework 1 for EECE 7216: Computer vision
# Angelin Favorito

# Arts & special effects using basic mathematical operations

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from pathlib import Path


# == Part 1: Double Exposure == #

display_plots = True # can turn on/off to show plots

# Load images:
part1_fp = Path(r'homework01/files/partA/raw_images')
image_list = []

# In this step, I load all of the images taken in different exposure times and
# lighting conditions into a list of image arrays
for file in os.listdir(part1_fp):
    file_fp = part1_fp/Path(file)
    img_array = cv2.imread(file_fp, cv2.IMREAD_UNCHANGED)
    image_list.append(img_array)

# We select a base image to align all succeeding images to. The first one in the list was 
# selected.
base_image = image_list[0]
h, w, b = base_image.shape # height, width, bands
aligned_images = [base_image] # here we initialize the list of aligned images, starting with the base


# There is a lot of misalignment in the images due to wobby camera movements.

# To align all the images, we use athe ORB functionality of cv2 to detect features. These features
# will be used to align the stack of images with one another. It will be used with a BF (brute force)
# matcher that will calculate how far the ORB's detected points are from each other

# TRY: SIFT, RANSAC see: history of gaussian
orb = cv2.ORB_create(500)  # this determines number of keypoints or features (Higher better but slower)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) # use a brute force matcher to match features made by the ORB

# Compute base keypoints/descriptors once
base_image_gray = cv2.cvtColor(base_image, cv2.COLOR_BGR2GRAY) if base_image.ndim == 3 else base_image
kp1, des1 = orb.detectAndCompute(base_image_gray, None) # detect keypoints

# here we use a for-loop and iteratre through the image_list to match the features
# of each image to the base and calculate a transformation matrix to align them

# first image as first 
for i, img in enumerate(image_list[1:], start=2):
    img_resized = cv2.resize(img, (w,h))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY) if img_resized.ndim == 3 else img_resized
    
    # Compute ORB features
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    
    # Match features
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched points
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    # Compute homography
    H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC)
    
    # NOTE: First filter out junk, find histogram of gaussian, then use matching
    
    # once the homography is computed, we use the H matrix to do an affine transform and 
    # align the current image to the base and append it to the aligned list
    aligned = cv2.warpPerspective(img_resized, H, (w,h)) 
    
    aligned_images.append(aligned)

# Next, we have to add all the images and then normalize them for displaying.
# this needs to be done because the iamges may not follow standard 0-255 values and be difficult
# to display or have irregularities

stack = np.stack(aligned_images, axis=0)  
sum_image = np.sum(stack, axis=0)      
sum_image_norm = cv2.normalize(sum_image, None, 0, 255, cv2.NORM_MINMAX)
sum_image_uint8 = sum_image_norm.astype(np.uint8)

# Now convert BGR → RGB because of openCV's image loading order
img_rgb = cv2.cvtColor(sum_image_uint8, cv2.COLOR_BGR2RGB)

# Next, display the image for viewing
if display_plots:

    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()


