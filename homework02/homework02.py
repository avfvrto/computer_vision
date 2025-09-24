import cv2
import numpy as np
import os
from pathlib import Path


def detect_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  # DoG-based detector
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors

def match_features(desc1, desc2, ratio=0.75):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good

def compute_homography(kp1, kp2, matches):
    if len(matches) < 4:
        return None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H, mask

def stitch_two(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Transform corners of img2
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H).reshape(-1,2)

    # Combine corners from both images
    all_corners = np.vstack((warped_corners, [[0,0],[0,h1],[w1,h1],[w1,0]]))
    [xmin, ymin] = np.int64(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int64(all_corners.max(axis=0).ravel())

    # Translate so result starts at (0,0)
    translate = [-xmin, -ymin]
    Ht = np.array([[1,0,translate[0]], [0,1,translate[1]], [0,0,1]]) @ H

    result = cv2.warpPerspective(img2, Ht, (xmax-xmin, ymax-ymin))
    result[translate[1]:h1+translate[1], translate[0]:w1+translate[0]] = img1
    return result

def stitch_images(images):
    pano = images[0]
    for i in range(1, len(images)):
        kp1, desc1 = detect_features(pano)
        kp2, desc2 = detect_features(images[i])
        matches = match_features(desc1, desc2)
        H, _ = compute_homography(kp1, kp2, matches)
        if H is None:
            raise RuntimeError(f"Not enough matches between image {i} and {i+1}")
        pano = stitch_two(pano, images[i], H)
    return pano

def load_images_from_folder(folder_path):
    folder = Path(folder_path)
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    images = [cv2.imread(str(f)) for f in files]
    return images, files

if __name__ == "__main__":
    folder = r"/Users/ainee_f/Documents/School - Docs/computer_vision/homework02/files/"  # change to your path
    images, files = load_images_from_folder(folder)
    print("Loaded:", [f.name for f in files])

    panorama = stitch_images(images)
    cv2.imwrite("panorama.jpg", panorama)
    print("Panorama saved as panorama.jpg")