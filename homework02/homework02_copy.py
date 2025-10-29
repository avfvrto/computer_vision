import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def detect_features(img):
    # This function is used to detect the key points in an images
    # this is so that we can properly align the images 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  # DoG-based detector
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_features(desc1, desc2, ratio=0.75):
    # This function uses a KDTreee matcher to 
    # to find which matched features are a good fit for homography
    # estimation
    # FLANN parameters for SIFT (float descriptors)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc1, desc2, k=2)

    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good

def show_feature_matches(img1, kp1, img2, kp2, matches, max_display=50):
    #visualize
    vis = cv2.drawMatches(
        img1, kp1, img2, kp2,
        matches[:max_display], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 7))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Showing {min(max_display, len(matches))} matches")
    plt.axis("off")
    plt.show()

def _dlt_homography(src_pts, dst_pts):
    # this function loops over correspondeces
    # x, y: points in the source image
    # u, v: points in the destination point (tries to match)
    n = src_pts.shape[0]
    A = []
    for i in range(n):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        
        # if a match is found, two rows are appended to matrix A
        # the rows come from re-arranging the equations for a projective transform
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])
        
    # make as array
    A = np.asarray(A)

    # Solve Ah = 0 using SVD
    # Use single value decomposition (SVD) to find the non-trivial solution
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]   # last row of V^T
    H = h.reshape(3, 3) # reshape into H-matrix 3x3

    # Normalize so bottom-right is 1
    if H[2,2] != 0:
        H = H / H[2,2]
    return H
        
def compute_homography(kp1, kp2, matches, ransac_thresh=5.0):
    # import matched coordinates from the match features set
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # In this stage, we use RANSAC to find inliers among the keypoints 
    # extracted from match features. H will be computed manually.
    _, mask = cv2.findHomography(pts2, pts1,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=ransac_thresh)
    if mask is None:
        return None, None

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    if len(inliers1) < 4:
        return None, None

    # manually compute homography based on the function _dlt_homography
    H = _dlt_homography(inliers2, inliers1)

    return H, mask

def compute_pairwise_homographies(images):
    # in this stage we compute pairwise homography
    # comparing a source image and a destination / projection points
    pair_H = []
    for i in range(len(images) - 1):
        kp1, desc1 = detect_features(images[i])
        kp2, desc2 = detect_features(images[i+1])
        matches = match_features(desc1, desc2)
        
        show_feature_matches(images[i], kp1, images[i+1], kp2, matches)
        
        H, _ = compute_homography(kp1, kp2, matches, ransac_thresh=5.0)
        if H is None:
            raise RuntimeError(f"Not enough matches between image {i} and {i+1}")
        pair_H.append(H)
    return pair_H

def accumulate_to_mid(pair_H, mid):
    n = len(pair_H) + 1
    H_to_mid = [None] * n
    H_to_mid[mid] = np.eye(3)

    # accumulate left side (towards 0)
    for i in range(mid-1, -1, -1):
        H_to_mid[i] = H_to_mid[i+1] @ np.linalg.inv(pair_H[i])
    # accumulate right side (towards n-1)
    for i in range(mid, len(pair_H)):
        H_to_mid[i+1] = H_to_mid[i] @ pair_H[i]

    return H_to_mid

def stitch_images(images):
    mid = len(images) // 2
    pair_H = compute_pairwise_homographies(images)
    H_to_mid = accumulate_to_mid(pair_H, mid)

    # find panorama bounds by warping corners of each image
    all_corners = []
    for img, H in zip(images, H_to_mid):
        h, w = img.shape[:2]
        corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
        warped = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped)

    all_corners = np.vstack(all_corners).reshape(-1,2)
    [xmin, ymin] = np.int64(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int64(all_corners.max(axis=0).ravel())

    translate = [-xmin, -ymin]
    T = np.array([[1,0,translate[0]], [0,1,translate[1]], [0,0,1]])

    pano_w, pano_h = xmax - xmin, ymax - ymin
    result = np.zeros((pano_h, pano_w, 4), dtype=np.uint8)

    # --- feather blend instead of hard overwrite ---
    acc = np.zeros((pano_h, pano_w, 3), np.float32)   # to accumulate weighted colors
    weight = np.zeros((pano_h, pano_w), np.float32)   # total weight per pixel

    for img, H in zip(images, H_to_mid):
        Ht = T @ H
        warped = cv2.warpPerspective(img, Ht, (pano_w, pano_h)).astype(np.float32)

        # simple binary mask (1 where pixels exist)
        mask = cv2.warpPerspective(
            np.ones(img.shape[:2], np.float32), Ht, (pano_w, pano_h)
            )        

        acc += warped * mask[:, :, None]  # add weighted colors
        weight += mask                    # add weights

    # avoid divide-by-zero
    result_rgb = (acc / np.maximum(weight, 1e-5)[:, :, None]).astype(np.uint8)

    # if you still want 4 channels with full alpha:
    result = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = (weight > 0).astype(np.uint8) * 255
        
    # --- Optional crop to match the middle image's height ---
    h_mid, w_mid = images[mid].shape[:2]
    pano_h, pano_w = result.shape[:2]

    if pano_h > h_mid:
        extra = pano_h - h_mid
        top = extra // 2           # crop equal from top and bottom
        bottom = top + h_mid
        result = result[top:bottom, :]
    return result


def load_images_from_folder(folder_path):
    folder = Path(folder_path)
    files = sorted([f for f in folder.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    images = [cv2.imread(str(f)) for f in files]
    return images, files

if __name__ == "__main__":
    folder = r"/Users/ainee_f/Documents/School - Docs/computer_vision/homework02/files/new_set"  # change to your path
    images, files = load_images_from_folder(folder)
    print("Loaded:", [f.name for f in files])

    panorama = stitch_images(images)
    cv2.imwrite("panorama.jpg", panorama)
    print("Panorama saved as panorama.jpg")
    
    
    
    
    
    
    
    