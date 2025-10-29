import cv2
import numpy as np
import os
from pathlib import Path


def detect_features(img):
    # This function is used to detect the key points in an images
    # this is so that we can properly align the images 
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()  # detects points like corners, blobs, edges, and creates descriptors
    keypoints, descriptors = sift.detectAndCompute(gray, None) # locations of those points
    return keypoints, descriptors

def match_features(desc1, desc2, ratio=0.75):
    # This function uses a basic brute force matcher to match each descriptor in base image
    # to other images. Uses Lowe's ratio test (see if point is significantly better than second best)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good

#def compute_homography(kp1, kp2, matches, ransac_thresh=5.0):
#    # this function creates the homography matrix between two images
#    # based on keypoints and the matches found in the match_features stage
#    if len(matches) < 4:
#        return None, None
#    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
#    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
#    # Using RANSAC with reprojection threshold
#    H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
#    return H, mask

def _dlt_homography(src_pts, dst_pts):
    """
    Compute homography H (3x3) that maps src_pts -> dst_pts using DLT.
    src_pts, dst_pts: Nx2 arrays of matching points.
    """
    n = src_pts.shape[0]
    A = []
    for i in range(n):
        x, y = src_pts[i][0], src_pts[i][1]
        u, v = dst_pts[i][0], dst_pts[i][1]
        A.append([-x, -y, -1,  0,  0,  0, u*x, u*y, u])
        A.append([ 0,  0,  0, -x, -y, -1, v*x, v*y, v])
    A = np.asarray(A)

    # Solve Ah = 0 using SVD
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]   # last row of V^T
    H = h.reshape(3, 3)

    # Normalize so bottom-right is 1
    if H[2,2] != 0:
        H = H / H[2,2]
    return H



def compute_homography(kp1, kp2, matches, ransac_thresh=5.0):
    # --- Step 1. Extract matched point coordinates ---
    if len(matches) < 4:
        return None, None
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # --- Step 2. Run RANSAC only to get inlier mask ---
    # This just finds which points are inliers; we will solve H ourselves.
    _, mask = cv2.findHomography(pts2, pts1,
                                 method=cv2.RANSAC,
                                 ransacReprojThreshold=ransac_thresh)
    if mask is None:
        return None, None

    inliers1 = pts1[mask.ravel() == 1]
    inliers2 = pts2[mask.ravel() == 1]

    if len(inliers1) < 4:
        return None, None

    # --- Step 3. Manually compute H from inliers using DLT ---
    H = _dlt_homography(inliers2, inliers1)

    return H, mask
    
    
    

def stitch_two(img1, img2, H):
    # this stage creates a stitch fo two matrices (two images in order)
    # using the homography matrix generated in the previous step
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Transform corners of img2
    corners_img2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners_img2, H).reshape(-1,2)

    # this stage determines the size of panorama canvas by finding the corners
    # of both images if overlaid w/ the translation
    all_corners = np.vstack((warped_corners, [[0,0],[0,h1],[w1,h1],[w1,0]]))
    [xmin, ymin] = np.int64(all_corners.min(axis=0).ravel())
    [xmax, ymax] = np.int64(all_corners.max(axis=0).ravel())

    # Translate so result starts at (0,0)
    translate = [-xmin, -ymin]
    Ht = np.array([[1,0,translate[0]], [0,1,translate[1]], [0,0,1]]) @ H

    # Warp img2 into panorama space, with alpha channel
    # based on the translation defined
    warped_img2 = cv2.warpPerspective(img2, Ht, (xmax-xmin, ymax-ymin))
    warped_alpha = cv2.warpPerspective(
        np.ones((h2, w2), dtype=np.uint8)*255, Ht, (xmax-xmin, ymax-ymin)
    )

    # Convert to BGRA 
    # we add this step to remove the black panels in between each image
    # to make the overlap smoother (makes blacks transparent)
    result = cv2.cvtColor(warped_img2, cv2.COLOR_BGR2BGRA)
    result[:,:,3] = warped_alpha  # set alpha from mask

    # Prepare img1 as BGRA
    img1_bgra = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)

    # Place img1 in correct offset based on the
    overlay = np.zeros_like(result, dtype=np.uint8)
    overlay[translate[1]:h1+translate[1], translate[0]:w1+translate[0]] = img1_bgra

    # Alpha-composite overlay on top of result
    mask = overlay[:,:,3] > 0
    result[mask] = overlay[mask]

    return result

def stitch_images(images):
    pano = images[0]
    for i in range(1, len(images)):
        kp1, desc1 = detect_features(pano)
        kp2, desc2 = detect_features(images[i])
        matches = match_features(desc1, desc2)
#        H, _ = compute_homography(kp1, kp2, matches)
        H, _ = compute_homography(kp1, kp2, matches, ransac_thresh=5.0)
        if H is None:
            raise RuntimeError(f"Not enough matches between image {i} and {i+1}")
        pano = stitch_two(pano, images[i], H)

    # === Crop panorama to align vertically with the first image ===
    h1, w1 = images[0].shape[:2]   # original first image size
    pano_h, pano_w = pano.shape[:2]

    # Compute center offset (if panorama taller than first image)
    if pano_h > h1:
        # take vertical crop that matches first image height, centered
        extra = pano_h - h1
        top = extra // 2
        bottom = top + h1
        pano = pano[top:bottom, :]
    
    return pano

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