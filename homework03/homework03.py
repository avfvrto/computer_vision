import glob
import numpy as np
import cv2

# ---------- Utilities ----------
def normalize_points_2d(pts):
    """Normalize 2D points so centroid is at origin and mean distance is sqrt(2)."""
    pts = np.asarray(pts)
    mean = np.mean(pts, axis=0)
    d = np.mean(np.sqrt(np.sum((pts - mean)**2, axis=1)))
    s = np.sqrt(2) / d
    T = np.array([[s, 0, -s*mean[0]],
                  [0, s, -s*mean[1]],
                  [0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    pts_n = (T @ pts_h.T).T[:, :2]
    return pts_n, T

def normalize_points_3d(pts):
    """Normalize 3D points so centroid is at origin and mean distance is sqrt(3)."""
    pts = np.asarray(pts)
    mean = np.mean(pts, axis=0)
    d = np.mean(np.sqrt(np.sum((pts - mean)**2, axis=1)))
    s = np.sqrt(3) / d
    T = np.array([[s, 0, 0, -s*mean[0]],
                  [0, s, 0, -s*mean[1]],
                  [0, 0, s, -s*mean[2]],
                  [0, 0, 0, 1]])
    pts_h = np.c_[pts, np.ones(len(pts))]
    pts_n = (T @ pts_h.T).T[:, :3]
    return pts_n, T


# in this stage, a homography mapping the real world points to the image plane.
# RELATION TO NOTES: Perspective projection to get homogenous coordinates of (u,v)
def homography_dlt(world_pts_xy, image_pts):
    """
    Compute H (3x3) s.t. lambda*[u v 1]^T = H * [X Y 1]^T
    world_pts_xy: Nx2, image_pts: Nx2
    """
    wp = np.asarray(world_pts_xy)
    ip = np.asarray(image_pts)
    
    # Normalize for numeric stability (2D -> 2D)
    wp_n, T_w = normalize_points_2d(wp)
    ip_n, T_i = normalize_points_2d(ip)

    N = wp.shape[0]
    
    A = []
    for i in range(N):
        X, Y = wp_n[i]
        u, v = ip_n[i]
        A.append([0, 0, 0, -X, -Y, -1, v*X, v*Y, v])
        A.append([X, Y, 1,  0,  0,  0, -u*X, -u*Y, -u])
    A = np.asarray(A)

    # Solve Ah=0 (SVD)
    _, _, VT = np.linalg.svd(A)
    h = VT[-1, :]
    H_n = h.reshape(3, 3)

    # Denormalize
    H = np.linalg.inv(T_i) @ H_n @ T_w
    H /= H[2, 2]
    return H

# ---------- Zhang’s method for K from multiple homographies ----------

# 
def v_ij(H, i, j):
    """Helper to form Zhang's linear constraints v_ij from a homography."""
    h = H.T  # columns are h1,h2,h3
    return np.array([
        h[i,0]*h[j,0],
        h[i,0]*h[j,1] + h[i,1]*h[j,0],
        h[i,1]*h[j,1],
        h[i,2]*h[j,0] + h[i,0]*h[j,2],
        h[i,2]*h[j,1] + h[i,1]*h[j,2],
        h[i,2]*h[j,2]
    ])

def intrinsics_from_homographies(H_list):
    """
    Solve for B = K^{-T} K^{-1} (6 unknowns, symmetric) using linear system from multiple H.
    Then recover K by Cholesky / inverse.
    """
    V = []
    for H in H_list:
        V.append(v_ij(H, 0, 1))                 # v01
        V.append(v_ij(H, 0, 0) - v_ij(H, 1, 1)) # v00 - v11
    V = np.vstack(V)

    # Solve V b = 0
    _, _, VT = np.linalg.svd(V)
    b = VT[-1, :]  # 6-vector: [B11, B12, B22, B13, B23, B33]

    B11, B12, B22, B13, B23, B33 = b
    # Recover K from B (see Zhang 2000)
    v0 = (B12*B13 - B11*B23) / (B11*B22 - B12**2)
    lam = B33 - (B13**2 + v0*(B12*B13 - B11*B23)) / B11
    alpha = np.sqrt(lam / B11)
    beta  = np.sqrt(lam * B11 / (B11*B22 - B12**2))
    gamma = -B12 * alpha**2 * beta / lam
    u0    = gamma * v0 / beta - B13 * alpha**2 / lam

    K = np.array([[alpha, gamma, u0],
                  [0,     beta,  v0],
                  [0,     0,     1]])
    return K

# ---------- Extrinsics from a homography ----------
def extrinsics_from_H(K, H):
    """
    For a planar board at Z=0:
      [r1 r2 t] = lambda * K^{-1} H
      r3 = r1 x r2; enforce orthonormality; det(R)=+1
    """
    Kinv = np.linalg.inv(K)
    h1, h2, h3 = H[:,0], H[:,1], H[:,2]
    lam = 1.0 / np.linalg.norm(Kinv @ h1)
    r1 = lam * (Kinv @ h1)
    r2 = lam * (Kinv @ h2)
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])

    # Orthonormalize R via SVD (closest rotation)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:,2] *= -1  # fix handedness

    t = lam * (Kinv @ h3)
    return R, t

# ---------- Optional: full 3D->2D DLT for P ----------
def projection_matrix_dlt(XYZ, uv):
    """
    Solve P (3x4) from many 3D-2D correspondences by linear DLT.
    """
    XYZ = np.asarray(XYZ); uv = np.asarray(uv)
    # Normalize 3D and 2D for stability
    XYZn, T3 = normalize_points_3d(XYZ)
    uvn,  T2 = normalize_points_2d(uv)

    A = []
    for (X,Y,Z), (u,v) in zip(XYZn, uvn):
        Xh = [X, Y, Z, 1]
        A.append([0,0,0,0,  *(-u*np.array(Xh)),  v*X, v*Y, v*Z, v])
        A.append([*Xh,      0,0,0,0,            -u*X, -u*Y, -u*Z, -u])
    A = np.asarray(A)

    _, _, VT = np.linalg.svd(A)
    p = VT[-1,:]
    Pn = p.reshape(3,4)

    # Denormalize
    P = np.linalg.inv(T2) @ Pn @ T3
    P /= P[-1,-1]
    return P

def K_R_t_from_P(P):
    """
    Decompose P = K [R|t] using RQ decomposition on left 3x3.
    Ensure K has positive diagonal; fix signs into R.
    """
    M = P[:, :3]
    # RQ via QR on reversed axes
    R_q, Q_r = np.linalg.qr(np.flipud(M).T)
    Rq = np.flipud(R_q.T); Qr = Q_r.T[:, ::-1]
    K = Rq
    R = Qr

    # Normalize K: positive diag
    T = np.diag(np.sign(np.diag(K)))
    K = K @ T
    R = T @ R

    # Normalize K so K[2,2]=1
    K = K / K[2,2]
    t = np.linalg.inv(K) @ P[:,3]
    return K, R, t

# ---------- Corner detection & world points ----------
def make_world_points(inner_corners_cols, inner_corners_rows, square_size_m):
    """
    Returns Nx3 world points on Z=0 plane, ordered row-major to match cv2.findChessboardCorners.
    """
    objp = []
    for r in range(inner_corners_rows):
        for c in range(inner_corners_cols):
            objp.append([c*square_size_m, r*square_size_m, 0.0])
    return np.asarray(objp, dtype=np.float64)

def detect_corners(image_path, pattern_size):
    """
    Returns (ok, corners Nx2) subpixel-refined if found.
    pattern_size: (cols, rows) of INNER corners, e.g., (7,5).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to read {image_path}"
    ok, corners = cv2.findChessboardCorners(img, pattern_size,
                                            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return False, None
    # Subpixel refine
    corners = cv2.cornerSubPix(img, corners, (11,11), (-1,-1),
                               (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
    return True, corners.reshape(-1,2)

# ---------- Main calibration pipeline ----------
def calibrate_from_planar(images_glob,
                          inner_cols=8, inner_rows=6,
                          square_size_m=0.03,
                          cam_height_m=None):
    # 1) Collect correspondences for each image, compute H
    obj_xy = make_world_points(inner_cols, inner_rows, square_size_m)[:, :2]  # drop Z
    H_list, corner_sets = [], []
    image_paths = sorted(glob.glob(images_glob))
    assert image_paths, f"No images match: {images_glob}"

    for p in image_paths:
        ok, img_pts = detect_corners(p, (inner_cols, inner_rows))
        if not ok:
            print(f"[WARN] Corners not found in {p}, skipping.")
            continue
        H = homography_dlt(obj_xy, img_pts)
        H_list.append(H)
        corner_sets.append((p, img_pts))
        print(f"[OK] Homography from {p}")

    assert len(H_list) >= 2, "Need at least 2 views to solve intrinsics (more is better)."

    # 2) Intrinsics from homographies
    K = intrinsics_from_homographies(H_list)
    print("\nEstimated intrinsics K:\n", K)

    # 3) Extrinsics per image from each H
    extrinsics = []
    for (p, img_pts), H in zip(corner_sets, H_list):
        R, t = extrinsics_from_H(K, H)
        extrinsics.append((p, R, t))

    # 4) Fix translation scale with measured camera height (optional but recommended)
    # World frame assumption: plane is Z=0, +Z points "up" from plane to camera.
    # Camera center C = -R^T t. We want C_z = cam_height_m.
    if cam_height_m is not None:
        # Use the first view to set a consistent scale s for all t's (H gives t up to scale).
        p0, R0, t0 = extrinsics[0]
        C0 = -R0.T @ t0
        s = cam_height_m / C0[2]  # scale to match the measured height
        print(f"\nApplying absolute scale using cam_height = {cam_height_m:.4f} m; scale s = {s:.6f}")
        extrinsics = [(p, R, s*t) for (p, R, t) in extrinsics]

    # 5) Build P_i = K [R|t]
    projections = []
    for (p, R, t) in extrinsics:
        P = K @ np.hstack([R, t.reshape(3,1)])
        projections.append((p, P))

    return K, extrinsics, projections

# ---------- Example CLI ----------
if __name__ == "__main__":
    # EDIT THESE:
    images_glob = "data/checkerboard/*.jpg"   # change to your folder/pattern
    inner_cols, inner_rows = 7, 5             # for an 8x6 squares board
    square_size_m = 0.03                      # 30 mm
    cam_height_m = 0.75                       # <-- put YOUR measured camera-to-plane height (meters). Or None.

    K, extrinsics, projections = calibrate_from_planar(
        images_glob,
        inner_cols, inner_rows,
        square_size_m,
        cam_height_m
    )

    print("\n=== RESULTS ===")
    print("K:\n", K)
    for p, R, t in extrinsics:
        C = -R.T @ t
        print(f"\nImage: {p}")
        print("R:\n", R)
        print("t:", t)
        print("Camera center (world):", C)