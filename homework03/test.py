import cv2

# load your image
img = cv2.imread("/Users/ainee_f/Documents/School - Docs/computer_vision/homework03/selected_photos/PXL_20251015_204106548.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# try both common patterns; adjust as needed
pattern_sizes = [(7,5), (8,6)]   # (cols, rows) of INNER corners

for ps in pattern_sizes:
    ok, corners = cv2.findChessboardCorners(gray, ps)
    print(f"{ps}: found={ok}, count={0 if corners is None else len(corners)}")

    if ok:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, ps, corners, ok)
        cv2.imwrite(f"corners_{ps[0]}x{ps[1]}.jpg", vis)
        print(f"Saved visualization as corners_{ps[0]}x{ps[1]}.jpg")