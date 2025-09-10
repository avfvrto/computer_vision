import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath
from pathlib import Path

# -------------------------------
# 1️⃣ Load images
# -------------------------------
part1_fp = Path(r'homework01/files/partC/raw_images')
image_list = []

for file in sorted(part1_fp.iterdir()):
    img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if img is not None:
        image_list.append(img)

print(f"Loaded {len(image_list)} images.")

# -------------------------------
# 2️⃣ Resize all images to same size
# -------------------------------
# Use first image as reference
ref_h, ref_w = image_list[0].shape[:2]

resized_images = []
for img in image_list:
    if img.shape[:2] != (ref_h, ref_w):
        img_resized = cv2.resize(img, (ref_w, ref_h))
    else:
        img_resized = img.copy()
    resized_images.append(img_resized)

# -------------------------------
# 3️⃣ Interactive polygon selection for each image
# -------------------------------
def select_polygon(image, target_size):
    points = []

    def onselect(verts):
        nonlocal points
        points = verts
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Draw polygon around region, close when done")
    selector = PolygonSelector(ax, onselect)
    plt.show()

    # Create mask at original size
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)  # float32
    poly_path = MplPath(points)

    yy, xx = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    mask_flat = poly_path.contains_points(coords)
    mask[:] = mask_flat.reshape(mask.shape).astype(np.float32)  # ensure float32

    # Resize mask to target size
    mask_resized = cv2.resize(mask, (target_size[1], target_size[0]))  # width, height

    # Smooth edges
    mask_resized = cv2.GaussianBlur(mask_resized, (51, 51), 0)
    return mask_resized


# Store masks for all images
masks = []
for idx, img in enumerate(resized_images):
    print(f"Select region for image {idx+1}")
    mask = select_polygon(img, target_size=(ref_h, ref_w))
    masks.append(mask)

# -------------------------------
# 4️⃣ Composite images using masks
# -------------------------------
composite = np.zeros_like(resized_images[0], dtype=np.float32)
for img, mask in zip(resized_images, masks):
    composite += img.astype(np.float32) * mask[..., np.newaxis]

# Normalize for display
composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
plt.imshow(cv2.cvtColor(composite_uint8, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Final Composite")
plt.show()
