import cv2
import numpy as np
# import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
from matplotlib.path import Path as MplPath
from pathlib import Path

# Here I load several images of vehicles oriented similarly that I found
# I used five images and loaded them into a list
part1_fp = Path(r'/Users/ainee_f/Documents/school_docs/computer_vision/homework01/files/partC/raw_images')
image_list = []

for file in sorted(part1_fp.iterdir()):
    img = cv2.imread(str(file), cv2.IMREAD_COLOR)
    if img is not None:
        image_list.append(img)

print(f"Loaded {len(image_list)} images.")

# next, I do image resizing to make sure that I'm able to create masks that are consistent
# with the size of the base image that I am going to multiply with
def resize_with_padding(img, target_size):

    h, w = img.shape[:2]
    target_h, target_w = target_size

    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Compute padding
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    # Pad image
    resized_padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )
    return resized_padded

# Use first image as reference size
ref_h, ref_w = image_list[0].shape[:2]
resized_images = [resize_with_padding(img, (ref_h, ref_w)) for img in image_list]

# In order to select the regions that I want to blend together, I want to use
# an interactive selection tool that lets me pick out points on matplotlib. These points
# will make a polygon that will be "masked" out of the previous images and blended into the final
# image to make a new car!

# TRY: Bring all images to canonical size

def select_polygon(image, target_size):
    points = []

    def onselect(verts):
        nonlocal points
        points = verts
        plt.close()

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax.set_title("Draw polygon around region, close when done")
    PolygonSelector(ax, onselect)
    plt.show()

    # Create mask at original size
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    poly_path = MplPath(points)

    yy, xx = np.meshgrid(np.arange(mask.shape[0]), np.arange(mask.shape[1]), indexing='ij')
    coords = np.stack((xx.ravel(), yy.ravel()), axis=-1)
    mask_flat = poly_path.contains_points(coords)
    mask[:] = mask_flat.reshape(mask.shape).astype(np.float32)

    # Resize mask to target size (with float32 type)
    mask_resized = cv2.resize(mask, (target_size[1], target_size[0]))
    mask_resized = cv2.GaussianBlur(mask_resized, (51, 51), 0)
    return mask_resized

# Store masks
masks = []
for idx, img in enumerate(resized_images):
    print(f"Select region for image {idx+1}")
    mask = select_polygon(img, target_size=(ref_h, ref_w))
    masks.append(mask)

# the final step is to combine all the images into a composite by putting together
# different parts of the car that were masked out. Here, we start by initializing a composite
# image with zeroes.

composite = np.zeros_like(resized_images[0], dtype=np.float32)

# in this loop, we add the masks from the resized images onto the composite by simply adding it
# onto the composite array initialized earlier. we loop through each combination of mask & image source
# there is some transition / blur applied to make the image smoother and more aesthetically
# pleasing to look at 
for img, mask in zip(resized_images, masks):
    composite += img.astype(np.float32) * mask[..., np.newaxis]

# the composite is then normalized so that it can be displayed with matplotlib
composite_uint8 = np.clip(composite, 0, 255).astype(np.uint8)
plt.imshow(cv2.cvtColor(composite_uint8, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Final Composite")
plt.show()

# NOTE: I couldn't figure out why my aspect ratio resizing was not working, even using external resources like
# the web and GPT tools. I also see that the composites are still imperfect due to the mis-matched aspect ratios.

# Lastly, I wasn't able to register the images. I couldn't figure out a way to use feature-matching advanced enough
# to perfectly align their features (ex: wheel to wheel, hood-to-hood, etc per image.)

# I generated three trials of differen car combinations. I realized that using an interactive selection tool was easier
# than trying to build polygons manually with coordinates. 