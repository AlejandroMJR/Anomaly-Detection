import PIL
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

input_folder = "/nas/home/ajaramillo/projects/datasets/dalle/mask-jpeg"
output_folder = "/nas/home/ajaramillo/projects/datasets/dalle/mask-jpeg"

for filename in os.listdir(input_folder):
    if filename.endswith(".PNG") or filename.endswith(".png"):

        # Open the image
        image = Image.open(os.path.join(input_folder, filename)).convert('L')
        numpy_image = np.array(image)
        numpy_image[1024:,:] = 255
        numpy_image[:,1024:] = 255
        numpy_image[numpy_image > 0] = 255
        numpy_image = 255 - numpy_image
        im = Image.fromarray(numpy_image)
        base_filename = os.path.splitext(filename)[0]
        mask_filename = f"{base_filename}-mask.png"
        im.save(os.path.join(output_folder, mask_filename), "PNG")
        print("check")
