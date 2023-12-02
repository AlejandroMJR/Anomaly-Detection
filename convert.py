from PIL import Image
import os

# Set the input folder and output folder paths
input_folder = "/nas/home/ajaramillo/projects/datasets/raise"
output_folder = "/nas/home/ajaramillo/projects/datasets/raise/raise-jpeg"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate through all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".TIF"):
        # Open the TIFF image
        image = Image.open(os.path.join(input_folder, filename))

        # Convert and save as PNG
        base_filename = os.path.splitext(filename)[0]
        png_filename = f"{base_filename}.jpeg"
        image.save(os.path.join(output_folder, png_filename), "JPEG")
        print(f"Converted: {filename} to {png_filename}")

print("Conversion complete.")