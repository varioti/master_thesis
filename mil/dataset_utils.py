import numpy as np
import pandas as pd
import os
import random
import shutil
import gc
import cv2
from PIL import Image, ImageOps

def discard_bg(img_path):
    # Load the image
    img = cv2.imread(img_path)

    # Convert the image from RGB to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Threshold the saturation channel
    mask = cv2.inRange(hsv, (0, 15, 0), (255, 255, 255))
    
    # Find connected components
    connectivity = 8
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)

    # Ignore very small connected components (less than 10 pixels)
    min_size = 8
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_size:
            mask[labels == i] = 0

    # Find the contours of the non-zero regions in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    x, y, w, h = cv2.boundingRect(np.concatenate(contours))
    cropped = img[y:y+h, x:x+w]

    # Show the cropped image
    cv2.imshow('Cropped', cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cropped

def create_validation_set(dataset_path, val_ratio):
    classes = ['normal', 'tumor']
    val_path = os.path.join(dataset_path, "val")
    train_path = os.path.join(dataset_path, "train")
    
    # Create val folder and subfolders for each class
    os.makedirs(val_path, exist_ok=True)
    for c in classes:
        os.makedirs(os.path.join(val_path, c), exist_ok=True)
    
    # Move random files to val folder
    for c in classes:
        files = os.listdir(os.path.join(train_path, c))
        random.shuffle(files)
        split_idx = int(len(files) * val_ratio)
        val_files = files[:split_idx]
        
        for f in val_files:
            src = os.path.join(train_path, c, f)
            dst = os.path.join(val_path, c, f)
            shutil.move(src, dst)
    
    print('Validation set created successfully!')


def pad_folder_to_square(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for filename in files:
            # Check if file is an image file
            if not filename.lower().endswith(('.png', '.jpg', '.tif')):
                continue
            
            # Get the input and output file paths
            input_path = os.path.join(root, filename)
            rel_path = os.path.relpath(input_path, input_folder)
            output_path = os.path.join(output_folder, rel_path)
            
            # Make sure the output directory exists
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # Load the image and get its dimensions
            img = cv2.imread(input_path)
            height, width, _ = img.shape
            
            # Determine the padding
            if height < width:
                pad_top = pad_bottom = (width - height) // 2
                pad_left = pad_right = 0
            else:
                pad_left = pad_right = (height - width) // 2
                pad_top = pad_bottom = 0
            
            # Pad the image and save it
            padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[255, 255, 255])
            cv2.imwrite(output_path, padded_img)

def organize_dataset(image_folder, csv_path):
    # Read the CSV file into a pandas dataframe
    df = pd.read_csv(csv_path)

    # Loop through each row of the dataframe
    for _, row in df.iterrows():
        # Get the image name and label from the row
        image_name = row[0]
        label = row[1]
        
        # Create the subfolder if it doesn't exist
        label_folder = os.path.join(image_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        # Get the full path to the image file
        image_path = os.path.join(image_folder, image_name)

        # Move the image file to the subfolder
        shutil.move(image_path, label_folder)

def resize_image(input_path, output_path, scale_factor=0.1):
    """
    Resizes an image to the specified size and saves it to the output path.
    """
    with Image.open(input_path) as img:
        try:
            width, height = img.size
            print(f"Size: [{width}, {height}]")
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"New size: [{new_width}, {new_height}]")
            resized_img = img.resize((new_width, new_height), resample=Image.BILINEAR)
            resized_img.save(output_path)
        except OSError as e:
            print(f"Error while processing image {input_path}: {str(e)}")
        finally:
            gc.collect()
    gc.collect()

def resize_images_in_folder(folder_path, output_folder, factor=0.1):
    """
    Resizes all .tif images in the folder and its subfolders by dividing each side by the specified factor and saves them to the output folder.
    """
    Image.MAX_IMAGE_PIXELS = None
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            print(filename)
            if filename.endswith(".tif"):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_folder, os.path.relpath(input_path, folder_path))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if not os.path.exists(output_path):
                    resize_image(input_path, output_path, factor)

            gc.collect()
        gc.collect()

def is_background(image, threshold=15):
    # Convert to the HSV color space
    image_hsv = image.convert("HSV")

    # Split the image into separate channels
    hue, saturation, value = image_hsv.split()

    # Calculate the average saturation value
    average_saturation = np.mean(saturation)

    # Check if the image contains only background
    if average_saturation < threshold:
        return True
    else:
        return False

def patchify_image(input_path, output_path, patch_size, threshold):
    """
    Transforms an image into multiple patches of a specific size and saves it to the output path.
    """
    with Image.open(input_path) as img:
        try:
            # Open the image
            img = Image.open(input_path)
            input_filename = os.path.basename(input_path)
            print(input_filename, flush=True)


            # Reduce the size of the image by 2
            img = img.resize((img.width // 2, img.height // 2), resample=Image.LANCZOS)

            # Generate up to 500 random patches
            max_patches = 500
            patch_count = 0
            patch_locations = []
            while patch_count < max_patches:
                # Generate a random patch location
                x = random.randint(0, img.width - patch_size)
                y = random.randint(0, img.height - patch_size)

                # Check if the patch location has already been generated
                if not ((x, y) in patch_locations):

                    # Crop the patch from the image
                    patch = img.crop((x, y, x + patch_size, y + patch_size))

                    # Check if the patch contains only background
                    if not is_background(patch, threshold):
                        # Save the patch to a file
                        output_filename = f"{output_path[:-4]}_{patch_count}.png"
                        patch.save(output_filename)

                        patch_count += 1
                        patch_locations.append((x, y))
                        gc.collect()
            print(patch_locations)

        except OSError as e:
            print(f"Error while processing image {input_path}: {str(e)}")
        finally:
            gc.collect()

def patchify_images_in_folder(folder_path, output_folder, patch_size=224, threshold=15):
    """
    Transforms all .tif images in the folder and its subfolders into a sequence of patches of a specific size.
    """
    Image.MAX_IMAGE_PIXELS = None
    files_exceptions = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            print(filename)
            if filename.endswith(".tif"):
                input_path = os.path.join(root, filename)
                output_path = os.path.join(output_folder, os.path.relpath(input_path, folder_path))
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                if not os.path.exists(f"{output_path[:-4]}_500.png"):
                    try:
                        patchify_image(input_path, output_path, patch_size, threshold)
                    except Exception as e:
                        print(f"{filename}: [{type(e)}] {e}")
                        files_exceptions.append(filename)
            gc.collect()
        gc.collect()

    print("Exceptions :")
    print(files_exceptions)

# Example usage
input_folder = "camelyon16/train/Normal/"
output_folder = "camelyon16_patch/train/Normal/"

patch_size = 384
threshold = 15

Image.MAX_IMAGE_PIXELS = None

patchify_images_in_folder(input_folder, output_folder, patch_size, threshold)

