import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def load_images(folder_path, img_size=300):
    """
    Loads all images from a specified folder, transforms, and stacks them into a batch tensor.

    Args:
        folder_path (str): The path to the folder containing images.
        img_size (int): The desired height and width to resize images to.

    Returns:
        torch.Tensor: A batch of image tensors with shape (N, C, H, W).
    """
    # Define the transformation pipeline
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Find all image files with ".jpg" extensions
    image_paths = glob.glob(os.path.join(folder_path, '*.jpg'))

    if not image_paths:
        raise ValueError(f"No .jpg images found in the directory: {folder_path}")

    # Load images, apply transformations, and collect them in a list
    images = [transform(Image.open(path)) for path in image_paths]

    # Stack the list of tensors into a single batch tensor
    image_batch = torch.stack(images)

    return image_batch


def display_grid(grid):
    """
    Displays a grid of images using matplotlib, clipping values to the valid range.

    Args:
        grid (torch.Tensor): A grid of images created by vutils.make_grid.
    """
    # Convert tensor to NumPy array and transpose from (C, H, W) to (H, W, C)
    grid_np = np.transpose(grid.numpy(), (1, 2, 0))

    # Clip the data to the valid display range [0, 1] for floats
    clipped_grid = np.clip(grid_np, 0, 1)

    # Display the clipped image
    plt.figure(figsize=(8, 8))
    plt.imshow(clipped_grid)
    plt.axis('off')
    plt.show()


def show_images(images, titles, cols=2):
    """
    Displays a list of images in a grid.

    Args:
        images (List): A list of the images to display.
        titles (Tuple, optional): A tuple of titles for each image. Defaults to None.
        cols (int, optional): The number of columns in the display grid. Defaults to 2.
    """
    # Calculate the number of rows needed to display all images
    num_images = len(images)
    rows = (num_images + cols - 1) // cols

    # Create a figure with an appropriate size
    plt.figure(figsize=(cols * 5, rows * 5))

    # Loop through the images and display them
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.imshow(image)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
    
def plot_histogram(tensor_image, normalized_tensor, title):
    """
    Plots a side-by-side comparison of histograms for each color channel
    of two image tensors.

    This function generates a 3x2 grid of plots. Each row corresponds to a
    color channel (Red, Green, Blue), and the columns show the pixel
    distribution before and after a transformation (e.g., normalization).

    Args:
        tensor_image (torch.Tensor): The image tensor before transformation.
        normalized_tensor (torch.Tensor): The image tensor after transformation.
        title (str): The main title for the entire figure.
    """
    # Isolate color channels for the 'before' tensor
    r_before = tensor_image[0, :, :].flatten()
    g_before = tensor_image[1, :, :].flatten()
    b_before = tensor_image[2, :, :].flatten()

    # Isolate color channels for the 'after' tensor
    r_after = normalized_tensor[0, :, :].flatten()
    g_after = normalized_tensor[1, :, :].flatten()
    b_after = normalized_tensor[2, :, :].flatten()

    # Create a 3x2 grid of subplots for the comparison
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Define consistent font sizes for plot elements
    TITLE_SIZE = 16
    LABEL_SIZE = 14
    TICK_SIZE = 12

    # Set the main title for the entire figure
    fig.suptitle(title, fontsize=TITLE_SIZE + 2, fontweight='bold')

    # --- Row 1: Red Channel Comparison ---
    axes[0, 0].hist(r_before.numpy(), bins=256, color='red', alpha=0.7)
    axes[0, 0].set_title("Red Channel - Before", fontsize=TITLE_SIZE)
    axes[0, 0].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[0, 0].set_ylabel("Frequency", fontsize=LABEL_SIZE)
    axes[0, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    axes[0, 1].hist(r_after.numpy(), bins=256, color='salmon', alpha=0.7)
    axes[0, 1].set_title("Red Channel - After", fontsize=TITLE_SIZE)
    axes[0, 1].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[0, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axes[0, 1].set_xlim(-3, 3)

    # --- Row 2: Green Channel Comparison ---
    axes[1, 0].hist(g_before.numpy(), bins=256, color='green', alpha=0.7)
    axes[1, 0].set_title("Green Channel - Before", fontsize=TITLE_SIZE)
    axes[1, 0].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[1, 0].set_ylabel("Frequency", fontsize=LABEL_SIZE)
    axes[1, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    axes[1, 1].hist(g_after.numpy(), bins=256, color='limegreen', alpha=0.7)
    axes[1, 1].set_title("Green Channel - After", fontsize=TITLE_SIZE)
    axes[1, 1].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[1, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axes[1, 1].set_xlim(-3, 3)

    # --- Row 3: Blue Channel Comparison ---
    axes[2, 0].hist(b_before.numpy(), bins=256, color='blue', alpha=0.7)
    axes[2, 0].set_title("Blue Channel - Before", fontsize=TITLE_SIZE)
    axes[2, 0].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[2, 0].set_ylabel("Frequency", fontsize=LABEL_SIZE)
    axes[2, 0].tick_params(axis='both', which='major', labelsize=TICK_SIZE)

    axes[2, 1].hist(b_after.numpy(), bins=256, color='cornflowerblue', alpha=0.7)
    axes[2, 1].set_title("Blue Channel - After", fontsize=TITLE_SIZE)
    axes[2, 1].set_xlabel("Pixel Intensity", fontsize=LABEL_SIZE)
    axes[2, 1].tick_params(axis='both', which='major', labelsize=TICK_SIZE)
    axes[2, 1].set_xlim(-3, 3)

    
    # Adjust subplot parameters for a clean layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Display the plots
    plt.show()
    
    