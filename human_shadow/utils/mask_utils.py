import pdb 
import os
import cv2
import numpy as np

from scipy.ndimage import shift


def overlay_masks_for_debug(mask1, mask2, color1=[255,0,0], color2=[0,0,255], overlap_color=[255,255,0]):
    """
    Overlay two masks with different colors for visualization.
    Args
        mask1: np.array of shape (H, W, 3)
        mask2: np.array of shape (H, W, 3)
        color1: list of 3 ints
        color2: list of 3 ints
        overlap_color: list of 3 ints
    Returns
        overlay_img: np.array of shape (H, W, 3)
    """
    overlay_img = np.ones_like(mask1) * 255
    overlay_img[mask1[:,:,0] > 0] = color1
    overlay_img[mask2[:,:,0] == 0] = color2
    overlay_img[(mask1[:,:,0] > 0) & (mask2[:,:,0] == 0)] = overlap_color
    return overlay_img


# def shift_mask(image, max_shift_x, max_shift_y):
#     # Generate random shifts within the specified ranges
#     shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
#     shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
    
#     # Apply the shift only to the black mask (0 values in the image)
#     mask = (image == 0)  # True for black mask pixels
#     shifted_mask = shift(mask.astype(float), shift=[shift_y, shift_x], mode='nearest')
    
#     # Create the new image by applying the shifted mask to the original image
#     new_image = np.ones_like(image) * image.max()
#     new_image[shifted_mask > 1e-6] = 0
    
#     return new_image


def shift_mask(image, max_shift_x, max_shift_y):
    # Generate random shifts within the specified ranges
    shift_x = np.random.randint(-max_shift_x, max_shift_x + 1)
    shift_y = np.random.randint(-max_shift_y, max_shift_y + 1)
    
    # Apply the shift only to the black mask (0 values in the image)
    mask = (image == 0).astype(np.uint8)  # Binary mask where black pixels are 1
    
    # Create a blank canvas the same size as the original image
    shifted_mask = np.zeros_like(mask)
    
    # Calculate the region where the shifted mask will be placed
    start_x = max(0, shift_x)
    end_x = min(image.shape[1], image.shape[1] + shift_x)
    start_y = max(0, shift_y)
    end_y = min(image.shape[0], image.shape[0] + shift_y)
    
    # Determine the region to copy from the original mask
    src_start_x = max(0, -shift_x)
    src_end_x = min(image.shape[1], image.shape[1] - shift_x)
    src_start_y = max(0, -shift_y)
    src_end_y = min(image.shape[0], image.shape[0] - shift_y)
    
    # Copy the shifted mask onto the blank canvas
    shifted_mask[start_y:end_y, start_x:end_x] = mask[src_start_y:src_end_y, src_start_x:src_end_x]
    
    # Create the new image by applying the shifted mask to the original image
    new_image = np.ones_like(image) * image.max()
    new_image[shifted_mask > 1e-6] = 0
    
    return new_image


def resize_binary_image(image, new_size):
    # Resize the image
    resized_image = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_NEAREST)
    
    # Threshold the resized image to keep only 0 and 255
    _, binary_image = cv2.threshold(resized_image, 127, 255, cv2.THRESH_BINARY)
    
    return binary_image