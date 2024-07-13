import cv2
import numpy as np
import os
import shutil

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def contour_filter(image_id, padding=10):
    image = cv2.imread(f'uploads/{image_id}.png')
    if image is None:
        print("Error loading image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_folder = os.path.join('extracted_characters', image_id)
    clear_folder(output_folder)
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate new bounding box with padding
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
        h_padded = min(image.shape[0] - y_padded, h + 2 * padding)
        
        cropped_character = image[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]
        
        # Create a blank image with extra whitespace
        blank_image = np.full((h_padded + 2 * padding, w_padded + 2 * padding, 3), 255, dtype=np.uint8)
        start_y = padding
        start_x = padding
        blank_image[start_y:start_y+h_padded, start_x:start_x+w_padded] = cropped_character
        
        # Resize the character to the desired size
        resized_character = cv2.resize(blank_image, (45, 45))
        
        cv2.imwrite(os.path.join(output_folder, f'{i}.png'), resized_character)