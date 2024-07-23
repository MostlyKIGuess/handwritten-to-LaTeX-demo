import cv2
import numpy as np
import os
import shutil

def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path)

def contour_filter(image_id):
    image = cv2.imread(f'uploads/{image_id}.png')
    if image is None:
        print("Error loading image")
        return
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3,3), np.uint8)  
    binary = cv2.dilate(binary, kernel, iterations=4) # Dilate to connect characters
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    output_folder = os.path.join('extracted_characters', image_id)
    clear_folder(output_folder)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 100:  # Ignore small contours
            continue
        cropped_character = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, f'{i}.png'), cropped_character)