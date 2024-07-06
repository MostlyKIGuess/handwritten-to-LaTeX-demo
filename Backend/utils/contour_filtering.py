import cv2
import numpy as np
import os
import shutil

output_folder = 'extracted_characters'

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
    
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    clear_folder(output_folder)
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        cropped_character = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, f'character_{i}.png'), cropped_character)