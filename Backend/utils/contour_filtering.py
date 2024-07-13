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
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  
    output_folder = os.path.join('extracted_characters', image_id)
    clear_folder(output_folder)
    
    margin = 200  
    
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        x = max(x - margin // 2, 0)
        y = max(y - margin // 2, 0)
        w = min(w + margin, image.shape[1] - x)
        h = min(h + margin, image.shape[0] - y)
        cropped_character = image[y:y+h, x:x+w]
        cv2.imwrite(os.path.join(output_folder, f'{i}.png'), cropped_character)