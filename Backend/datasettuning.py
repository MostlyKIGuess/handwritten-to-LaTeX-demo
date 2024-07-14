import cv2
import glob

def is_image_file(filename):
    valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    return any(filename.lower().endswith(ext) for ext in valid_extensions)


image_paths = glob.glob("./learner/datasets/extracted_images/**/*", recursive=True)

count = 0

for image_path in image_paths:
    if not is_image_file(image_path):
        continue

    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load image: {image_path}")
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(image, (45, 45), interpolation=cv2.INTER_AREA)
    if len(resized_image.shape) == 3:
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = resized_image

    _, bw_image = cv2.threshold(gray_image, 220, 256, cv2.THRESH_BINARY)
    count+=1
    cv2.imwrite(image_path, bw_image)


print(count)