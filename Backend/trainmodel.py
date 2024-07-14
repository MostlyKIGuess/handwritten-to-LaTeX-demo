import glob
import cv2
import torchvision.transforms as transforms
from PIL import Image
from learner.resnet18_vgg16 import HandwrittenSymbolsClassifier

model_name = "resnet34"  # or 'vgg16', 'resnet50', 'resnet34', and 'resnet18'

classifier = HandwrittenSymbolsClassifier(
    root_dir="./learner/datasets/extracted_images/",
    epochs=5,
    batch_size=64,  # for vgg16 and resnet50 use 32, otherwise 64
    model_type=f"{model_name}",
)

try:
    classifier.load_model(f"./learner/models/test_{model_name}.torch")
    print("Model loaded successfully")
except Exception as e:
    try:
        def is_image_file(filename):
            valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
            return any(filename.lower().endswith(ext) for ext in valid_extensions)
        
        image_paths = glob.glob("./learner/datasets/extracted_images/**/*", recursive=True)
        count = 0
        preprocessed_images = []

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

            _, bw_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
            preprocessed_images.append((image_path, bw_image))
            count += 1

        print(f"Total preprocessed images: {count}")

        classifier.train()
        classifier.save_model("./learner/models/", f"test_{model_name}.torch")
        classifier.load_model(f"./learner/models/test_{model_name}.torch")
        print("Model trained and saved successfully")
    except Exception as e:
        print(e)


# Testing
image_paths = glob.glob("tests/testfornew/**/*", recursive=True)

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

    _, bw_image = cv2.threshold(gray_image, 220, 255, cv2.THRESH_BINARY)
    
    pil_image = Image.fromarray(bw_image).convert('RGB')
    prediction = classifier.predict(image_path=pil_image)
    print(f"Prediction for {image_path}: {prediction}")
    # it should come cos, Beta, equalto, exclamation, three
