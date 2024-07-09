from learner.resnet18_vgg16 import HandwrittenSymbolsClassifier
from PIL import Image
import glob
import torchvision.transforms as transforms


model_name = 'resnet' # or 'vgg'

classifier = HandwrittenSymbolsClassifier(root_dir='./learner/datasets/extracted_images/', epochs=5, model_type='resnet')

try:
    classifier.load_model(f'./learner/models/test_{model_name}.torch')
    print("Model loaded successfully")
except Exception as e:
    try:
        classifier.train()
        classifier.save_model("./learner/models/", f"test_{model_name}.torch")
        classifier.load_model(f'./learner/models/test_{model_name}.torch')
        print("Model trained and saved successfully")
    except Exception as e:
        print(e)



# testing

transformtosize = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_paths = glob.glob('tests/testfornew/**/*.png', recursive=True)

for image_path in image_paths:
    image = Image.open(image_path).convert('RGB')
    image = transformtosize(image)
    image = image.unsqueeze(0)
    prediction = classifier.predict(image)
    print(f'Prediction for {image_path}: {prediction}')
    # it should come cos,Beta,equalto,exclamation, three