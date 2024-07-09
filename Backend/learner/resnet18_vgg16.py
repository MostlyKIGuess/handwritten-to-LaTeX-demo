import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models import resnet18, vgg16, ResNet18_Weights, VGG16_Weights




class HandwrittenSymbolsClassifier:
    def __init__(self, root_dir, batch_size=64, lr=0.001, epochs=10, device=None, model_type='resnet'):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
        ])
        self.dataset = self.HandwrittenSymbolsDataset(root_dir, transform=self.transform)
        self.train_loader, self.test_loader = self._prepare_data_loaders()
        self.model = self.CNN(len(self.dataset.classes), model_type=model_type)
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.epoch_losses = []
        self.epoch_accuracies = []

    class HandwrittenSymbolsDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.classes = os.listdir(root_dir)
            self.images = []
            self.labels = []
            for idx, label in enumerate(self.classes):
                class_path = os.path.join(root_dir, label)
                for image in os.listdir(class_path):
                    self.images.append(os.path.join(class_path, image))
                    self.labels.append(idx)
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            image = Image.open(self.images[idx]).convert('RGB') 
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    class CNN(nn.Module):
        def __init__(self, num_classes, model_type='resnet'):
            super().__init__()
            if model_type == 'resnet':
                self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1) 
            elif model_type == 'vgg':
                self.model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            else:
                raise ValueError("Unsupported model type")
            
            if model_type == 'resnet':
                num_ftrs = self.model.fc.in_features
                self.model.fc = nn.Linear(num_ftrs, num_classes) 
            elif model_type == 'vgg':
                num_ftrs = self.model.classifier[6].in_features
                self.model.classifier[6] = nn.Linear(num_ftrs, num_classes)  

        def forward(self, x):
            return self.model(x)

    def _prepare_data_loaders(self):
        train_size = int(0.8 * len(self.dataset))
        test_size = len(self.dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, test_loader

    def train(self):
        for epoch in range(self.epochs):
            running_loss = 0.0
            for data in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss / len(self.train_loader)
            self.epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch + 1}, Loss: {epoch_loss}")
            self.evaluate(epoch)

    def evaluate(self, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        self.epoch_accuracies.append(accuracy)
        print(f'Accuracy after epoch {epoch + 1}: {accuracy}%')

    def save_model(self, folder, model_name):
        if not os.path.exists(folder):
            os.makedirs(folder)
        torch.save(self.model.state_dict(), os.path.join(folder, model_name))
        print('Model saved successfully.')

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        print('Model loaded successfully.')

    def plot_metrics(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, self.epochs + 1), self.epoch_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')

        plt.subplot(1, 2, 2)
        plt.plot(range(1, self.epochs + 1), self.epoch_accuracies, label='Accuracy', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy')
        plt.show()

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')  
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
        return self.dataset.classes[predicted.item()]
