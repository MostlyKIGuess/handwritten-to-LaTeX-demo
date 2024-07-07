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

class HandwrittenSymbolsClassifier:
    def __init__(self, root_dir, batch_size=64, lr=0.001, epochs=10, device=None):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.dataset = self.HandwrittenSymbolsDataset(root_dir, transform=self.transform)
        self.train_loader, self.test_loader = self._prepare_data_loaders()
        self.model = self.CNN(len(self.dataset.classes))
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
            image = Image.open(self.images[idx]).convert('L')
            label = self.labels[idx]
            if self.transform:
                image = self.transform(image)
            return image, label

    class CNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
            self.fc1 = nn.Linear(64 * 7 * 7, 1000)
            self.fc2 = nn.Linear(1000, num_classes)
        
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 64 * 7 * 7)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

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
        image = Image.open(image_path).convert('L')
        image = self.transform(image).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs.data, 1)
        return self.dataset.classes[predicted.item()]
