import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
import os
import random

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.20)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 50)
        self.bn3 = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 25)
        self.fc4 = nn.Linear(25, 10)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(self.pool(F.relu(self.conv2(x))))
        x = self.pool(F.relu(self.bn2(self.conv3(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_epochs = 10
    batch_size = 64
    learning_rate = 0.01

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=30,  
            translate=(0.2, 0.2),
            scale=(0.8, 1.2)
        ),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, 
                                download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, 
                                download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def print_accuracy(model, test_loader):

        model.eval()
        with torch.no_grad():

            n_samples = 0
            n_correct = 0

            for i, (images, labels) in enumerate(test_loader):

                images = images.to(device)
                labels = labels.to(device)

                output = model(images)

                _, predicted = torch.max(output, 1)

                n_samples += labels.size(0)
                n_correct += (labels == predicted).sum().item()
                
            accuracy = 100.0 * n_correct / n_samples
            print(f"Accuracy: {accuracy} %")

    model = ConvNet().to(device)

    def check_rand_img(model, test_loader):

        model.eval()
        with torch.no_grad():

            random_num = random.randint(0, len(test_loader.dataset) - 1)
            random_image,_ = test_loader.dataset[random_num]

            random_image = random_image.to(device).unsqueeze(0)

            output = model(random_image)
            _, predicted = torch.max(output, 1)

            print(predicted.item())

            plt.imshow(random_image[0, 0,: , :].cpu(), cmap='gray') 
            plt.show()
        

    if os.path.exists("CNN_Weights.pth"):
        model.load_state_dict(torch.load("CNN_Weights.pth"))
        check_rand_img(model, test_loader)
        print_accuracy(model, test_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] | Loss: {loss}')
        
        print_accuracy(model, test_loader)
        torch.save(model.state_dict(), "CNN_Weights.pth")



        
