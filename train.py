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
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.20)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x)))) 
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = F.relu(self.bn3(self.conv3(x)))             
        x = self.pool(F.relu(self.bn4(self.conv4(x)))) 
        x = self.dropout1(x)
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    num_epochs = 1000
    batch_size = 64
    learning_rate = 0.005

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=30,  
            translate=(0.3, 0.3),
            scale=(0.8, 1.15)
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

            for _, (images, labels) in enumerate(test_loader):

                images = images.to(device)
                labels = labels.to(device)

                output = model(images)

                _, predicted = torch.max(output, 1)

                n_samples += labels.size(0)
                n_correct += (labels == predicted).sum().item()
                
            accuracy = 100.0 * n_correct / n_samples
            print(f"Accuracy: {accuracy:.2f} %")
            return accuracy

    

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
        
    model = ConvNet().to(device)

    if os.path.exists("CNN_Weights.pth"):
        model.load_state_dict(torch.load("CNN_Weights.pth", map_location=device))
        model.to(device)
        check_rand_img(model, test_loader)
        print_accuracy(model, test_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if i % 200 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] | Loss: {loss}')
        
        
        accuracy = print_accuracy(model, test_loader)
        scheduler.step()
        torch.save(model.state_dict(), "CNN_Weights.pth")



        
