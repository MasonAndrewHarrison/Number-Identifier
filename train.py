import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np 
import os


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, 3)
        self.conv2 = nn.Conv2d(2, 2, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2 * 5 * 5, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_epochs = 300
    batch_size = 100
    learning_rate = 0.01

    transform = transforms.Compose([
        transforms.ToTensor(),
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

    if os.path.exists("CNN_Weights.pth"):
        model.load_state_dict(torch.load("CNN_Weights.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] | Loss: {loss}')
        
        print_accuracy(model, test_loader)

    torch.save(model.state_dict(), "CNN_Weights.pth")



        
