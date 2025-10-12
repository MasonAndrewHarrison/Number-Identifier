import torch
import matplotlib.pyplot as plt
import numpy as np
from train import ConvNet
import pygame
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ConvNet().to(device)
model.load_state_dict(torch.load("CNN_Weights.pth"))
model.eval()

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 1000

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

current_stroke = []
strokes_list = []

clock = pygame.time.Clock()

running = True
mouse_x = 0
mouse_y = 0
left_clicked = False
right_clicked = False

def get_28x28_matrix():

    string_image = pygame.image.tostring(screen, 'RGB')
    temp_image = Image.frombytes('RGB', (SCREEN_WIDTH, SCREEN_HEIGHT), string_image)
    
    temp_image = temp_image.convert('L')
    temp_image = temp_image.resize((28, 28), Image.LANCZOS)

    matrix = np.array(temp_image) / 255.0
    return matrix

def prediction(matrix):

    matrix = get_28x28_matrix()
    tensor = torch.from_numpy(matrix).float()
    tensor = tensor.unsqueeze(0).unsqueeze(0).to(device)
    output = 0

    with torch.no_grad():
        output = model(tensor).to(device)

    _, predicted = torch.max(output, 1)

    return predicted.item()


def drawStroke(stroke):

    for i, point in enumerate(stroke):
        if len(stroke) > 1 and not i == 0:
            pygame.draw.line(screen, WHITE, stroke[i-1], point, 30)

while running:

    mouse_moved = False
    if_unleft_clicked = False
    space_pressed = False

    for event in pygame.event.get():

        if event.type == pygame.MOUSEMOTION:
            mouse_moved = True
            mouse_x, mouse_y = event.pos
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                left_clicked = True
            elif event.button == 3:
                right_clicked = True
        
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                space_pressed = True

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                left_clicked = False
                if_unleft_clicked = True
            elif event.button == 3:
                right_clicked = False

        if event.type == pygame.QUIT:
            running = False

    screen.fill(BLACK)

    if (left_clicked and mouse_moved): 
        current_stroke.append((mouse_x, mouse_y))

    elif if_unleft_clicked:
        strokes_list.append(current_stroke)
        current_stroke = []

    if right_clicked:
        strokes_list = []

    for stroke in strokes_list:
        drawStroke(stroke)

    drawStroke(current_stroke)

    matrix = get_28x28_matrix()
    output = prediction(matrix)
    print(output)

    if space_pressed:
        print(matrix.shape)
        plt.imshow(matrix, cmap='gray') 
        plt.show()
        
    pygame.display.update()

    clock.tick(60)

pygame.quit()