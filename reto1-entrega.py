import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
import json
from PIL import Image


# Definir transformaciones de datos para las imágenes
transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Definir la estructura de la red neuronal
class MyCNN(nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.fc2 = nn.Linear(512, 8)
    
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Definir el modelo y los hiperparámetros
model = MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Cargar el conjunto de datos de entrenamiento
train_df = pd.read_csv('train.csv')
train_images = []
train_labels = []
for _, row in train_df.iterrows():
    img_path = row['path_img']
    label = row['label']
    img = Image.open(img_path).convert('RGB')
    img = transformations(img)
    train_images.append(img)
    train_labels.append(label)
train_dataset = torch.utils.data.TensorDataset(torch.stack(train_images), torch.LongTensor(train_labels))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Entrenamiento del modelo
model.train()
for epoch in range(50):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print('Epoch: %d, loss: %.3f' % (epoch + 1, running_loss / len(train_loader)))

# Cargar el conjunto de datos de prueba
test_df = pd.read_csv('test.csv')
test_images = []
test_labels = []
for _, row in test_df.iterrows():
    img_path = row['path_img']
    label = row['label']
    img = Image.open(img_path).convert('RGB')
    img = transformations(img)
    test_images.append(img)
    test_labels.append(label)
test_dataset = torch.utils.data.TensorDataset(torch.stack(test_images), torch.LongTensor(test_labels))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Evaluar el modelo en el conjunto de datos de prueba
model.eval()
predictions = {}
with torch.no_grad():
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(len(predicted)):
            predictions[str(i*16+j)] = predicted[j].item()

# Escribir las predicciones en un archivo JSON
with open('predictions.json', 'w') as f:
    json.dump({'target': predictions}, f)
