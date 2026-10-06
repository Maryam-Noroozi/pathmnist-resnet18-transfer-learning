# pathmnist-resnet18-transfer-learning
Fine-tuning ResNet18 for medical image classification on the PathMNIST dataset using PyTorch, achieving ~84.6% test accuracy.
# PathMNIST Classification using ResNet18 & PyTorch

## 📊 Results
* **Test Accuracy:** `84.64%`
* **Model:** ResNet18 (Pre-trained on ImageNet, fine-tuned)
* **Optimizer:** AdamW (`lr=0.001`)
* **Loss Function:** CrossEntropyLoss

---

## 🛠️ Code Implementation

### 2. Model Setup
```python
import torch
import torchvision.models

model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 9)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)


### 1. Data Transforms & Dataloaders
!pip install medmnist

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO, Evaluator

transform = transforms.Compose([
    transforms.Resize(224), transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

!pip install medmnist


data_flag='pathmnist'
info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

data_class = getattr(medmnist, info['python_class'])
train_dataset = data_class(split='train', transform=transform, download=True)
test_dataset = data_class(split='test', transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


### 3. Training Loop
num_epochs = 10
for epoch in range(num_epochs):
  model.train()
  running_loss = 0.0
  for images, labels in train_loader:
    images, labels = images.to(device), labels.to(device).squeeze()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
  epoch_loss = running_loss / len(train_loader)
  print(f"Epoch {epoch + 1} / {num_epochs} , loss: {epoch_loss:0.4f}")


### 4. Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
  for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device).squeeze()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test Accuracy:{accuracy:0.2f}%")



