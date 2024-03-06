# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as profiler

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

# Instantiate the model, loss function, and optimizer
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model without optimization strategies
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Validation function
def validate(model, val_loader, criterion, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / total_samples
    return accuracy

# Function to log results in TensorBoard
def log_results(writer, epoch, loss, accuracy):
    writer.add_scalar('Loss/train', loss, epoch)
    writer.add_scalar('Accuracy/val', accuracy, epoch)

# Train and log results without optimizations
with SummaryWriter(log_dir='logs/original') as writer:
    for epoch in range(5):
        train(model, train_loader, criterion, optimizer, device)
        accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Accuracy: {accuracy}')
        log_results(writer, epoch, 0, accuracy)

# Apply optimization strategies

# A. Multi-process Data Loading
# Use multi-process data loading for faster data loading
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

# B. Memory Pinning
# Enable memory pinning for faster data transfer
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, pin_memory=True)

# C. Increase Batch Size
# Experiment with a larger batch size for improved GPU utilization
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, pin_memory=True)

# D. Reduce Host to Device Copy
# Use memory pinning and increase batch size to minimize copy overhead
train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, pin_memory=True)

# E. Set Gradients to None
# Directly set gradients to None for efficient zeroing of gradients
def zero_grad(model):
    for param in model.parameters():
        param.grad = None

# F. Automatic Mixed Precision (AMP)
# Utilize automatic mixed precision for faster training
scaler = torch.cuda.amp.GradScaler()

# G. Train in Graph Mode
# Enable torch.jit.graph mode for improved computational efficiency
model = torch.jit.script(model)

# Highlight the final results after optimizations
with SummaryWriter(log_dir='logs/optimized') as writer:
    for epoch in range(5):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # AMP: Scale the loss to prevent underflow
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()

            # AMP: Unscales the gradients and performs optimization
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        accuracy = validate(model, val_loader, criterion, device)
        print(f'Epoch {epoch + 1}, Loss: {total_loss}, Accuracy: {accuracy}')
        log_results(writer, epoch, total_loss, accuracy)
