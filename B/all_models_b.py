import torch.nn as nn
import torch.nn.functional as F

"""
        You can select any model to train.
        These models have been compared and results disaplyed in the report.
        Model 1 was the final model.
"""


#########################    Model 1 (Final Model)    #########################

# Define the CNN model
class BloodCellClassifier1(nn.Module):
    def __init__(self):
        super(BloodCellClassifier1, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128 channels
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8)  # Output layer (8 classes)

    def forward(self, x):
        # Convolutional + ReLU + Pooling + BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> ReLU -> Pool -> BatchNorm
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected Layers + Dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation, raw logits)
        
        return x




#########################    Model 5    #########################

# Define the CNN model
class BloodCellClassifier5(nn.Module):
    def __init__(self):
        super(BloodCellClassifier5, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 channels
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)  # Output layer (8 classes)

    def forward(self, x):
        # Convolutional + ReLU + Pooling + BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> ReLU -> Pool -> BatchNorm
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully Connected Layers + Dropout
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation, raw logits)
        
        return x




#########################    Model 3     #########################
    
# Define the CNN model
class BloodCellClassifier3(nn.Module):
    def __init__(self):
        super(BloodCellClassifier3, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128 channels
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8)  # Output layer (8 classes)

    def forward(self, x):
        # Convolutional + ReLU + Pooling + BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> ReLU -> Pool -> BatchNorm
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected Layers + Dropout
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # Output layer (no activation, raw logits)
        
        return x
    



#########################    Model 2 s    #########################

# Define the CNN model
class BloodCellClassifier2(nn.Module):
    def __init__(self):
        super(BloodCellClassifier2, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Input: 3 channels (RGB), Output: 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Output: 128 channels
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Adjust input size based on final feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8)  # Output layer (8 classes)

    def forward(self, x):
        # Convolutional + ReLU + Pooling + BatchNorm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> ReLU -> Pool -> BatchNorm
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> ReLU -> Pool -> BatchNorm
        
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 128 * 3 * 3)
        
        # Fully Connected Layers + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # Output layer (no activation, raw logits)
        
        return x
    