import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

"""
        You can only select Model 4,5,6,9 to train in task_a_train.py
        This is because their output is one logit which is compatible with BCEWithLogitsLoss function.
        The other Models have two logit output which I used CrossEntropyLoss for. 
        But they performed worse so I chose one logit output in the end.
        Model 9 is the final model.
"""

#########################    Model 1 (Baseline Model)    #########################

# Define the CNN model
class BreastCancerClassifier1(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.fc1(x))  # Fully connected -> ReLU
        x = self.fc2(x)  # Final output
        return x
    



#########################    Model 2     #########################

class BreastCancerClassifier2(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.fc2(x)  # Final output
        return x
    



#########################    Model 7     #########################

class BreastCancerClassifier7(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier7, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # First fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        
        self.fc_middle = nn.Linear(128, 64)  # Newly added FC layer with 64 outputs
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm for the middle FC layer
        
        self.fc2 = nn.Linear(64, 2)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = F.relu(self.bn4(self.fc_middle(x)))  # Middle FC layer -> BatchNorm -> ReLU
        x = self.fc2(x)  # Final output
        return x




#########################    Model  8    #########################

class BreastCancerClassifier8(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier8, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # First fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.dropout1 = nn.Dropout(0.5)  # Dropout after FC1 with a 50% probability
        
        self.fc_middle = nn.Linear(128, 64)  # Newly added FC layer with 64 outputs
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm for the middle FC layer
        self.dropout2 = nn.Dropout(0.5)  # Dropout after middle FC layer with a 50% probability
        
        self.fc2 = nn.Linear(64, 2)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.dropout1(x)  # Apply Dropout after FC1
        
        x = F.relu(self.bn4(self.fc_middle(x)))  # Middle FC layer -> BatchNorm -> ReLU
        x = self.dropout2(x)  # Apply Dropout after middle FC layer
        
        x = self.fc2(x)  # Final output
        return x
    


#########################    Model 4     #########################

class BreastCancerClassifier4(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier4, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.fc2 = nn.Linear(128, 1)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.fc2(x)  # Final output
        return x




#########################    Model 9 (Final Model)     #########################

class BreastCancerClassifier9(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier9, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm for Conv1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm for Conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.fc2 = nn.Linear(128, 1)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.fc2(x)  # Final output
        return x




#########################    Model 5     #########################

class BreastCancerClassifier5(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier5, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer (Input: 32 * 7 * 7, Output: 128)
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.fc2 = nn.Linear(128, 64)  # New Fully connected layer (Input: 128, Output: 64)
        self.bn4 = nn.BatchNorm1d(64)  # BatchNorm for new FC2 layer
        self.fc3 = nn.Linear(64, 1)  # Final output layer (Input: 64, Output: 1)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = F.relu(self.bn4(self.fc2(x)))  # New FC2 -> BatchNorm -> ReLU
        x = self.fc3(x)  # Final output
        return x




#########################    Model 6     #########################

class BreastCancerClassifier6(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier6, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # New Conv layer: Input: 32 channels, Output: 64 channels
        self.bn3 = nn.BatchNorm2d(64)  # BatchNorm for Conv3
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Fully connected layer (update input size after additional conv layer)
        self.bn4 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        
        self.fc2 = nn.Linear(128, 1)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = F.relu(self.bn2(self.conv2(x)))  # Conv2 -> BatchNorm -> ReLU
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Conv3 -> BatchNorm -> ReLU -> Pool
        
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor (update dimensions after additional conv layer)
        x = F.relu(self.bn4(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.fc2(x)  # Final output
        return x




#########################    Model 3     #########################

class BreastCancerClassifier3(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier3, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)  # Input: 1 channel, Output: 16 channels
        self.bn1 = nn.BatchNorm2d(16)  # BatchNorm for Conv1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # Input: 16 channels, Output: 32 channels
        self.bn2 = nn.BatchNorm2d(32)  # BatchNorm for Conv2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        self.dropout1 = nn.Dropout(p=0.3)  # Dropout after convolutional layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # Fully connected layer
        self.bn3 = nn.BatchNorm1d(128)  # BatchNorm for FC1
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout after FC1
        self.fc2 = nn.Linear(128, 2)  # Output layer (2 classes: benign, malignant)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Conv1 -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Conv2 -> BatchNorm -> ReLU -> Pool
        x = self.dropout1(x)  # Apply dropout after pooling
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = F.relu(self.bn3(self.fc1(x)))  # FC1 -> BatchNorm -> ReLU
        x = self.dropout2(x)  # Apply dropout after FC1
        x = self.fc2(x)  # Final output
        return x
    
"""
        I added the ResNet models to check their performance 
        I have commented it out because it shows some warning messages
"""

# # ResNet18
# class ResNet18Classifier(nn.Module):
#     def __init__(self):
#         super(ResNet18Classifier, self).__init__()
#         self.base_model = models.resnet18(pretrained=True)
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Adjust output layer for 2 classes

#     def forward(self, x):
#         return self.base_model(x)

# resnet18_model = ResNet18Classifier()
# resnet18_model.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# # Optionally freeze all layers except the final one (feature extraction)
# for param in resnet18_model.base_model.parameters():
#     param.requires_grad = False

# # Unfreeze the final fully connected layer
# resnet18_model.base_model.fc.weight.requires_grad = True
# resnet18_model.base_model.fc.bias.requires_grad = True

# # ResNet50
# class ResNet50Classifier(nn.Module):
#     def __init__(self):
#         super(ResNet50Classifier, self).__init__()
#         self.base_model = models.resnet50(pretrained=True)
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 2)  # Adjust output layer for 2 classes

#     def forward(self, x):
#         return self.base_model(x)

# resnet50_model = ResNet50Classifier()
# resnet50_model.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# # Optionally freeze all layers except the final one (feature extraction)
# for param in resnet50_model.base_model.parameters():
#     param.requires_grad = False

# # Unfreeze the final fully connected layer
# resnet50_model.base_model.fc.weight.requires_grad = True
# resnet50_model.base_model.fc.bias.requires_grad = True