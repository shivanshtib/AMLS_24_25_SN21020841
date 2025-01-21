# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Set directory paths
TASK_B_DIR = Path(__file__).resolve().parent.parent / "B"

# Import custom modules
import data_prep as dp # Custom data preparation module

# Define a Convolutional Neural Network Classifier
class BloodCellClassifier1(nn.Module):
    def __init__(self):
        super(BloodCellClassifier1, self).__init__()
        # Define convolutional layers with batch normalization and pooling layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # First conv layer taking 3-channel input
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Second conv layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Third conv layer
        self.bn1 = nn.BatchNorm2d(32)  # Batch norm for the first layer outputs
        self.bn2 = nn.BatchNorm2d(64)  # Batch norm for the second layer outputs
        self.bn3 = nn.BatchNorm2d(128)  # Batch norm for the third layer outputs
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling to reduce spatial dimensions
        self.fc1 = nn.Linear(128 * 3 * 3, 256)  # Fully connected layer, adjust input size based on output from conv layers
        self.fc2 = nn.Linear(256, 128)  # Second fully connected layer
        self.fc3 = nn.Linear(128, 8)  # Output layer for 8 classes

    def forward(self, x):
        # Forward pass through the network
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Apply conv1 -> ReLU -> BatchNorm -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Apply conv2 -> ReLU -> BatchNorm -> Pool
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Apply conv3 -> ReLU -> BatchNorm -> Pool
        x = x.view(-1, 128 * 3 * 3)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))  # First fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))  # Second fully connected layer with ReLU activation
        x = self.fc3(x)  # Output layer (logits, no activation)
        return x

# Function to evaluate the model's performance on a test dataset
def evaluate_model(test_loader, device):
    """
    Evaluates the performance of a trained PyTorch model on a test dataset using given DataLoader.

    Parameters:
        test_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        None. Outputs metrics and displays them as a DataFrame.
    """
    #Load the best saved model
    model = BloodCellClassifier1().to(device)
    model.load_state_dict(torch.load(TASK_B_DIR / "best_model_B_final.pth", weights_only=True))
    model.eval()  # Set the model to evaluation mode

    criterion = nn.CrossEntropyLoss() #Loss function for multi-classification
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    test_predictions = []
    test_targets = []
    test_probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).squeeze()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predictions = torch.max(outputs, 1)
            probabilities = torch.softmax(outputs, dim=1)
            test_probabilities.extend(probabilities.detach().cpu().numpy())
            correct_test += (predictions == labels).sum().item()
            total_test += labels.size(0)
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())

    # Calculate performance metrics
    test_loss /= len(test_loader)
    test_accuracy = correct_test / total_test
    test_f1 = f1_score(test_targets, test_predictions, average='macro')
    test_precision = precision_score(test_targets, test_predictions, average='macro')
    test_recall = recall_score(test_targets, test_predictions, average='macro')
    test_auc = roc_auc_score(test_targets, test_probabilities, multi_class='ovr')

    # Compile metrics into a DataFrame and display it
    metrics = {
        "Metric": ["Loss", "Accuracy", "F1 Score", "AUC",  "Precision", "Recall"],
        "Value": [f"{test_loss:.4f}", f"{test_accuracy:.4f}", f"{test_f1:.4f}", f"{test_auc:.4f}", f"{test_precision:.4f}", f"{test_recall:.4f}"]
    }
    metrics_df = pd.DataFrame(metrics)
    print("Test Split Performance:")
    print(metrics_df)

# Prepare data using a custom function from imported module
_,_,test_loader = dp.prep_dataB()

# Set device based on availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Function to run evaluation
def run():
    evaluate_model(test_loader, device)

