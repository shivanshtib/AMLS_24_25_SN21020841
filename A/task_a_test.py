# Import necessary libraries and modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Set directory paths
TASK_A_DIR = Path(__file__).resolve().parent.parent / "A"

# Import custom modules
import data_prep as dp # Custom data preparation module

# Define the CNN model for breast cancer classification
class BreastCancerClassifier9(nn.Module):
    def __init__(self):
        super(BreastCancerClassifier9, self).__init__()
        # Convolutional layer 1: input has 1 channel, output has 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization after conv1
        
        # Convolutional layer 2: input has 32 channels, output has 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization after conv2
        
        # Max pooling layer to downsample the feature maps by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layer: input features are flattened to match (64 * 7 * 7), output is 128
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.bn3 = nn.BatchNorm1d(128)  # Batch normalization after fc1
        
        # Fully connected layer: output is 1 (binary classification: benign/malignant)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # Forward pass through the first convolutional layer -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Forward pass through the second convolutional layer -> BatchNorm -> ReLU -> Pool
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the output to feed into the fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        
        # Fully connected layer 1 -> BatchNorm -> ReLU
        x = F.relu(self.bn3(self.fc1(x)))
        
        # Final output layer (logits for binary classification)
        x = self.fc2(x)
        return x

# Function to evaluate the model on a test dataset
def evaluate_model(test_loader):
    """
    Evaluates the performance of the CNN model on a test dataset.

    Parameters:
    - test_loader: DataLoader object for the test dataset.

    Returns:
    - None. Displays metrics and a confusion matrix.
    """
    device='cpu' #as dataset is small, cpu is faster than gpu
    # Define the positive weight for the binary classification loss (adjusted based on dataset imbalance)
    pos_weight = torch.tensor(0.4 / 0.9, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Loss function for binary classification

    # Initialize the model
    model = BreastCancerClassifier9()



    # Load the saved model weights
    model.load_state_dict(torch.load(
        TASK_A_DIR / "best_model_A_final.pth",
        weights_only=True
    ))
    model.eval()  # Set the model to evaluation mode

    # Initialize metrics for evaluation
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    test_predictions = []
    test_probabilities = []
    test_targets = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)  # Move images to the specified device
            labels = labels.to(device).float()  # Move labels to the specified device

            # Forward pass
            outputs = model(images)  # Raw logits from the model
            loss = criterion(outputs, labels)  # Calculate the loss
            test_loss += loss.item()  # Accumulate the total loss

            # Calculate probabilities and predictions
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
            predictions = (probabilities >= 0.5).float()  # Convert probabilities to binary predictions

            # Update metrics
            correct_test += (predictions == labels).sum().item()  # Count correct predictions
            total_test += labels.size(0)  # Total number of test samples
            test_probabilities.extend(probabilities.cpu().numpy())  # Store probabilities
            test_predictions.extend(predictions.cpu().numpy())  # Store predictions
            test_targets.extend(labels.cpu().numpy())  # Store true labels

    # Compute overall metrics
    test_loss = test_loss / len(test_loader)  # Average test loss
    test_accuracy = correct_test / total_test  # Accuracy
    test_f1 = f1_score(test_targets, test_predictions, average='macro')  # F1 Score
    test_auc = roc_auc_score(test_targets, test_probabilities)  # AUC-ROC Score
    test_conf_matrix = confusion_matrix(test_targets, test_predictions)  # Confusion matrix

    # Calculate Precision and Recall
    s, t = test_conf_matrix
    a, b = s
    c, d = t
    precision = (a / (a + c))
    recall = (a / (a + b))

    # Create a DataFrame to display the metrics
    metrics = {
        "Metric": ["Loss", "Accuracy", "F1 Score", "AUC", "Precision", "Recall"],
        "Value": [f"{test_loss:.4f}", f"{test_accuracy:.4f}", f"{test_f1:.4f}", f"{test_auc:.4f}", f"{precision:.4f}", f"{recall:.4f}"]
    }
    metrics_df = pd.DataFrame(metrics, index=[1, 2, 3, 4, 5, 6])

    # Print the metrics DataFrame
    print("Test Split Performance")
    print(metrics_df)

    print("\nPlease close the plot to continue")

    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(4, 4))
    sns.heatmap(test_conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False)

    # Add labels and title to the heatmap
    plt.title('Confusion Matrix', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"], fontsize=10)
    plt.yticks(ticks=[0.5, 1.5], labels=["Malignant", "Benign"], fontsize=10, rotation=90)

    plt.show()

# Prepare the data for testing
_, _, test_loader = dp.prep_dataA()

# Run the evaluation process
def run():
    evaluate_model(test_loader)
