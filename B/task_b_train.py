# Importing necessary libraries
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

# Set directory paths
TASK_B_DIR = Path(__file__).resolve().parent.parent / "B"

# Import custom modules
import data_prep as dp  # Custom data preparation module
import all_models_b as bm  # Module containing the BloodCellClassifier1 model definition

# Function to train and validate the model
def train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20):
    """
    Trains and validates the BloodCellClassifier1 model.

    Parameters:
        model (torch.nn.Module): The PyTorch model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for training data.
        val_loader (torch.utils.data.DataLoader): DataLoader for validation data.
        device (torch.device): The device to run training ('cpu', 'cuda', or 'mps').
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs to wait for validation loss improvement before early stopping.

    Returns:
        dict: Dictionary containing the best metrics (from the epoch with the lowest validation loss).
    """
    # Start timer to measure total training time
    start_time = time.time()

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss().to(device)  # Cross-entropy loss for multi-class classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5) 

    # Initialize lists to track metrics
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_auc_scores, val_auc_scores = [], []
    val_precisions, val_recalls = [], []

    # Variables to track the best validation performance and early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_metrics = {}

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_predictions, train_probabilities, train_targets = [], [], []

        # Iterate over training data
        for images, labels in train_loader:
            # Transfer data to the device (GPU or CPU)
            images, labels = images.to(device), labels.to(device).squeeze()

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backward pass (gradient computation)
            optimizer.step()  # Update model parameters

            running_loss += loss.item()  # Accumulate batch loss
            _, predicted = torch.max(outputs, 1)  # Get predicted class
            total_train += labels.size(0)  # Update total number of samples
            correct_train += (predicted == labels).sum().item()  # Count correct predictions

            # Store predictions and probabilities for metrics
            probabilities = torch.softmax(outputs, dim=1)
            train_probabilities.extend(probabilities.detach().cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        # Calculate training metrics for the epoch
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_f1 = f1_score(train_targets, train_predictions, average='weighted')
        train_auc = roc_auc_score(train_targets, train_probabilities, multi_class='ovr')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)
        train_auc_scores.append(train_auc)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_predictions, val_probabilities, val_targets = [], [], []

        with torch.no_grad():  # Disable gradient computation
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).squeeze()
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)  # Get predicted class
                total_val += labels.size(0)  # Update total number of samples
                correct_val += (predicted == labels).sum().item()  # Count correct predictions

                # Store predictions and probabilities for metrics
                probabilities = torch.softmax(outputs, dim=1)
                val_probabilities.extend(probabilities.detach().cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        # Calculate validation metrics for the epoch
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_f1 = f1_score(val_targets, val_predictions, average='macro')
        val_auc = roc_auc_score(val_targets, val_probabilities, multi_class='ovr')
        val_precision = precision_score(val_targets, val_predictions, average='macro', zero_division=0)
        val_recall = recall_score(val_targets, val_predictions, average='macro', zero_division=0)

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        val_auc_scores.append(val_auc)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        # Adjust learning rate based on validation loss
        scheduler.step(val_loss)

        # Update best metrics if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), TASK_B_DIR / "best_model_B.pth")  # Save the best model state
            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy/100,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_precision': val_precision,
                'val_recall': val_recall
            }
        else:
            epochs_no_improve += 1

        # Trigger early stopping if no improvement for specified epochs
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    # Output training summary
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Epoch Training Time: {total_time / (epoch + 1):.2f} seconds")

    print("\nPlease close the plots to continues")

    # Plot training and validation metrics
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    axs[0].plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label="Training Accuracy")
    axs[1].plot(range(1, len(val_accuracies) + 1), val_accuracies, label="Validation Accuracy")
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    return best_metrics


# Load the training and validation datasets
train_loader, val_loader, _ = dp.prep_dataB()  # Custom function to load data

# Set the device based on availability (GPU, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Initialize the model
model = bm.BloodCellClassifier1().to(device)

# Function to run the training and validation process
def run():
    """
    Executes the training and validation process.
    Outputs the best metrics in a readable format.
    """
    best_metrics = train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20)
    best_metrics_df = pd.DataFrame([best_metrics])
    transposed_best_metrics_df = best_metrics_df.transpose()

    # Display the best metrics
    print("\nBest Metrics DataFrame:\n")
    print(transposed_best_metrics_df)
