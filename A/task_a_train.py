import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

# Set directory paths
TASK_A_DIR = Path(__file__).resolve().parent.parent / "A"

# Import custom modules
import data_prep as dp # Custom data preparation module
import all_models_a as am # Custom script to import CNN Model

# Function to plot the confusion matrix
def plot_confusion_matrix(conf_matrix):
    """
    Plots a confusion matrix as a heatmap.

    Parameters:
    - conf_matrix: Confusion matrix to plot.

    Returns:
    - None.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malignant', 'Benign'], 
                yticklabels=['Malignant', 'Benign'])
    plt.title("Best Model Iteration Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

# Function to train and validate the model
def train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20):
    """
    Trains and evaluates a PyTorch model, plots training/validation metrics, and implements early stopping.

    Parameters:
    - model: PyTorch model to train and evaluate.
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - device: Device to run the model ('cpu' or 'cuda').
    - epochs: Maximum number of training epochs.
    - patience: Number of epochs to wait for improvement before stopping early.

    Returns:
    - best_metrics: Dictionary containing metrics for the best epoch.
    """
    # Start timing the training process
    start_time = time.time()

    # Define the loss function, optimizer, and learning rate scheduler
    pos_weight = torch.tensor(0.4 / 0.9, device=device)  # Adjust weight for imbalanced dataset
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0005)  # Adam optimizer
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # LR scheduler

    # Initialize variables for tracking metrics
    best_metrics = {}
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    cumulative_val_conf_matrix = None

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()  # Set model to training mode
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_predictions, train_probabilities, train_targets = [], [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update parameters

            running_loss += loss.item()  # Accumulate loss
            probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
            predictions = (probabilities >= 0.5).float()  # Convert probabilities to binary predictions

            correct_train += (predictions == labels).sum().item()  # Count correct predictions
            total_train += labels.size(0)  # Count total samples
            train_probabilities.extend(probabilities.detach().cpu().numpy())  # Store probabilities
            train_predictions.extend(predictions.cpu().numpy())  # Store predictions
            train_targets.extend(labels.cpu().numpy())  # Store true labels

        # Compute training metrics
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_f1 = f1_score(train_targets, train_predictions, average='weighted')
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_predictions, val_probabilities, val_targets = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)  # Compute validation loss
                val_loss += loss.item()
                probabilities = torch.sigmoid(outputs)  # Convert logits to probabilities
                predictions = (probabilities >= 0.5).float()  # Convert probabilities to binary predictions

                correct_val += (predictions == labels).sum().item()  # Count correct predictions
                total_val += labels.size(0)  # Count total samples
                val_probabilities.extend(probabilities.cpu().numpy())  # Store probabilities
                val_predictions.extend(predictions.cpu().numpy())  # Store predictions
                val_targets.extend(labels.cpu().numpy())  # Store true labels

        # Compute validation metrics
        val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct_val / total_val
        val_f1 = f1_score(val_targets, val_predictions, average='macro')
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Compute confusion matrix for validation set
        val_conf_matrix = confusion_matrix(val_targets, val_predictions)
        if cumulative_val_conf_matrix is None:
            cumulative_val_conf_matrix = val_conf_matrix
        else:
            cumulative_val_conf_matrix += val_conf_matrix

        # Early stopping: check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), TASK_A_DIR / "best_model_A.pth")  # Save best model

            # Update best metrics dictionary
            s, t = val_conf_matrix
            a, b = s
            c, d = t
            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy/100,
                'val_f1': val_f1,
                'val_precision': (a / (a + c)),
                'val_recall': (a / (a + b))
            }
        else:
            epochs_no_improve += 1

        # Stop training if no improvement for a specified number of epochs
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}.")
            break

    # Compute total training time
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Epoch Training Time: {total_time / (epoch+1):.2f} seconds")

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

    # Plot the final confusion matrix
    plot_confusion_matrix(val_conf_matrix)

    return best_metrics

# Load the training and validation datasets
train_loader, val_loader, _ = dp.prep_dataA()

# Define the device and model
device = 'cpu' #as dataset is small, cpu is faster than gpu
model = am.BreastCancerClassifier9()

# Run the training and validation process
def run():
    best_metrics = train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20)
    best_metrics_df = pd.DataFrame([best_metrics])
    transposed_best_metrics_df = best_metrics_df.transpose()
    # Display the best metrics DataFrame
    print("\nBest Metrics DataFrame:\n")
    print(transposed_best_metrics_df)
