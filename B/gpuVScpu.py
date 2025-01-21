import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import sys
sys.path.append('/Users/shivanshtibrewala/Desktop/Year_4/ELEC0134/CW')
sys.path.append('/Users/shivanshtibrewala/Desktop/Year_4/ELEC0134/CW/Task_A')
import data_prep as dp
import all_models_b as bm


def train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20):
    """
    Trains and evaluates a PyTorch model for Task B with metrics specific to multi-class classification.

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
    # Initialize variables for timing
    start_time = time.time()

    # Loss criterion, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Metrics tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_auc_scores, val_auc_scores = [], []
    val_precisions, val_recalls = [], []
    cumulative_val_conf_matrix = None

    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_metrics = {}

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0
        train_predictions, train_probabilities, train_targets = [], [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).squeeze()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            probabilities = torch.softmax(outputs, dim=1)
            train_probabilities.extend(probabilities.detach().cpu().numpy())
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        train_f1 = f1_score(train_targets, train_predictions, average='weighted')
        train_auc = roc_auc_score(train_targets, train_probabilities, multi_class='ovr')

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1_scores.append(train_f1)
        train_auc_scores.append(train_auc)

        # Validation phase
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        val_predictions, val_probabilities, val_targets = [], [], []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).squeeze()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                probabilities = torch.softmax(outputs, dim=1)
                val_probabilities.extend(probabilities.detach().cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

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

        # Update scheduler
        scheduler.step(val_loss)

        # Confusion matrix
        val_conf_matrix = confusion_matrix(val_targets, val_predictions)
        if cumulative_val_conf_matrix is None:
            cumulative_val_conf_matrix = val_conf_matrix
        else:
            cumulative_val_conf_matrix += val_conf_matrix

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model_B.pth")

            best_metrics = {
                'best_epoch': epoch + 1,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'val_f1': val_f1,
                'val_auc': val_auc,
                'val_precision': val_precision,
                'val_recall': val_recall
            }

        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}.")
            break

    # Training summary
    total_time = time.time() - start_time
    print(f"Total Training Time: {total_time:.2f} seconds")
    print(f"Epoch Training Time: {total_time / (epoch + 1):.2f} seconds")

    # Plot Training and Validation Metrics
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


train_loader, val_loader,_ = dp.prep_dataB()
devices=['mps']
model = bm.BloodCellClassifier1().to(device)

def run():
    best_metrics = train_and_val(model, train_loader, val_loader, device, epochs=500, patience=20)
    best_metrics_df = pd.DataFrame([best_metrics], index=[1])
    # Display the DataFrame
    print("\nBest Metrics DataFrame:\n")
    print(best_metrics_df)
