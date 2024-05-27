import csv
import os

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np
from tqdm import tqdm

from models.cnn import CNN


def calculate_metrics(outputs, targets):
    outputs = (outputs > 0.5).astype(int)
    targets = targets.astype(int)

    accuracy = accuracy_score(targets, outputs)
    precision = precision_score(targets, outputs, zero_division=0)
    recall = recall_score(targets, outputs)
    f1 = f1_score(targets, outputs)
    mcc = matthews_corrcoef(targets, outputs)
    iou = np.sum((outputs & targets)) / np.sum((outputs | targets))  # Intersection over Union

    return accuracy, precision, recall, f1, mcc, iou


def evaluate_model(model, title, data_loader, criterion, device):
    model.to(device)
    model.eval()
    total_loss = 0.0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc=f"Evaluating {title}", unit='batch') as pbar:
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                targets = targets.float().view(-1, 1)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                pbar.update(1)

    avg_loss = total_loss / len(data_loader)
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    accuracy, precision, recall, f1, mcc, iou = calculate_metrics(all_outputs, all_targets)

    return avg_loss, accuracy, precision, recall, f1, mcc, iou


def update_csv(model_name, avg_loss, accuracy, precision, recall, f1, mcc, iou):
    file_path = "results/models_results.csv"
    temp_file_path = "results/temp_models_results.csv"

    # Write data to a temporary CSV file excluding the first 3 rows
    with open(file_path, 'r', newline='') as infile, open(temp_file_path, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        for index, row in enumerate(reader):
            if index >= 3:
                writer.writerow(row)

    # Replace the original CSV file with the temporary one
    os.remove(file_path)
    os.rename(temp_file_path, file_path)

    # Save new data to the CSV file
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([model_name])
        writer.writerow(['Average Loss', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'MCC', 'IoU'])
        writer.writerow([avg_loss, accuracy, precision, recall, f1, mcc, iou])
