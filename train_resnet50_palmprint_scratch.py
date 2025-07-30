import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

def get_next_training_folder():
    base = "training"
    existing = sorted(Path('.').glob(f"{base}[0-9][0-9]_2*"))
    next_id = 1
    if existing:
        last = existing[-1].name
        last_id = int(last[len(base):len(base)+2])
        next_id = last_id + 1
    now = datetime.now().strftime("%Y%m%d_%H%M")
    folder = f"{base}{next_id:02d}_{now}_scratch"
    os.makedirs(folder, exist_ok=True)
    return folder

def log(msg, file):
    print(msg)
    file.write(msg + "\n")

def plot_training_curves(train_loss, val_loss, train_acc, val_acc, output_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "history.png"))
    plt.close()

def plot_confusion_matrix(cm, class_names, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def simple_progress(current, total, prefix=""):
    percent = 100 * (current + 1) / total
    print(f"\r{prefix} [{current + 1}/{total}] ({percent:.1f}%)", end="", flush=True)

def create_dataset(path, transform, class_names):
    dataset = datasets.ImageFolder(path, transform=transform)
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    dataset.class_to_idx = class_to_idx
    dataset.classes = class_names
    return dataset


# Main Training Script
def main():
    DATASET_PATH = r"C:\Users\zwoll\PycharmProjects\yolo_crop_try\dataset_augment_split"
    TRAIN_DIR = os.path.join(DATASET_PATH, "train")
    VAL_DIR = os.path.join(DATASET_PATH, "val")
    BATCH_SIZE = 32
    NUM_EPOCHS = 80
    NUM_WORKERS = 2
    IMAGE_SIZE = 224
    LEARNING_RATE = 1e-4
    PATIENCE = 10
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    OUTPUT_DIR = get_next_training_folder()

    # Step 1: Prepare class names
    class_names = sorted(os.listdir(TRAIN_DIR), key=lambda x: int(x[4:]))
    num_classes = len(class_names)

    with open(os.path.join(OUTPUT_DIR, "classes.json"), "w") as f:
        json.dump(class_names, f, indent=4)

    # Step 2: Transforms
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Step 3: Dataset + Loader with fixed class order
    train_dataset = create_dataset(TRAIN_DIR, transform, class_names)
    val_dataset = create_dataset(VAL_DIR, transform, class_names)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Step 4: Compute class weights
    class_weights = compute_class_weight(class_weight='balanced',
                                         classes=np.unique(train_dataset.targets),
                                         y=train_dataset.targets)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

    # Step 5: Build model from scratch
    model = models.resnet50(weights=None)  # no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # Training Loop
    best_val_acc = 0
    epochs_no_improve = 0
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    log_file = open(os.path.join(OUTPUT_DIR, "log.txt"), "w")

    for epoch in range(NUM_EPOCHS):
        log(f"\nEpoch {epoch+1}/{NUM_EPOCHS}", log_file)

        # Train
        model.train()
        total_loss, correct = 0.0, 0
        for i, (inputs, labels) in enumerate(train_loader):
            simple_progress(i, len(train_loader), prefix="Training")
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            total_loss += loss.item() * inputs.size(0)
            correct += (preds == labels).sum().item()
        print()
        epoch_loss = total_loss / len(train_dataset)
        epoch_acc = correct / len(train_dataset)
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        log(f"Train Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%", log_file)

        # Validate
        model.eval()
        total_loss, correct = 0.0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                simple_progress(i, len(val_loader), prefix="Validation")
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                total_loss += loss.item() * inputs.size(0)
                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        print()
        epoch_loss = total_loss / len(val_dataset)
        epoch_acc = correct / len(val_dataset)
        val_loss_history.append(epoch_loss)
        val_acc_history.append(epoch_acc)
        log(f"Val   Loss: {epoch_loss:.4f} | Acc: {epoch_acc*100:.2f}%", log_file)

        # Checkpoints
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_last.pt"))

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model_best.pt"))
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step(epoch_acc)

        if epochs_no_improve >= PATIENCE:
            log("Early stopping triggered.", log_file)
            break

    # Final Evaluation
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, OUTPUT_DIR)
    plot_training_curves(train_loss_history, val_loss_history, train_acc_history, val_acc_history, OUTPUT_DIR)

    log(f"\n✅ Training complete. Best val acc: {best_val_acc:.4f}", log_file)
    log_file.close()

if __name__ == "__main__":
    main()
