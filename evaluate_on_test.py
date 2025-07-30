import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np


class CustomImageFolder(Dataset):
    def __init__(self, root, transform, class_names):
        self.transform = transform
        self.loader = default_loader
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

        for cls in class_names:
            cls_path = os.path.join(root, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.samples.append((os.path.join(cls_path, fname), self.class_to_idx[cls]))

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = self.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label, path

    def __len__(self):
        return len(self.samples)


def plot_confusion_matrix(cm, class_names, accuracy, output_path, report=None):
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names,
                yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix (Counts) — Accuracy: {accuracy:.2%}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if report:
        # Prepare precision and F1-score lines
        precision_lines = []
        f1_lines = []
        for cls in class_names:
            p = report[cls]["precision"]
            f1 = report[cls]["f1-score"]
            precision_lines.append(f"{cls}: {p:.2f}")
            f1_lines.append(f"{cls}: {f1:.2f}")

        # Combine into one text block
        extra_text = "Precision:\n" + "\n".join(precision_lines) + "\n\n" + \
                     "F1-score:\n" + "\n".join(f1_lines)

        # Add textbox below matrix
        fig.subplots_adjust(bottom=0.25)  # make space below
        fig.text(0.5, 0.01, extra_text, ha="center", va="bottom", fontsize=9, family="monospace")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_summary_report(report, accuracy, output_path):
    per_class_data = {
        "Class": [],
        "Precision": [],
        "Recall": [],
        "F1-score": []
    }

    for cls, metrics in report.items():
        if cls not in ["accuracy", "macro avg", "weighted avg"]:
            per_class_data["Class"].append(cls)
            per_class_data["Precision"].append(metrics["precision"])
            per_class_data["Recall"].append(metrics["recall"])
            per_class_data["F1-score"].append(metrics["f1-score"])

    df = pd.DataFrame(per_class_data)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df.melt(id_vars="Class", var_name="Metric", value_name="Score"),
                x="Class", y="Score", hue="Metric")
    plt.ylim(0, 1.05)
    plt.title(f"Classification Report Summary (Accuracy: {accuracy:.2%})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    DATASET_PATH = r"C:\Users\zwoll\PycharmProjects\yolo_crop_try\dataset_augment_split"
    #MODEL_DIR = r"C:\Users\zwoll\PycharmProjects\ELWIN_project_Resnet50\training11_20250710_0312_finetune"
    MODEL_DIR = r"C:\Users\zwoll\PycharmProjects\ELWIN_project_Resnet50\training12_20250710_0410_scratch"
    TEST_DIR = os.path.join(DATASET_PATH, "test")
    OUTPUT_DIR = MODEL_DIR
    MODEL_PATH = os.path.join(MODEL_DIR, "model_best.pt")
    CLASSES_PATH = os.path.join(MODEL_DIR, "classes.json")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found: {MODEL_PATH}")
    if not os.path.exists(CLASSES_PATH):
        raise FileNotFoundError(f"❌ classes.json not found: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_dataset = CustomImageFolder(TEST_DIR, transform, class_names)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds, all_labels, all_paths = [], [], []
    print("🔍 Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels, paths in tqdm(test_loader, desc="Testing", dynamic_ncols=True):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_paths.extend(paths)

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = report["accuracy"]
    cm = confusion_matrix(all_labels, all_preds)

    # Save JSON report
    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Save CSV
    df = pd.DataFrame({
        "filename": [os.path.basename(p) for p in all_paths],
        "true_label": [class_names[i] for i in all_labels],
        "predicted_label": [class_names[i] for i in all_preds],
        "correct": [t == p for t, p in zip(all_labels, all_preds)]
    })
    df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Save log
    with open(os.path.join(OUTPUT_DIR, "test_log.txt"), "w") as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Per-class recall:\n")
        for cls in class_names:
            f.write(f"  {cls:20s}: {report[cls]['recall']:.4f}\n")

    # Terminal summary
    print(f"\n✅ Test Accuracy: {accuracy:.4f}")
    print("📊 Per-class recall:")
    for cls in class_names:
        print(f"  {cls:20s}: {report[cls]['recall']:.4f}")

    # Save confusion matrix
    plot_confusion_matrix(cm, class_names, accuracy, os.path.join(OUTPUT_DIR, "test_confusion_matrix.png"))

    # Save visual summary
    plot_summary_report(report, accuracy, os.path.join(OUTPUT_DIR, "test_summary_report.png"))

    print(f"\n📁 Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
