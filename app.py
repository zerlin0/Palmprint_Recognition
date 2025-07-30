import os
import random
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from torchvision import models, transforms

# ==== Page Config ====
st.set_page_config(page_title="Palmprint Model Comparison", layout="wide")
st.title("Palmprint Recognition Model Comparison")

# ==== Initial State ====
if "image_source" not in st.session_state:
    st.session_state.image_source = None
if "image_name" not in st.session_state:
    st.session_state.image_name = None
if "log_records" not in st.session_state:
    st.session_state.log_records = []

# ==== Helper Functions ====
def predict(image, model, class_names, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy().flatten()
    top3_idx = probs.argsort()[-3:][::-1]
    return {
        "predicted": class_names[top3_idx[0]],
        "confidence": probs[top3_idx[0]],
        "top3": [(class_names[i], float(probs[i])) for i in top3_idx]
    }

def load_all_models(selected_folders, base_path):
    models_dict = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for folder in selected_folders:
        folder_path = os.path.join(base_path, folder)
        model_path = os.path.join(folder_path, "model_best.pt")
        class_json = os.path.join(folder_path, "classes.json")
        if os.path.exists(model_path) and os.path.exists(class_json):
            with open(class_json, "r") as f:
                class_names = json.load(f)
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, len(class_names))
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            models_dict[folder] = (model, class_names)
    return models_dict, device

# ==== Sidebar Config ====
st.sidebar.header("Configuration")

#model_root = st.sidebar.text_input("📁 Model Root Folder", value=r"C:\Users\zwoll\PycharmProjects\ELWIN_all_mode\training_result\training01_20250719_0302")
model_root = st.sidebar.text_input("📁 Model Root Folder", value=r"C:\\Users\\zwoll\\PycharmProjects\\ELWIN_project_Resnet50")
test_folder = st.sidebar.text_input("📂 Test Image Folder", value=r"C:\Users\zwoll\PycharmProjects\yolo_crop_try\dataset_augment_split\test")

model_folders = []
if os.path.isdir(model_root):
    for folder in os.listdir(model_root):
        full = os.path.join(model_root, folder)
        if os.path.isdir(full) and all(os.path.exists(os.path.join(full, f)) for f in ["model_best.pt", "classes.json"]):
            model_folders.append(folder)
else:
    st.sidebar.warning("Model root not found.")

selected_models = st.sidebar.multiselect("Select Models to Compare", model_folders, default=model_folders[:2])

if not selected_models:
    st.sidebar.warning("Please select at least one model to compare.")
    st.stop()

# ==== Load Models ====
all_models, device = load_all_models(selected_models, model_root)

# ==== Upload or Random Image ====
st.subheader("📸 Input Image")
col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp"])
    if st.button("Use Random Image"):
        valid_ext = (".jpg", ".jpeg", ".png", ".bmp")
        candidates = [
            os.path.join(dp, f)
            for dp, _, files in os.walk(test_folder)
            for f in files if f.lower().endswith(valid_ext)
        ]
        if candidates:
            path = random.choice(candidates)
            img = Image.open(path).convert("RGB")
            st.session_state.image_source = img
            st.session_state.image_name = os.path.basename(path)
            st.success(f"Random image selected: {st.session_state.image_name}")
        else:
            st.warning("No test images found.")
    elif uploaded_file:
        st.session_state.image_source = Image.open(uploaded_file).convert("RGB")
        st.session_state.image_name = uploaded_file.name

with col2:
    if st.session_state.image_source:
        st.image(
            st.session_state.image_source,
            caption=f"Test Image: {st.session_state.image_name}",
            width=300,
        )

# ==== Prediction Outputs ====
if st.session_state.image_source:
    image = st.session_state.image_source
    image_name = st.session_state.image_name
    col_models = st.columns(len(all_models))

    for idx, (folder_name, (model, class_names)) in enumerate(all_models.items()):
        with col_models[idx]:
            actual = next((c for c in class_names if c in image_name.lower()), "unknown")
            result = predict(image, model, class_names, device)
            predicted = result["predicted"]
            confidence = result["confidence"]
            top3 = result["top3"]
            correct = actual.lower() in predicted.lower()

            st.markdown(f"### MODEL : {folder_name}")
            st.markdown(f"**Predicted:** `{predicted}` ({confidence:.2%})")
            st.markdown(f"**Actual:** `{actual}`")
            st.markdown(f"**Match:** {'✅' if correct else '❌'}")
            st.markdown("**Top-3 Predictions:**")
            for lbl, conf in top3:
                st.markdown(f"- `{lbl}`: {conf:.2%}")

            st.session_state.log_records.append({
                "Model": folder_name,
                "Image": image_name,
                "Actual": actual,
                "Predicted": predicted,
                "Confidence": confidence,
                "Correct": "TRUE" if correct else "FALSE"
            })

# ==== Logs ====
st.subheader("Prediction Logs")
log_df = pd.DataFrame(st.session_state.log_records)

col_a, col_b = st.columns([1, 1])
with col_a:
    if st.button("Clear Logs"):
        st.session_state.log_records = []
        st.session_state.image_source = None
        st.session_state.image_name = None
        st.rerun()

with col_b:
    if not log_df.empty:
        csv = log_df.to_csv(index=False).encode('utf-8')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button("Download CSV", data=csv, file_name=f"comparison_logs_{timestamp}.csv", mime='text/csv')

# ==== Summary ====
if not log_df.empty:
    st.subheader("Summary")
    correct = (log_df["Correct"] == "TRUE").sum()
    total = len(log_df)
    accuracy = correct / total * 100

    # --- Dynamic group index ---
    group_size = len(selected_models)
    conf_df = log_df.copy().reset_index(drop=True)
    conf_df["Index"] = (conf_df.index // group_size) + 1

    log_styled = log_df.copy().reset_index(drop=True)
    log_styled.insert(0, "No.", (log_styled.index // group_size) + 1)
    log_styled["Correct"] = log_styled["Correct"].map(
        lambda x: f"<span style='color:green;'>TRUE</span>" if x == "TRUE" else f"<span style='color:red;'>FALSE</span>"
    )

    # === Charts ===
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ✅ Accuracy")
        fig, ax = plt.subplots(figsize=(3, 3))
        counts = log_df["Correct"].value_counts().rename({"TRUE": "True", "FALSE": "False"})
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig, clear_figure=True)
        st.markdown(f"**Total:** {total}")
        st.markdown(f"**Correct:** {correct}")
        st.markdown(f"**Accuracy Overall:** {accuracy:.2f}%")

    with col2:
        st.markdown("#### Confidence by Model")
        fig2, ax2 = plt.subplots(figsize=(4, 3))
        for model_name in conf_df["Model"].unique():
            model_data = conf_df[conf_df["Model"] == model_name]
            ax2.plot(model_data["Index"], model_data["Confidence"] * 100, marker='o', label=model_name)
        ax2.set_xlabel("Index")
        ax2.set_ylabel("Confidence (%)")
        ax2.set_title("Confidence Trend")
        ax2.legend(fontsize=8)
        ax2.grid(True)
        st.pyplot(fig2, clear_figure=True)

    html_table = f"""
    <div style='max-height:300px; overflow-y:auto; width:100%; border:1px solid #ccc;
                padding:10px; border-radius:6px; font-size: 13px'>
    {log_styled.to_html(escape=False, index=False)}
    </div>
    """

    st.markdown(html_table, unsafe_allow_html=True)
