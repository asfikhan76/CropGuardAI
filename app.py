"""
CropGuard AI - Streamlit App
==============================
Run with:  streamlit run app.py
"""

import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# ─────────────────────────────────────────────────────────────
# Disease treatment database
# ─────────────────────────────────────────────────────────────
TREATMENTS = {

    "Apple___Apple_scab": {
        "description": "Apple Scab is a fungal disease that affects apple leaves and fruits.",
        "symptoms": [
            "Olive-green spots on leaves",
            "Dark lesions on fruits",
            "Premature leaf drop"
        ],
        "treatment": [
            "Apply fungicides such as captan or myclobutanil",
            "Remove infected leaves",
            "Prune infected branches"
        ],
        "prevention": [
            "Maintain orchard cleanliness",
            "Improve air circulation",
            "Use disease-resistant varieties"
        ]
    },

    "Apple___Black_rot": {
        "description": "Black Rot is a fungal disease affecting apples.",
        "symptoms": [
            "Dark circular lesions",
            "Fruit rot",
            "Leaf spots"
        ],
        "treatment": [
            "Remove infected fruits",
            "Apply fungicides",
            "Prune infected branches"
        ],
        "prevention": [
            "Remove dead plant material",
            "Keep orchard clean"
        ]
    },

    "Tomato___Early_blight": {
        "description": "Early Blight is a common fungal disease in tomatoes.",
        "symptoms": [
            "Brown spots with concentric rings",
            "Yellowing leaves",
            "Leaf drop"
        ],
        "treatment": [
            "Remove infected leaves",
            "Apply copper fungicide",
            "Improve airflow"
        ],
        "prevention": [
            "Avoid overhead watering",
            "Rotate crops",
            "Maintain plant spacing"
        ]
    }
}


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropGuard AI",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 CropGuard AI")

st.markdown(
    """
AI-powered plant disease detection system using deep learning.

Upload a leaf image and the AI will:
- Detect the disease
- Show prediction confidence
- Suggest treatment recommendations
"""
)

st.divider()


# ─────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_labels():

    model = tf.keras.models.load_model(
        "crop_disease_model.h5"
    )

    with open("class_indices.json") as f:
        class_indices = json.load(f)

    return model, class_indices


model, class_indices = load_model_and_labels()


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.header("About")

    st.info(
        """
Model: MobileNetV2

Dataset: PlantVillage Dataset

Classes: 38 Plant Diseases

Accuracy: 96.77%
"""
    )

    st.header("How to Use")

    st.markdown(
        """
1. Upload a clear leaf image
2. Wait for AI analysis
3. View disease prediction
4. Read treatment recommendation
"""
    )


# ─────────────────────────────────────────────────────────────
# Upload image
# ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    # Load image
    pil_image = Image.open(uploaded_file).convert("RGB")

    # Resize
    resized_image = pil_image.resize((224, 224))

    # Preprocess
    img_array = np.array(resized_image) / 255.0

    img_batch = np.expand_dims(
        img_array,
        axis=0
    ).astype(np.float32)

    # Prediction
    with st.spinner("Analyzing leaf..."):

        predictions = model.predict(img_batch)[0]

    # Get top prediction
    top_idx = int(np.argmax(predictions))

    confidence = float(
        predictions[top_idx]
    ) * 100

    disease_name = class_indices[str(top_idx)]

    display_name = disease_name.replace(
        "___",
        " — "
    ).replace("_", " ")

    # Top 3 predictions
    top3_idx = np.argsort(predictions)[::-1][:3]

    top3 = [
        (
            class_indices[str(i)].replace(
                "___",
                " — "
            ).replace("_", " "),
            float(predictions[i]) * 100
        )
        for i in top3_idx
    ]

    # Layout
    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            pil_image,
            use_container_width=True
        )

    with col2:

        st.subheader("Prediction Result")

        st.success(
            f"Detected Disease: {display_name}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

    # Top predictions
    st.divider()

    st.subheader("Top 3 Predictions")

    for name, prob in top3:

        st.progress(
            int(prob),
            text=f"{name} — {prob:.2f}%"
        )

    # Treatment recommendation
    st.divider()

    st.subheader("Treatment Recommendation")

    treatment = TREATMENTS.get(
        disease_name,
        "Consult a plant disease expert for treatment."
    )

    st.info(treatment)

    # AI explanation
    st.divider()

    with st.expander("About This AI Model"):

        st.markdown(
            """
This project uses:

- Deep Learning
- Convolutional Neural Networks (CNN)
- Transfer Learning (MobileNetV2)
- PlantVillage Dataset

The model was trained to classify 38 different plant disease categories.
"""
        )

else:

    st.info(
        "Upload a leaf image to begin disease detection."
    )

    st.markdown(
        """
Supported crops include:
- Apple
- Corn
- Grape
- Potato
- Tomato
- Cherry
- Peach
- Strawberry
"""
    )
