from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import time

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Blood Smear Classification",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem;}
    .app-subtitle {color: #9ca3af; margin-top: -0.7rem;}
    .result-card {
        border: 1px solid rgba(128, 128, 128, 0.28);
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
    }
    .model-label {font-size: 0.86rem; color: #9ca3af; margin-bottom: 0.2rem;}
    .prediction-label {font-size: 1.8rem; font-weight: 700; margin-bottom: 0.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🩸 Blood Smear Classification System")
st.markdown(
    '<p class="app-subtitle">Compare ResNet-50 and Vision Transformer '
    "predictions on the same peripheral blood smear image.</p>",
    unsafe_allow_html=True,
)
st.warning(
    "Academic prototype only. The result is not a medical diagnosis and must "
    "be verified by a qualified healthcare professional."
)


#app ui config

CLASS_NAMES = ["leukemia", "normal", "sickle_cell", "thalassemia"]
DISPLAY_NAMES = {
    "leukemia": "Leukemia",
    "normal": "Normal",
    "sickle_cell": "Sickle Cell",
    "thalassemia": "Thalassemia",
}

BASE_DIR = Path(__file__).resolve().parent
RESNET_PATH = BASE_DIR / "models" / "best_resnet50.pth"
VIT_PATH = BASE_DIR / "models" / "best_vit_b16.pth"
ASSETS_DIR = BASE_DIR / "assets"

VIT_IS_PLACEHOLDER = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PERFORMANCE = {
    "ResNet-50": {
        "Test accuracy": "92%",
        "Weighted precision": "0.93",
        "Weighted recall": "0.92",
        "Weighted F1-score": "0.92",
        "Test images": "164",
    },
    "ViT-B/16": {
        "Test accuracy": "94%",
        "Weighted precision": "0.94",
        "Weighted recall": "0.94",
        "Weighted F1-score": "0.94",
        "Test images": "164",
    },
}

DISEASE_INFORMATION = {
    "leukemia": (
        "Leukemia is a group of cancers affecting blood-forming tissues. "
        "Abnormal white blood cells may be visible in a blood smear, but "
        "confirmation requires professional laboratory assessment."
    ),
    "normal": (
        "The model did not identify the visual patterns associated with the "
        "three disease classes. A normal model prediction does not rule out "
        "disease or replace a clinical blood test."
    ),
    "sickle_cell": (
        "Sickle cell disease can cause red blood cells to develop an abnormal "
        "curved or sickle-like shape. Diagnosis requires professional review "
        "and appropriate laboratory testing."
    ),
    "thalassemia": (
        "Thalassemia affects haemoglobin production and may produce changes "
        "in red blood cell size, colour and shape. A smear image alone is not "
        "sufficient for clinical diagnosis."
    ),
}

IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


#IMAGE VALIDATION
def validate_blood_smear_image(image, file_size):
    """Perform basic checks and reject obvious non-blood-smear images."""

    if file_size > 15 * 1024 * 1024:
        return False, "The uploaded image is larger than the 15 MB limit."

    if image.width < 150 or image.height < 150:
        return False, "The image must be at least 150 × 150 pixels."

    aspect_ratio = max(image.width / image.height, image.height / image.width)
    if aspect_ratio > 4:
        return False, "The image dimensions are unusually narrow or wide."

    sample = image.resize((96, 96))
    pixels = np.asarray(sample, dtype=np.float32) / 255.0

    maximum = pixels.max(axis=2)
    minimum = pixels.min(axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1e-6)

    stained_pixels = (saturation > 0.06) & (maximum < 0.98)
    stained_fraction = float(stained_pixels.mean())

    horizontal_changes = np.mean(stained_pixels[:, 1:] != stained_pixels[:, :-1])
    vertical_changes = np.mean(stained_pixels[1:, :] != stained_pixels[:-1, :])
    region_changes = float((horizontal_changes + vertical_changes) / 2)

    mean_red = float(pixels[:, :, 0].mean())
    mean_green = float(pixels[:, :, 1].mean())
    mean_blue = float(pixels[:, :, 2].mean())
    red_dominance = mean_red - max(mean_green, mean_blue)

    if red_dominance > 0.18 and stained_fraction > 0.20:
        return False, "The uploaded image appears to be a non-blood-smear object."

    if not 0.015 <= stained_fraction <= 0.85 or region_changes < 0.04:
        return False, "The uploaded image does not appear to be a full-field blood smear image."

    return True, "Image validation passed."


# =========================================================
# MODEL LOADING
# =========================================================
def extract_state_dict(checkpoint):
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]
        return checkpoint
    raise TypeError("The checkpoint format is not supported.")


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key.replace("module.", "", 1)
        if key.startswith("model."):
            key = key.replace("model.", "", 1)
        cleaned[key] = value
    return cleaned


@st.cache_resource
def load_resnet(model_path, modification_time):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(clean_state_dict(extract_state_dict(checkpoint)))
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_vit(model_path, modification_time):
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(clean_state_dict(extract_state_dict(checkpoint)))
    model.to(DEVICE)
    model.eval()
    return model


def predict_image(model, image_tensor):
    input_tensor = image_tensor.clone().to(DEVICE)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.inference_mode():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_time = time.perf_counter() - start_time
    predicted_index = int(torch.argmax(probabilities).item())
    probability_values = probabilities.detach().cpu().tolist()

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": probability_values[predicted_index] * 100,
        "inference_time": inference_time,
        "probabilities": {
            class_name: probability * 100
            for class_name, probability in zip(CLASS_NAMES, probability_values)
        },
    }


# =========================================================
# DISPLAY HELPERS
# =========================================================
def confidence_interpretation(confidence):
    if confidence >= 80:
        return "Higher model confidence", "success"
    if confidence >= 60:
        return "Moderate model confidence", "warning"
    return "Low model confidence", "error"


def show_model_card(model_name, result, status_text):
    prediction = DISPLAY_NAMES[result["predicted_class"]]

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="model-label">{model_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="prediction-label">{prediction}</div>', unsafe_allow_html=True)

    confidence_col, time_col = st.columns(2)
    confidence_col.metric("Confidence", f'{result["confidence"]:.2f}%')
    time_col.metric("Inference time", f'{result["inference_time"] * 1000:.2f} ms')

    message, message_type = confidence_interpretation(result["confidence"])
    getattr(st, message_type)(message)

    st.caption(status_text)

    top_predictions = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    top_table = pd.DataFrame(
        {
            "Class": [DISPLAY_NAMES[name] for name, _ in top_predictions],
            "Probability": [f"{value:.2f}%" for _, value in top_predictions],
        }
    )

    st.dataframe(top_table, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def performance_dataframe(model_name):
    values = MODEL_PERFORMANCE[model_name]
    return pd.DataFrame({"Metric": values.keys(), "Result": values.values()})


def show_optional_image(filename, caption):
    filename_path = Path(filename)
    candidates = [
        ASSETS_DIR / filename_path.name,
        ASSETS_DIR / filename_path.stem,
    ]

    image_path = next((path for path in candidates if path.is_file()), None)

    if image_path is not None:
        st.image(str(image_path), caption=caption, use_container_width=True)
    else:
        checked_paths = " or ".join(str(path) for path in candidates)
        st.warning(f"Performance image not found: {checked_paths}")


# =========================================================
# CHECK AND LOAD MODELS
# =========================================================
missing_models = []

if not RESNET_PATH.exists():
    missing_models.append(f"ResNet-50: {RESNET_PATH}")

if not VIT_PATH.exists():
    missing_models.append(f"ViT-B/16: {VIT_PATH}")

if missing_models:
    st.error("One or more model files are missing.")
    for missing_model in missing_models:
        st.code(missing_model)
    st.info("Place both .pth files inside the models folder, then rerun the app.")
    st.stop()

try:
    with st.spinner("Loading ResNet-50 and ViT-B/16..."):
        resnet_model = load_resnet(str(RESNET_PATH), RESNET_PATH.stat().st_mtime)
        vit_model = load_vit(str(VIT_PATH), VIT_PATH.stat().st_mtime)

except Exception as error:
    st.error("The trained models could not be loaded.")
    st.exception(error)
    st.stop()

status_col1, status_col2, status_col3 = st.columns(3)

status_col1.success("ResNet-50 loaded")

if VIT_IS_PLACEHOLDER:
    status_col2.warning("Previous ViT placeholder loaded")
else:
    status_col2.success("ViT-B/16 loaded")

status_col3.info(f"Device: {str(DEVICE).upper()}")


# MAIN TABS

prediction_tab, performance_tab, guide_tab = st.tabs(
    ["Image Analysis", "Model Performance", "About the System"]
)


with prediction_tab:
    st.subheader("Upload a blood smear image")

    uploaded_file = st.file_uploader(
        "Accepted formats: PNG, JPG and JPEG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.info("Upload one peripheral blood smear image to begin the comparison.")

    else:
        uploaded_bytes = uploaded_file.getvalue()
        upload_id = hashlib.sha256(uploaded_bytes).hexdigest()

        try:
            uploaded_image = Image.open(uploaded_file).convert("RGB")

        except Exception:
            st.error("The uploaded file could not be opened as an image.")
            st.stop()

        image_is_valid, validation_message = validate_blood_smear_image(
            uploaded_image,
            len(uploaded_bytes),
        )

        if not image_is_valid:
            st.error(validation_message)
            st.info(
                "Please upload a clear microscopic peripheral blood smear image. "
                "Images such as fruits, objects, screenshots, or unrelated photos "
                "will not be analysed by the models."
            )
            st.stop()

        image_col, details_col = st.columns([1, 1.25])

        with image_col:
            st.image(
                uploaded_image,
                caption=f"Uploaded image: {uploaded_file.name}",
                use_container_width=True,
            )

        with details_col:
            st.markdown("#### Image information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(
                f"**Original dimensions:** {uploaded_image.width} × "
                f"{uploaded_image.height} pixels"
            )
            st.write("**Model input:** 224 × 224 RGB")
            st.write("**Preprocessing:** Resize, tensor conversion and normalization")
            st.success(validation_message)

            if uploaded_image.width < 224 or uploaded_image.height < 224:
                st.warning(
                    "This image is smaller than the model input size. Enlarging it may reduce visible detail."
                )

        if (
            "analysis" not in st.session_state
            or st.session_state["analysis"].get("upload_id") != upload_id
        ):
            transformed_image = IMAGE_TRANSFORM(uploaded_image).unsqueeze(0)

            with st.spinner("Both models are analysing the image..."):
                with ThreadPoolExecutor(max_workers=2) as executor:
                    resnet_future = executor.submit(
                        predict_image,
                        resnet_model,
                        transformed_image,
                    )
                    vit_future = executor.submit(
                        predict_image,
                        vit_model,
                        transformed_image,
                    )

                    st.session_state["analysis"] = {
                        "upload_id": upload_id,
                        "resnet": resnet_future.result(),
                        "vit": vit_future.result(),
                    }

        analysis = st.session_state.get("analysis")

        if analysis and analysis.get("upload_id") == upload_id:
            resnet_result = analysis["resnet"]
            vit_result = analysis["vit"]

            st.divider()
            st.subheader("Side-by-side prediction comparison")

            resnet_col, vit_col = st.columns(2)

            with resnet_col:
                show_model_card(
                    "ResNet-50",
                    resnet_result,
                    "Evaluated on image based dataset",
                )

            with vit_col:
                vit_status = (
                    "-"
                    if VIT_IS_PLACEHOLDER
                    else "Evaluated on image based dataset"
                )

                show_model_card(
                    "Vision Transformer (ViT-B/16)",
                    vit_result,
                    vit_status,
                )

            st.subheader("Agreement analysis")

            resnet_class = resnet_result["predicted_class"]
            vit_class = vit_result["predicted_class"]
            minimum_confidence = min(
                resnet_result["confidence"],
                vit_result["confidence"],
            )

            if resnet_class == vit_class and minimum_confidence >= 60:
                st.success(
                    f"The models agree: both predicted "
                    f"**{DISPLAY_NAMES[resnet_class]}**. Agreement does not confirm "
                    "a medical diagnosis."
                )

            elif resnet_class == vit_class:
                st.warning(
                    f"Both models predicted **{DISPLAY_NAMES[resnet_class]}**, but at "
                    "least one model has low confidence. The result requires caution."
                )

            else:
                st.warning(
                    f"The models disagree. ResNet-50 predicted "
                    f"**{DISPLAY_NAMES[resnet_class]}**, while ViT-B/16 predicted "
                    f"**{DISPLAY_NAMES[vit_class]}**. The image requires further "
                    "verification."
                )

            st.subheader("Class probability comparison")

            probability_chart = pd.DataFrame(
                {
                    "ResNet-50": [
                        resnet_result["probabilities"][name] for name in CLASS_NAMES
                    ],
                    "ViT-B/16": [
                        vit_result["probabilities"][name] for name in CLASS_NAMES
                    ],
                },
                index=[DISPLAY_NAMES[name] for name in CLASS_NAMES],
            )

            st.bar_chart(probability_chart, y_label="Probability (%)")

            st.caption(
                "Confidence is the model's certainty in its prediction. It is not the probability that the patient has the disease."
            )

            st.subheader("Educational information")

            if resnet_class == vit_class:
                st.info(DISEASE_INFORMATION[resnet_class])

            else:
                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.markdown(f"**{DISPLAY_NAMES[resnet_class]}**")
                    st.write(DISEASE_INFORMATION[resnet_class])

                with info_col2:
                    st.markdown(f"**{DISPLAY_NAMES[vit_class]}**")
                    st.write(DISEASE_INFORMATION[vit_class])

            with st.expander("Technical details"):
                st.write(f"**Processing device:** {DEVICE}")
                st.write(f"**ResNet checkpoint:** {RESNET_PATH.name}")
                st.write(f"**ViT checkpoint:** {VIT_PATH.name}")
                st.write("**Class order:** " + ", ".join(CLASS_NAMES))
                st.write("**Input normalization:** ImageNet mean and standard deviation")


with performance_tab:
    st.subheader("Overall test-set performance")

    st.caption(
        "These results were calculated using the complete unseen test set. They "
        "are not calculated from the uploaded image."
    )

    resnet_perf_col, vit_perf_col = st.columns(2)

    with resnet_perf_col:
        st.markdown("### ResNet-50")

        st.dataframe(
            performance_dataframe("ResNet-50"),
            hide_index=True,
            use_container_width=True,
        )

        show_optional_image("resnet_confusion_matrix.jpeg", "ResNet-50 confusion matrix")
        show_optional_image("resnet_accuracy.jpeg", "ResNet-50 training accuracy")
        show_optional_image("resnet_loss.jpeg", "ResNet-50 training loss")

    with vit_perf_col:
        st.markdown("### Vision Transformer (ViT-B/16)")

        st.dataframe(
            performance_dataframe("ViT-B/16"),
            hide_index=True,
            use_container_width=True,
        )
        show_optional_image("vit_confusion_matrix.jpeg", "ViT confusion matrix")
        show_optional_image("vit_accuracy.jpeg", "ViT training accuracy")
        show_optional_image("vit_loss.jpeg", "ViT training loss")


with guide_tab:
    st.subheader("How the system works")

    st.markdown(
        """
        1. Upload one peripheral blood smear image.
        2. The uploaded image is checked to make sure it appears to be a blood smear.
        3. The image is resized to 224 × 224 and normalized.
        4. ResNet-50 and ViT-B/16 process the same image.
        5. The system compares their predictions, confidence and inference time.
        6. The result is displayed for academic comparison and must not be treated
           as a clinical diagnosis.
        """
    )

    st.subheader("Confidence guide")

    confidence_guide = pd.DataFrame(
        {
            "Confidence": ["80% and above", "60% to 79.99%", "Below 60%"],
            "Interpretation": [
                "Higher model confidence",
                "Moderate model confidence",
                "Low model confidence",
            ],
        }
    )

    st.dataframe(confidence_guide, hide_index=True, use_container_width=True)

    st.subheader("Important limitations")

    st.markdown(
        """
        - The models were trained using a limited and imbalanced academic dataset.
        - Images from different sources may contain different staining and camera styles.
        - The image validation only rejects obvious unrelated images and is not perfect.
        - Model confidence does not measure clinical certainty.
        - Predictions require verification by a qualified medical professional.
        """
    )


st.divider()

st.caption(
    "FYP ACADEMIC PROTOTYPE | TANESHEN MAHINDRAN TP078396 | ResNet-50 and Vision Transformer comparison for blood smear image classification."
)
