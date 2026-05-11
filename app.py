"""
Streamlit app — Breast Cancer Cell MoA Classification
Loads best_resnet50_transfer_model.keras and classifies uploaded fluorescence
microscopy images into one of 5 MoA categories with Grad-CAM explainability.
"""

import io
import os

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Cancer Cell MoA Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* Main background */
.main { background-color: #0e1117; }

/* Card-style containers */
.result-card {
    background: linear-gradient(135deg, #1e2130 0%, #262b3d 100%);
    border: 1px solid #3d4466;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Prediction label */
.pred-label {
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}

/* Confidence badge */
.conf-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    margin-top: 0.5rem;
}

/* Section headers */
.section-header {
    color: #a0aec0;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

/* Probability bar labels */
.prob-class {
    font-size: 0.85rem;
    color: #cbd5e0;
}
.prob-value {
    font-size: 0.85rem;
    color: #a0aec0;
    float: right;
}

/* Warning box */
.warn-box {
    background: #2d2010;
    border-left: 4px solid #f6ad55;
    border-radius: 4px;
    padding: 1rem;
    color: #fbd38d;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background-color: #141620;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_PATH = "outputs/models/best_resnet50_transfer_model.keras"
IMG_SIZE = 128
LAST_CONV_LAYER = "conv5_block3_out"

# Alphabetical order — matches the LabelEncoder used during training
CLASS_NAMES = [
    "Aurora kinase inhibitors",
    "DNA damage agents",
    "Eg5 inhibitors",
    "Microtubule destabilizers",
    "Microtubule stabilizers",
]

CLASS_INFO = {
    "Aurora kinase inhibitors": {
        "icon": "⚡",
        "desc": "Inhibit Aurora A/B kinases, disrupting mitotic spindle assembly and causing abnormal cell division.",
        "colour": "#fc8181",
    },
    "DNA damage agents": {
        "icon": "🧬",
        "desc": "Induce DNA strand breaks and trigger cell cycle arrest via checkpoint activation.",
        "colour": "#f6ad55",
    },
    "Eg5 inhibitors": {
        "icon": "🌀",
        "desc": "Block Eg5 kinesin motor proteins, producing monopolar mitotic spindles.",
        "colour": "#68d391",
    },
    "Microtubule destabilizers": {
        "icon": "💥",
        "desc": "Depolymerise the microtubule network, collapsing the cell cytoskeleton.",
        "colour": "#76e4f7",
    },
    "Microtubule stabilizers": {
        "icon": "🔒",
        "desc": "Stabilise and bundle microtubules, preventing normal depolymerisation.",
        "colour": "#b794f4",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading ResNet50 model…")
def load_resnet_model():
    return load_model(MODEL_PATH)


def model_available() -> bool:
    return os.path.exists(MODEL_PATH)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_raw_array(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image to a uint8 (0–255) HxWx3 numpy array.

    BBBC021 TIFFs are 16-bit. PIL preserves the raw bit depth, so we scale
    16-bit values down to 8-bit the same way cv2.IMREAD_GRAYSCALE does
    (right-shift by 8 / divide by 256). This ensures the model receives the
    same value range it was trained on.
    """
    arr = np.array(pil_image)

    # Collapse any extra dimensions (e.g. RGBA → RGB)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    # Scale 16-bit → 8-bit (matches cv2.IMREAD_GRAYSCALE behaviour used in training)
    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def percentile_normalise(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """
    Stretch each channel independently to [0, 1] using percentile clipping.

    Fluorescence images have very sparse intensity distributions — most pixels
    are near-zero background with a small number of bright structures. A plain
    /255 normalisation leaves the image invisible. Percentile stretching maps
    the 1st–99th percentile range to the full [0, 1] display range, making
    cellular structures visible regardless of the raw bit depth.
    """
    result = np.zeros(arr.shape, dtype=np.float32)
    for c in range(arr.shape[2]):
        channel = arr[:, :, c].astype(np.float32)
        p_low, p_high = np.percentile(channel, [low, high])
        if p_high > p_low:
            result[:, :, c] = np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)
        # else: channel stays all-zero (uniform channel — nothing to show)
    return result


def preprocess_for_display(raw_arr: np.ndarray) -> np.ndarray:
    """
    Resize to 128×128 and apply per-channel percentile normalisation.
    Returns float32 in [0, 1] suitable for matplotlib / st.image.
    """
    resized = cv2.resize(raw_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return percentile_normalise(resized)


def preprocess_for_model(raw_arr: np.ndarray) -> np.ndarray:
    """
    Reproduce the exact training pipeline:
      resize → float32 in [0, 255] → resnet50.preprocess_input → add batch dim.

    No percentile normalisation here — the model was trained on plain
    uint8 pixel values passed through preprocess_input, not contrast-stretched
    images.
    """
    resized = cv2.resize(raw_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)          # still in [0, 255]
    img_resnet = preprocess_input(img_float)         # ImageNet mean/std shift
    return np.expand_dims(img_resnet, axis=0)

# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def make_gradcam_heatmap(
    img_array: np.ndarray,
    model,
    last_conv_layer_name: str,
    pred_index: int | None = None,
) -> np.ndarray:
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        outputs = grad_model(img_array)
        conv_outputs = outputs[0]
        raw_preds = outputs[1]

        # Depending on the TF/Keras version and how the model was saved,
        # raw_preds can arrive as any of:
        #   (1, 5)  – standard batch-first tensor
        #   (5, 1)  – per-class (1,) tensors stacked by tf.convert_to_tensor
        #   (5,)    – flat tensor
        #   list    – Python list of scalars or tensors
        # tf.reshape(…, [-1]) collapses all of these to a guaranteed (5,)
        # 1-D tensor so tf.gather can index it safely.
        if not isinstance(raw_preds, tf.Tensor):
            raw_preds = tf.convert_to_tensor(raw_preds)
        preds_flat = tf.reshape(raw_preds, [-1])   # → (num_classes,)

        if pred_index is None:
            pred_index = int(tf.argmax(preds_flat))

        # tf.gather on a 1-D tensor is unambiguous and works regardless of
        # how predictions arrived; [:, idx] slice indexing is not.
        class_channel = tf.gather(preds_flat, pred_index)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
    return heatmap.numpy()


def overlay_gradcam(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet_colors = plt.colormaps["jet"](np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = np.array(
        Image.fromarray(np.uint8(jet_heatmap * 255)).resize(
            (original_img.shape[1], original_img.shape[0])
        )
    ) / 255.0
    superimposed = jet_heatmap * alpha + original_img
    return np.clip(superimposed, 0, 1)


def build_gradcam_figure(
    original_img: np.ndarray,
    heatmap: np.ndarray,
    overlay_img: np.ndarray,
    pred_class: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0e1117")

    panels = [
        (original_img, "Original Image", None),
        (heatmap,      "Grad-CAM Heatmap", "jet"),
        (overlay_img,  "Heatmap Overlay", None),
    ]
    for ax, (data, title, cmap) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color="#cbd5e0", fontsize=11, pad=8)
        ax.axis("off")
        ax.set_facecolor("#0e1117")

    plt.suptitle(
        f"Grad-CAM — {pred_class}",
        color="#ffffff",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 MoA Classifier")
    st.markdown("**Breast Cancer Cell Analysis**")
    st.divider()

    st.markdown("### About")
    st.markdown(
        "This tool classifies the **mechanism of action** (MoA) of chemical "
        "compounds from fluorescence microscopy images of MCF-7 breast cancer "
        "cells using a fine-tuned **ResNet50** model."
    )
    st.markdown(
        "**Accuracy:** 87.34%  \n"
        "**Macro F1:** 0.8802  \n"
        "**Dataset:** BBBC021 (788 samples)  \n"
        "**Input:** 128 × 128 px, 3-channel"
    )
    st.divider()

    st.markdown("### MoA Classes")
    for name, info in CLASS_INFO.items():
        with st.expander(f"{info['icon']} {name}"):
            st.markdown(info["desc"])

    st.divider()
    st.markdown("### Image Channels")
    st.markdown(
        "| Ch | Stain | Structure |\n"
        "|---|---|---|\n"
        "| R | DAPI | Nucleus |\n"
        "| G | Tubulin | Microtubules |\n"
        "| B | Actin | Cytoskeleton |"
    )
    st.divider()
    st.caption(
        "Final Year Project — BSc AI  \n"
        "Muhammad Ertaza Manzoor  \n"
        "Anglia Ruskin University, 2025–26"
    )

# ── Main page ─────────────────────────────────────────────────────────────────

st.markdown("# 🔬 Breast Cancer Cell MoA Classification")
st.markdown(
    "Upload a fluorescence microscopy image to classify its **mechanism of action** "
    "and visualise which cellular regions drove the prediction via **Grad-CAM**."
)
st.divider()

# Model availability check
if not model_available():
    st.markdown(
        '<div class="warn-box">'
        "<strong>⚠️ Model file not found</strong><br/>"
        f"Place <code>best_resnet50_transfer_model.keras</code> at:<br/>"
        f"<code>{MODEL_PATH}</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

model = load_resnet_model()

# ── Upload ────────────────────────────────────────────────────────────────────

upload_col, info_col = st.columns([1.6, 1], gap="large")

with upload_col:
    st.markdown('<p class="section-header">Upload Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a fluorescence microscopy image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed",
    )
    st.caption(
        "Upload a 3-channel composite image (DAPI / Tubulin / Actin stacked as RGB). "
        "The image will be resized to 128 × 128 px automatically."
    )

with info_col:
    st.markdown('<p class="section-header">How it works</p>', unsafe_allow_html=True)
    st.markdown(
        "1. **Resize** image to 128 × 128 px  \n"
        "2. **Normalise** pixel values to [0, 1]  \n"
        "3. **ResNet50 preprocess** (ImageNet channel shift)  \n"
        "4. **Classify** into one of 5 MoA categories  \n"
        "5. **Grad-CAM** highlights the deciding cell regions"
    )

# ── Inference ─────────────────────────────────────────────────────────────────

if uploaded_file is not None:
    st.divider()

    pil_image = Image.open(uploaded_file)
    raw_arr = load_raw_array(pil_image)
    display_img = preprocess_for_display(raw_arr)   # percentile-normalised, for visualisation only
    model_input = preprocess_for_model(raw_arr)     # plain uint8 → preprocess_input, matches training

    with st.spinner("Classifying…"):
        probs = model.predict(model_input, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    with st.spinner("Generating Grad-CAM…"):
        heatmap = make_gradcam_heatmap(model_input, model, LAST_CONV_LAYER, pred_index=pred_idx)
        overlay_img = overlay_gradcam(display_img, heatmap)

    # ── Result header ─────────────────────────────────────────────────────────

    res_col, spacer = st.columns([2, 1])
    with res_col:
        class_colour = CLASS_INFO[pred_class]["colour"]
        class_icon   = CLASS_INFO[pred_class]["icon"]
        st.markdown(
            f'<div class="result-card">'
            f'<p class="section-header">Predicted MoA</p>'
            f'<p class="pred-label">{class_icon} {pred_class}</p>'
            f'<span class="conf-badge">Confidence: {confidence:.1%}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Three-panel visualisation ─────────────────────────────────────────────

    st.markdown("#### Grad-CAM Visualisation")
    fig = build_gradcam_figure(display_img, heatmap, overlay_img, pred_class)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor="#0e1117")
    st.image(buf, use_container_width=True)
    plt.close(fig)

    st.divider()

    # ── Class probabilities ───────────────────────────────────────────────────

    prob_col, desc_col = st.columns([1, 1], gap="large")

    with prob_col:
        st.markdown("#### Class Probabilities")
        sorted_indices = np.argsort(probs)[::-1]
        for i in sorted_indices:
            name  = CLASS_NAMES[i]
            prob  = float(probs[i])
            icon  = CLASS_INFO[name]["icon"]
            colour = CLASS_INFO[name]["colour"]
            is_top = i == pred_idx

            label = f"{'**' if is_top else ''}{icon} {name}{'**' if is_top else ''}"
            st.markdown(
                f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
                f'<span class="prob-class">{"🏆 " if is_top else ""}{icon} {name}</span>'
                f'<span class="prob-value">{prob:.1%}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(
                prob,
                text=None,
            )
            st.markdown("<div style='margin-bottom:6px'></div>", unsafe_allow_html=True)

    with desc_col:
        st.markdown("#### Biological Interpretation")
        st.markdown(
            f'<div class="result-card">'
            f'<p class="section-header">Predicted class</p>'
            f'<p style="font-size:1.1rem;font-weight:600;color:{class_colour}">'
            f'{class_icon} {pred_class}</p>'
            f'<p style="color:#cbd5e0;margin-top:0.5rem">{CLASS_INFO[pred_class]["desc"]}</p>'
            f"</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="result-card">'
            '<p class="section-header">Grad-CAM explanation</p>'
            '<p style="color:#cbd5e0">'
            "Warm regions (red/yellow) in the heatmap indicate where the model's "
            "attention was strongest. For this prediction, the model focused on "
            "cellular structures associated with the predicted MoA class."
            "</p></div>",
            unsafe_allow_html=True,
        )

    # ── Download button ───────────────────────────────────────────────────────

    st.divider()
    dl_col, _ = st.columns([1, 2])
    with dl_col:
        buf2 = io.BytesIO()
        fig2 = build_gradcam_figure(display_img, heatmap, overlay_img, pred_class)
        fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=200, facecolor="#0e1117")
        plt.close(fig2)
        st.download_button(
            label="⬇️ Download Grad-CAM Figure",
            data=buf2.getvalue(),
            file_name=f"gradcam_{pred_class.replace(' ', '_').lower()}.png",
            mime="image/png",
        )

else:
    # ── Placeholder when no image uploaded ────────────────────────────────────
    st.markdown(
        '<div style="text-align:center;padding:3rem;color:#4a5568;border:2px dashed #2d3748;border-radius:12px">'
        "<h3 style='color:#718096'>Upload an image to get started</h3>"
        "<p>Supported formats: PNG, JPG, TIFF</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### Example results from the test set")
    fig_col1, fig_col2 = st.columns(2)
    sample_figs = [
        ("outputs/figures/resnet50_gradcam_correct_samples.png",   "Correctly classified samples"),
        ("outputs/figures/resnet50_gradcam_misclassified_samples.png", "Misclassified samples"),
    ]
    for col, (path, caption) in zip([fig_col1, fig_col2], sample_figs):
        if os.path.exists(path):
            with col:
                st.image(path, caption=caption, use_container_width=True)
