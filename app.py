"""
Streamlit app — Breast Cancer Cell MoA Classification
Loads best_resnet50_transfer_model.keras and classifies uploaded fluorescence
microscopy images into one of 5 MoA categories with Grad-CAM explainability.
"""

import io
import os

import cv2
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

/* ── Global ── */
html, body, [class*="css"], .stMarkdown, p, li, span {
    font-family: 'Inter', sans-serif !important;
}
.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1200px;
}
.main { background-color: #f0f2f9; }
h1, h2, h3, h4 { font-family: 'Inter', sans-serif !important; letter-spacing: -0.02em; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 40%, #a855f7 70%, #3b82f6 100%);
    border-radius: 24px;
    padding: 3rem 3rem 2.5rem;
    margin: 1.5rem 0 2rem;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 60px rgba(99,102,241,0.35);
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: rgba(255,255,255,0.06);
    border-radius: 50%;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -80px; left: -40px;
    width: 240px; height: 240px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 900;
    margin: 0 0 0.5rem;
    line-height: 1.15;
    letter-spacing: -0.03em;
}
.hero-sub {
    font-size: 1.1rem;
    opacity: 0.88;
    margin-bottom: 1.5rem;
    font-weight: 400;
    max-width: 640px;
}
.chip {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    margin: 0.2rem 0.3rem 0.2rem 0;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.01em;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 18px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 2px 16px rgba(60,70,120,0.07);
    border: 1px solid #e8ecf6;
}
.card-title {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8892b0;
    margin-bottom: 0.6rem;
}

/* ── Prediction card ── */
.pred-card {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 60%, #3b82f6 100%);
    border-radius: 22px;
    padding: 2rem 2.25rem;
    color: white;
    box-shadow: 0 12px 40px rgba(99,102,241,0.45);
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.pred-card::after {
    content: "";
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: rgba(255,255,255,0.07);
    border-radius: 50%;
}
.pred-card-label {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    opacity: 0.8;
    margin-bottom: 0.4rem;
}
.pred-class {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.15;
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}
.pred-sub {
    font-size: 1rem;
    opacity: 0.82;
    margin-bottom: 1rem;
    font-weight: 400;
}
.conf-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.conf-track {
    background: rgba(255,255,255,0.25);
    border-radius: 50px;
    height: 10px;
    overflow: hidden;
}
.conf-fill {
    background: white;
    height: 100%;
    border-radius: 50px;
}

/* ── Prob bars ── */
.prob-row { margin-bottom: 1rem; }
.prob-label-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.prob-name {
    font-size: 0.92rem;
    font-weight: 600;
    color: #2d3561;
}
.prob-pct {
    font-size: 0.92rem;
    font-weight: 700;
    color: #4f5a8a;
}
.prob-track {
    background: #edf0fb;
    border-radius: 50px;
    height: 9px;
    overflow: hidden;
}
.prob-fill {
    height: 100%;
    border-radius: 50px;
}

/* ── Section heading ── */
.section-heading {
    font-size: 1.25rem;
    font-weight: 800;
    color: #1a1f3c;
    margin: 1.5rem 0 1rem;
    letter-spacing: -0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Gradcam panel ── */
.gradcam-wrap {
    background: white;
    border-radius: 18px;
    padding: 1.5rem;
    box-shadow: 0 2px 16px rgba(60,70,120,0.07);
    border: 1px solid #e8ecf6;
    margin-bottom: 1.25rem;
}
.gradcam-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #1a1f3c;
    margin-bottom: 0.2rem;
    letter-spacing: -0.01em;
}
.gradcam-sub {
    font-size: 0.87rem;
    color: #7a85b0;
    margin-bottom: 1rem;
}

/* ── Bio card ── */
.bio-card {
    background: linear-gradient(135deg, #f5f7ff 0%, #ede9ff 100%);
    border: 1px solid #d8d5f7;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.bio-class-name {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.bio-desc {
    font-size: 0.9rem;
    color: #4a5280;
    line-height: 1.6;
}

/* ── Empty state ── */
.empty-state {
    background: white;
    border-radius: 20px;
    padding: 4rem 2rem;
    text-align: center;
    border: 2px dashed #d0d7f0;
    box-shadow: 0 2px 16px rgba(60,70,120,0.05);
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2d3561;
    margin-bottom: 0.5rem;
}
.empty-sub { font-size: 1rem; color: #8892b0; }

/* ── Warning ── */
.warn-box {
    background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #78350f;
    font-weight: 500;
}

/* ── Footer ── */
.footer {
    background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 50%, #1e2d5a 100%);
    border-radius: 20px;
    padding: 2.25rem 2rem;
    text-align: center;
    color: white;
    margin-top: 3rem;
    box-shadow: 0 8px 32px rgba(30,40,90,0.2);
}
.footer-name {
    font-size: 1.25rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    margin-bottom: 0.3rem;
}
.footer-uni {
    font-size: 0.9rem;
    opacity: 0.7;
    margin-bottom: 1.2rem;
}
.footer-links { display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; }
.footer-link {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 50px;
    padding: 0.45rem 1.2rem;
    color: white !important;
    text-decoration: none !important;
    font-size: 0.88rem;
    font-weight: 600;
    transition: background 0.2s;
}
.footer-link:hover { background: rgba(255,255,255,0.22); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9ff 0%, #f0f3ff 100%);
    border-right: 1px solid #dde2f5;
}
.sidebar-profile {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border-radius: 16px;
    padding: 1.25rem 1.4rem;
    color: white;
    margin-bottom: 1.25rem;
}
.sidebar-name {
    font-size: 1rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}
.sidebar-uni {
    font-size: 0.78rem;
    opacity: 0.82;
    line-height: 1.4;
}
.sidebar-stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 6px rgba(60,70,120,0.06);
    border: 1px solid #eaeef8;
}
.sidebar-stat-label { font-size: 0.8rem; color: #7a85b0; font-weight: 500; }
.sidebar-stat-value { font-size: 0.88rem; font-weight: 700; color: #2d3561; }
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
        "colour": "#ef4444",
        "bar":    "#ef4444",
        "bg":     "#fef2f2",
    },
    "DNA damage agents": {
        "icon": "🧬",
        "desc": "Induce DNA strand breaks and trigger cell cycle arrest via checkpoint activation.",
        "colour": "#f97316",
        "bar":    "#f97316",
        "bg":     "#fff7ed",
    },
    "Eg5 inhibitors": {
        "icon": "🌀",
        "desc": "Block Eg5 kinesin motor proteins, producing monopolar mitotic spindles.",
        "colour": "#22c55e",
        "bar":    "#22c55e",
        "bg":     "#f0fdf4",
    },
    "Microtubule destabilizers": {
        "icon": "💥",
        "desc": "Depolymerise the microtubule network, collapsing the cell cytoskeleton.",
        "colour": "#06b6d4",
        "bar":    "#06b6d4",
        "bg":     "#ecfeff",
    },
    "Microtubule stabilizers": {
        "icon": "🔒",
        "desc": "Stabilise and bundle microtubules, preventing normal depolymerisation.",
        "colour": "#8b5cf6",
        "bar":    "#8b5cf6",
        "bg":     "#f5f3ff",
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

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if arr.dtype == np.uint16:
        arr = (arr >> 8).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    return arr


def percentile_normalise(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    """
    Stretch each channel independently to [0, 1] using percentile clipping.
    Makes cellular structures visible in fluorescence images with sparse intensities.
    """
    result = np.zeros(arr.shape, dtype=np.float32)
    for c in range(arr.shape[2]):
        channel = arr[:, :, c].astype(np.float32)
        p_low, p_high = np.percentile(channel, [low, high])
        if p_high > p_low:
            result[:, :, c] = np.clip((channel - p_low) / (p_high - p_low), 0.0, 1.0)
    return result


def preprocess_for_display(raw_arr: np.ndarray) -> np.ndarray:
    """Resize to 128×128 and apply per-channel percentile normalisation for display."""
    resized = cv2.resize(raw_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    return percentile_normalise(resized)


def preprocess_for_model(raw_arr: np.ndarray) -> np.ndarray:
    """
    Reproduce the exact training pipeline:
      resize → float32 in [0, 255] → resnet50.preprocess_input → add batch dim.
    No percentile normalisation — matches how the model was trained.
    """
    resized = cv2.resize(raw_arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    img_float = resized.astype(np.float32)
    img_resnet = preprocess_input(img_float)
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

        # raw_preds can be (1,5), (5,1), (5,) or a list depending on TF/Keras version.
        # tf.reshape(…, [-1]) collapses all cases to a guaranteed (num_classes,) tensor.
        if not isinstance(raw_preds, tf.Tensor):
            raw_preds = tf.convert_to_tensor(raw_preds)
        preds_flat = tf.reshape(raw_preds, [-1])

        if pred_index is None:
            pred_index = int(tf.argmax(preds_flat))

        # tf.gather on a 1-D tensor is unambiguous; [:, idx] slice indexing is not.
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
    class_colour: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.patch.set_facecolor("#ffffff")

    labels = ["Original Image", "Grad-CAM Heatmap", "Heatmap Overlay"]
    images = [original_img, heatmap, overlay_img]
    cmaps  = [None, "jet", None]

    for ax, img, title, cmap in zip(axes, images, labels, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=12, fontweight="600", color="#1a1f3c",
                     pad=10, fontfamily="DejaVu Sans")
        ax.axis("off")
        ax.set_facecolor("#f8f9ff")
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"Grad-CAM Explanation  ·  {pred_class}",
        fontsize=13, fontweight="bold", color=class_colour,
        y=1.03, fontfamily="DejaVu Sans",
    )
    plt.tight_layout(pad=1.5)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="sidebar-profile">'
        '<div class="sidebar-name">Muhammad Ertaza Manzoor</div>'
        '<div class="sidebar-uni">BSc Artificial Intelligence<br>Anglia Ruskin University</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Model Performance**")
    for label, value in [
        ("Accuracy",   "87.34 %"),
        ("Macro F1",   "0.8802"),
        ("Loss",       "0.3818"),
        ("Dataset",    "BBBC021"),
        ("Samples",    "788"),
        ("Classes",    "5 MoA"),
        ("Input size", "128 × 128 px"),
    ]:
        st.markdown(
            f'<div class="sidebar-stat-row">'
            f'<span class="sidebar-stat-label">{label}</span>'
            f'<span class="sidebar-stat-value">{value}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**MoA Classes**")
    for name, info in CLASS_INFO.items():
        with st.expander(f"{info['icon']}  {name}"):
            st.markdown(
                f'<p style="font-size:0.88rem;color:#4a5280;line-height:1.55">{info["desc"]}</p>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Image Channels**")
    st.markdown(
        "| Channel | Stain | Structure |\n"
        "|---|---|---|\n"
        "| R | DAPI | Nucleus |\n"
        "| G | Tubulin | Microtubules |\n"
        "| B | Actin | Cytoskeleton |"
    )

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🔬 Cancer Cell MoA Classifier</div>'
    '<div class="hero-sub">'
    "Deep learning classification of drug mechanism-of-action from fluorescence "
    "microscopy images of MCF-7 breast cancer cells, with Grad-CAM explainability."
    "</div>"
    '<span class="chip">ResNet50 Transfer Learning</span>'
    '<span class="chip">87.34% Accuracy</span>'
    '<span class="chip">Grad-CAM Explainability</span>'
    '<span class="chip">BBBC021 Dataset</span>'
    '<span class="chip">5 MoA Classes</span>'
    "</div>",
    unsafe_allow_html=True,
)

# ── Model availability check ──────────────────────────────────────────────────

if not model_available():
    st.markdown(
        '<div class="warn-box">'
        "⚠️ <strong>Model file not found.</strong> "
        f"Place <code>best_resnet50_transfer_model.keras</code> at "
        f"<code>{MODEL_PATH}</code> and reload the page."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()

model = load_resnet_model()

# ── Upload row ────────────────────────────────────────────────────────────────

up_col, how_col = st.columns([1.6, 1], gap="large")

with up_col:
    st.markdown('<div class="card-title">Upload Microscopy Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "image",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        label_visibility="collapsed",
    )
    st.caption(
        "Upload a 3-channel composite image (DAPI / Tubulin / Actin stacked as RGB). "
        "16-bit TIFFs are supported — contrast is enhanced automatically."
    )

with how_col:
    st.markdown('<div class="card-title">How it works</div>', unsafe_allow_html=True)
    st.markdown(
        "1. **Load** — 16-bit TIFF or PNG/JPG composite  \n"
        "2. **Enhance** — per-channel percentile normalisation  \n"
        "3. **Preprocess** — resize to 128 × 128, ResNet50 pipeline  \n"
        "4. **Classify** — softmax over 5 MoA categories  \n"
        "5. **Explain** — Grad-CAM highlights decisive regions"
    )

# ── Inference & results ───────────────────────────────────────────────────────

if uploaded_file is not None:

    pil_image  = Image.open(uploaded_file)
    raw_arr    = load_raw_array(pil_image)
    display_img = preprocess_for_display(raw_arr)   # percentile-normalised, visualisation only
    model_input = preprocess_for_model(raw_arr)     # plain uint8 → preprocess_input, matches training

    with st.spinner("Classifying…"):
        probs      = model.predict(model_input, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])

    with st.spinner("Generating Grad-CAM…"):
        heatmap     = make_gradcam_heatmap(model_input, model, LAST_CONV_LAYER, pred_index=pred_idx)
        overlay_img = overlay_gradcam(display_img, heatmap)

    class_colour = CLASS_INFO[pred_class]["colour"]
    class_icon   = CLASS_INFO[pred_class]["icon"]
    class_desc   = CLASS_INFO[pred_class]["desc"]

    # ── Prediction card ───────────────────────────────────────────────────────

    conf_pct = int(confidence * 100)
    st.markdown(
        f'<div class="pred-card">'
        f'  <div class="pred-card-label">Predicted Mechanism of Action</div>'
        f'  <div class="pred-class">{class_icon} {pred_class}</div>'
        f'  <div class="pred-sub">{class_desc}</div>'
        f'  <div class="conf-label-row">'
        f'    <span>Model Confidence</span>'
        f'    <span style="font-size:1.2rem;font-weight:900">{conf_pct}%</span>'
        f'  </div>'
        f'  <div class="conf-track">'
        f'    <div class="conf-fill" style="width:{conf_pct}%"></div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Grad-CAM panel ────────────────────────────────────────────────────────

    st.markdown(
        '<div class="gradcam-wrap">'
        '<div class="gradcam-title">🧠 Grad-CAM Explainability</div>'
        '<div class="gradcam-sub">'
        "Warm colours (red / yellow) show the cell regions that most strongly "
        "activated the predicted MoA class."
        "</div>",
        unsafe_allow_html=True,
    )
    fig = build_gradcam_figure(display_img, heatmap, overlay_img, pred_class, class_colour)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150, facecolor="#ffffff")
    st.image(buf, use_container_width=True)
    plt.close(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Probabilities + biological context ────────────────────────────────────

    prob_col, bio_col = st.columns([1, 1], gap="large")

    with prob_col:
        st.markdown(
            '<div class="section-heading">📊 Class Probabilities</div>',
            unsafe_allow_html=True,
        )
        sorted_idx = np.argsort(probs)[::-1]
        bars_html = ""
        for i in sorted_idx:
            name   = CLASS_NAMES[i]
            prob   = float(probs[i])
            info   = CLASS_INFO[name]
            is_top = i == pred_idx
            trophy = "🏆 " if is_top else ""
            weight = "800" if is_top else "600"
            bars_html += (
                f'<div class="prob-row">'
                f'  <div class="prob-label-row">'
                f'    <span class="prob-name" style="font-weight:{weight}">'
                f'      {trophy}{info["icon"]} {name}'
                f'    </span>'
                f'    <span class="prob-pct" style="color:{info["colour"]}">{prob:.1%}</span>'
                f'  </div>'
                f'  <div class="prob-track">'
                f'    <div class="prob-fill" '
                f'         style="width:{prob*100:.1f}%;background:{info["bar"]}"></div>'
                f'  </div>'
                f'</div>'
            )
        st.markdown(bars_html, unsafe_allow_html=True)

    with bio_col:
        st.markdown(
            '<div class="section-heading">🔬 Biological Context</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="bio-card">'
            f'  <div class="bio-class-name" style="color:{class_colour}">'
            f'    {class_icon} {pred_class}'
            f'  </div>'
            f'  <div class="bio-desc">{class_desc}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card">'
            '<div class="card-title">About Grad-CAM</div>'
            '<p style="font-size:0.92rem;color:#4a5280;line-height:1.65;margin:0">'
            "Gradient-weighted Class Activation Mapping (Grad-CAM) computes the "
            "gradient of the predicted class score with respect to the final "
            "convolutional layer (<code>conv5_block3_out</code>). Regions with "
            "large positive gradients are highlighted in red — these are the "
            "cellular structures the model found most discriminative."
            "</p></div>",
            unsafe_allow_html=True,
        )

    # ── Download ──────────────────────────────────────────────────────────────

    dl_col, _ = st.columns([1, 2])
    with dl_col:
        buf2 = io.BytesIO()
        fig2 = build_gradcam_figure(display_img, heatmap, overlay_img, pred_class, class_colour)
        fig2.savefig(buf2, format="png", bbox_inches="tight", dpi=200, facecolor="#ffffff")
        plt.close(fig2)
        st.download_button(
            label="⬇️  Download Grad-CAM Figure",
            data=buf2.getvalue(),
            file_name=f"gradcam_{pred_class.replace(' ', '_').lower()}.png",
            mime="image/png",
            use_container_width=True,
        )

else:
    # ── Empty state ───────────────────────────────────────────────────────────

    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">🔬</div>'
        '<div class="empty-title">Upload an image to get started</div>'
        '<div class="empty-sub">Supported formats: PNG · JPG · TIFF (8-bit and 16-bit)</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-heading" style="margin-top:2rem">📸 Example Results</div>',
        unsafe_allow_html=True,
    )
    fig_col1, fig_col2 = st.columns(2, gap="large")
    sample_figs = [
        ("outputs/figures/resnet50_gradcam_correct_samples.png",      "Correctly classified samples"),
        ("outputs/figures/resnet50_gradcam_misclassified_samples.png", "Misclassified samples"),
    ]
    for col, (path, cap) in zip([fig_col1, fig_col2], sample_figs):
        if os.path.exists(path):
            with col:
                st.markdown(
                    f'<div class="card" style="padding:1rem">'
                    f'<div class="card-title" style="margin-bottom:0.75rem">{cap}</div>',
                    unsafe_allow_html=True,
                )
                st.image(path, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="footer">'
    '  <div class="footer-name">Muhammad Ertaza Manzoor</div>'
    '  <div class="footer-uni">BSc Artificial Intelligence · Anglia Ruskin University · 2025–26</div>'
    '  <div class="footer-links">'
    '    <a class="footer-link" href="https://github.com/ezzu9" target="_blank">'
    '      ⌥ GitHub</a>'
    '    <a class="footer-link" href="https://www.linkedin.com/in/ertaza-manzoor" target="_blank">'
    '      in LinkedIn</a>'
    '    <a class="footer-link" href="https://github.com/ezzu9/cancer-cell-moa-classification" target="_blank">'
    '      🔬 Project Repo</a>'
    '  </div>'
    '</div>',
    unsafe_allow_html=True,
)
