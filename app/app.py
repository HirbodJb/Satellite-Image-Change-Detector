"""
app/app.py
==========
Streamlit web application for the Satellite Change Detector.

Provides a dark-themed UI where users can:
  - Upload a before/after satellite image pair
  - Tune detection threshold and image size via the sidebar
  - Run the Siamese U-Net inference pipeline
  - View results as a heatmap overlay, binary change mask,
    and per-pixel change probability map

Run locally:
    streamlit run app/app.py

Deployed at:
    https://huggingface.co/spaces/HirbodJB/satellite-change-detector
"""

import sys
import os

# ---------------------------------------------------------------------------
# Make src/ importable when running from the project root or from app/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors  # noqa: F401 (available for future palette work)

from inference import ChangeDetector


# ---------------------------------------------------------------------------
# Page configuration
# Must be the first Streamlit call in the script.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Satellite Change Detector",
    page_icon="🛰️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Custom CSS
# Injects Google Fonts and overrides Streamlit defaults to achieve the
# dark satellite-dashboard aesthetic.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  /* ── Base typography & background ── */
  html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #e0e6f0;
  }
  .stApp { background-color: #0a0e1a; }

  h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
  h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }

  /* ── Metric display cards ── */
  .metric-box {
    background: linear-gradient(135deg, #1a2235, #0f1726);
    border: 1px solid #2a3a55;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
  }
  .metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #4fc3f7;
  }
  .metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #7a90b0;
    margin-top: 4px;
  }

  /* ── Change-level badge (HIGH / MODERATE / LOW) ── */
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    font-weight: 700;
  }
  .badge-high   { background:#ff3c3c22; color:#ff6b6b; border:1px solid #ff3c3c55; }
  .badge-medium { background:#ffa50022; color:#ffb74d; border:1px solid #ffa50055; }
  .badge-low    { background:#00e67622; color:#66bb6a; border:1px solid #00e67655; }

  /* ── File uploader & button styling ── */
  .stFileUploader > div { background:#0f1726; border:1px dashed #2a3a55; border-radius:12px; }
  .stButton > button {
    background: linear-gradient(90deg, #1565c0, #0288d1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 1px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
  }
  .stButton > button:hover { opacity: 0.85; }

  hr { border-color: #1e2d45; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("# 🛰️ Satellite Change Detector")
st.markdown(
    "<p style='color:#7a90b0; font-size:1.05rem; margin-top:-10px;'>"
    "Upload before &amp; after satellite images to detect and quantify surface changes.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")


# ---------------------------------------------------------------------------
# Sidebar — model & inference settings
# Users can swap the checkpoint path, adjust the binary threshold, and
# choose the resolution at which images are processed.
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    # Path to the trained model checkpoint (.pth file)
    model_path = st.text_input("Model checkpoint", value="models/best_model.pth")

    # Pixels with probability > threshold are marked as changed.
    # Lower values → more sensitive (more detections, more false positives).
    # Higher values → more conservative (fewer detections, fewer false positives).
    threshold = st.slider("Detection threshold", 0.1, 0.9, 0.5, 0.05)

    # Resolution fed to the model. 256 matches the LEVIR-CD training crop size.
    img_size = st.selectbox("Image size", [256, 512], index=0)

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#4a6080;'>"
        "Trained on LEVIR-CD + LEVIR-CD+ dataset.<br>"
        "Architecture: Siamese U-Net (ResNet-34 encoder)<br>"
        "Loss: BCE + Dice &nbsp;|&nbsp; IoU: 0.70 &nbsp;|&nbsp; F1: 0.79</p>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Image upload — two columns: Before (A) and After (B)
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Before Image")
    before_file = st.file_uploader(
        "Upload BEFORE image",
        type=["png", "jpg", "jpeg", "tif"],
        key="before",
    )
    if before_file is not None:
        # Preview the uploaded image immediately
        st.image(before_file, use_container_width=True)

with col2:
    st.markdown("#### After Image")
    after_file = st.file_uploader(
        "Upload AFTER image",
        type=["png", "jpg", "jpeg", "tif"],
        key="after",
    )
    if after_file is not None:
        st.image(after_file, use_container_width=True)


# ---------------------------------------------------------------------------
# Inference trigger
# ---------------------------------------------------------------------------
st.markdown("")
run_btn = st.button("🔍 DETECT CHANGES")

if run_btn:
    # ── Validation ──────────────────────────────────────────────────────────
    if before_file is None or after_file is None:
        st.warning("Please upload both a Before and After image.")

    elif not os.path.exists(model_path):
        st.error(
            f"Model not found at `{model_path}`. "
            "Train the model first — see README, or adjust the path in the sidebar."
        )

    else:
        # ── Run inference ────────────────────────────────────────────────────
        with st.spinner("Running inference..."):
            # Load model onto CPU (HF Spaces free tier has no GPU)
            detector = ChangeDetector(model_path, device="cpu")

            before_pil = Image.open(before_file)
            after_pil  = Image.open(after_file)

            # predict() returns heatmap, binary mask, probability map, and change %
            # TTA (Test Time Augmentation) is enabled by default inside predict()
            result = detector.predict(
                before_pil,
                after_pil,
                threshold=threshold,
                img_size=img_size,
            )

        st.markdown("---")
        st.markdown("## 📊 Results")

        # ── Metric cards ─────────────────────────────────────────────────────
        pct = result["change_pct"]

        # Assign a human-readable severity badge based on area changed
        if pct > 20:
            badge = '<span class="badge badge-high">HIGH CHANGE</span>'
        elif pct > 5:
            badge = '<span class="badge badge-medium">MODERATE CHANGE</span>'
        else:
            badge = '<span class="badge badge-low">LOW CHANGE</span>'

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{pct:.1f}%</div>'
                f'<div class="metric-label">Area Changed</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m2:
            changed_px = int(result["mask"].sum())
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value">{changed_px:,}</div>'
                f'<div class="metric-label">Changed Pixels</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                f'<div class="metric-box">'
                f'<div class="metric-value" style="font-size:1.4rem">{badge}</div>'
                f'<div class="metric-label">Change Level</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Visualisations ───────────────────────────────────────────────────
        v1, v2, v3 = st.columns(3)

        with v1:
            # Heatmap: original before image with changed pixels painted red
            st.markdown("**Heatmap Overlay**")
            st.image(result["heatmap"], use_container_width=True)

        with v2:
            # Binary mask: white = changed, black = no change
            st.markdown("**Binary Change Mask**")
            mask_display = (result["mask"] * 255).astype(np.uint8)
            st.image(mask_display, use_container_width=True, clamp=True)

        with v3:
            # Probability map: plasma colormap showing raw model confidence
            # Yellow = high probability of change, purple = low
            st.markdown("**Change Probability Map**")
            fig, ax = plt.subplots(figsize=(4, 4))
            fig.patch.set_facecolor("#0a0e1a")
            ax.set_facecolor("#0a0e1a")
            im   = ax.imshow(result["prob_map"], cmap="plasma", vmin=0, vmax=1)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)  # Free memory — important in long-running Streamlit sessions