import html
import os
import textwrap

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

from utils.preprocessing import preprocess_single_image

MODEL_PATH = os.path.join('models', 'traffic_sign_cnn.keras')
NUM_CLASSES = 5
CLASS_NAMES = ['Stop', 'No Entry', 'Speed Limit 50', 'Yield', 'Pedestrian Crossing']

CLASS_ICONS = {
    'Stop': '🛑',
    'No Entry': '⛔',
    'Speed Limit 50': '🚫',
    'Yield': '⚠️',
    'Pedestrian Crossing': '🚸',
}

st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
)

CSS = """
<style>
:root {
    --navy-950: #050d17;
    --navy-900: #071523;
    --navy-800: #0b1f36;
    --navy-700: #0c2941;
    --white: #f5f9ff;
    --muted: #9fb3c8;
    --green: #22c55e;
    --green-light: #4ade80;
    --green-soft: rgba(34, 197, 94, 0.14);
}

html, body, [class*="css"] {
    font-family: "Segoe UI", "Inter", sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% -10%, rgba(34, 197, 94, 0.12), transparent 35%),
        radial-gradient(circle at 88% 0%, rgba(56, 189, 248, 0.10), transparent 32%),
        linear-gradient(180deg, #071523 0%, #050d17 100%);
    color: var(--white);
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.6rem;
    padding-bottom: 1rem;
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] h1,
[data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3 {
    color: var(--white);
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    border-radius: 24px;
    padding: 2.5rem 2.2rem;
    margin-bottom: 1.8rem;
    background: linear-gradient(125deg, #0b1f36 0%, #0c2941 55%, #08382a 100%);
    border: 1px solid rgba(74, 222, 128, 0.22);
    box-shadow: 0 24px 60px rgba(0, 0, 0, 0.3);
}

.hero::before {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.20), transparent 65%);
    top: -160px;
    right: -80px;
}

.hero::after {
    content: "";
    position: absolute;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.12), transparent 65%);
    bottom: -150px;
    left: -70px;
}

.hero > * {
    position: relative;
    z-index: 1;
}

.hero-signs {
    font-size: 2.4rem;
    letter-spacing: 0.35rem;
    margin-bottom: 0.9rem;
}

.hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #86efac;
    font-weight: 700;
}

.hero-eyebrow .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #22c55e;
    border-radius: 50%;
    box-shadow: 0 0 0 5px rgba(34, 197, 94, 0.15);
}

.hero-title {
    font-size: 2.9rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.08;
    margin: 0.35rem 0 0.25rem;
    color: #ffffff !important;
}

.hero-subtitle {
    color: #b9cbe0 !important;
    font-size: 1.08rem;
    margin: 0 0 1.1rem;
}

.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.badge {
    background: rgba(34, 197, 94, 0.10);
    color: #86efac !important;
    border: 1px solid rgba(34, 197, 94, 0.28);
    padding: 0.32rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
}

/* Section labels */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9fb3c8 !important;
    font-weight: 700;
    margin: 0.2rem 0 0.8rem;
}

.section-label .step {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    background: var(--green-soft);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.35);
    font-size: 0.78rem;
}

/* Uploader */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.035);
    border: 1px dashed rgba(74, 222, 128, 0.42);
    border-radius: 18px;
    padding: 0.85rem;
}

[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"] {
    background: transparent;
    border: none;
}

[data-testid="stFileUploader"] small {
    color: #9fb3c8 !important;
}

[data-testid="stFileUploader"] button {
    background: rgba(34, 197, 94, 0.14);
    color: #86efac;
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-radius: 10px;
}

/* Image preview */
[data-testid="stImage"] img {
    display: block;
    margin: 0 auto;
    border-radius: 16px;
    border: 1px solid rgba(74, 222, 128, 0.28);
    box-shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
}

[data-testid="stImage"] figcaption {
    color: #9fb3c8 !important;
    text-align: center;
}

/* Placeholders */
.preview-placeholder,
.result-placeholder {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 250px;
    margin-bottom: 0.9rem;
    padding: 2.2rem 1.2rem;
    text-align: center;
    border: 1px dashed rgba(148, 163, 184, 0.3);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.025);
}

.result-placeholder {
    min-height: 330px;
}

.placeholder-icon {
    font-size: 2rem;
    margin-bottom: 0.4rem;
}

.preview-placeholder p,
.result-placeholder p {
    margin: 0.15rem 0;
    color: #d7e4f3 !important;
}

.muted {
    color: #8ca2b8 !important;
    font-size: 0.85rem;
}

.upload-hint {
    color: #8ca2b8 !important;
    font-size: 0.82rem;
    text-align: center;
    margin-top: 0.35rem;
}

/* Analyze button */
[data-testid="stButton"] > button {
    width: 100%;
    border-radius: 12px;
    border: 1px solid rgba(74, 222, 128, 0.35);
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 55%, #4ade80 100%);
    color: #04160a !important;
    font-size: 1rem;
    font-weight: 800;
    padding: 0.72rem 1rem;
    box-shadow: 0 12px 28px rgba(34, 197, 94, 0.22);
    transition: all 0.18s ease;
}

[data-testid="stButton"] > button:hover {
    transform: translateY(-1px);
    border-color: rgba(74, 222, 128, 0.7);
    box-shadow: 0 16px 34px rgba(34, 197, 94, 0.3);
    color: #04160a !important;
}

[data-testid="stButton"] > button:disabled {
    background: rgba(255, 255, 255, 0.07);
    color: #7d90a6 !important;
    border-color: rgba(255, 255, 255, 0.1);
    box-shadow: none;
    transform: none;
}

[data-testid="stButton"] > button p {
    color: inherit !important;
    font-weight: 800;
}

/* Result card */
.result-card {
    padding: 1.6rem 1.2rem;
    text-align: center;
    border-radius: 20px;
    border: 1px solid rgba(74, 222, 128, 0.4);
    background: linear-gradient(160deg, rgba(34, 197, 94, 0.18), rgba(255, 255, 255, 0.035));
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06), 0 18px 40px rgba(0, 0, 0, 0.25);
    animation: rise 0.35s ease-out;
}

.result-eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.16em;
    font-size: 0.7rem;
    color: #86efac !important;
    font-weight: 700;
}

.result-icon {
    font-size: 2.6rem;
    margin: 0.55rem 0 0.15rem;
}

.result-class {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1.15;
    color: #ffffff !important;
}

.result-confidence {
    color: #b9cbe0 !important;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}

@keyframes rise {
    from {
        opacity: 0;
        transform: translateY(8px);
    }
    to {
        opacity: 1;
        transform: none;
    }
}

/* Progress bar */
[data-testid="stProgress"] {
    margin-top: 0.35rem;
}

[data-testid="stProgress"] > div > div > div > div,
[data-testid="stProgress"] [role="progressbar"] > div {
    background: linear-gradient(90deg, #16a34a, #4ade80) !important;
}

[data-testid="stProgress"] p {
    color: #9fb3c8 !important;
    font-size: 0.82rem;
}

/* Supported classes */
.sign-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 0.7rem;
    margin-bottom: 1.2rem;
}

.sign-card {
    padding: 1rem 0.5rem;
    text-align: center;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.02));
    transition: all 0.18s ease;
}

.sign-card:hover {
    transform: translateY(-2px);
    border-color: rgba(74, 222, 128, 0.5);
    background: rgba(34, 197, 94, 0.07);
}

.sign-emoji {
    display: block;
    font-size: 1.7rem;
    margin-bottom: 0.35rem;
}

.sign-name {
    display: block;
    color: #e8f1ff !important;
    font-size: 0.82rem;
    font-weight: 600;
    line-height: 1.2;
}

/* Alerts */
[data-testid="stAlert"] {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(148, 163, 184, 0.2);
    border-radius: 14px;
}

[data-testid="stAlert"] p {
    color: #e8f1ff !important;
}

/* Footer */
.footer {
    margin-top: 1.4rem;
    padding-top: 1.1rem;
    text-align: center;
    color: #7d90a6 !important;
    font-size: 0.8rem;
    border-top: 1px solid rgba(148, 163, 184, 0.13);
}

/* Responsive */
@media (max-width: 900px) {
    .hero-title {
        font-size: 2.3rem;
    }

    .sign-grid {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }
}

@media (max-width: 640px) {
    .block-container {
        padding-top: 1rem;
    }

    .hero {
        padding: 1.5rem 1.1rem;
    }

    .hero-title {
        font-size: 1.9rem;
    }

    .hero-subtitle {
        font-size: 0.95rem;
    }

    .hero-signs {
        font-size: 2rem;
    }

    .sign-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .result-class {
        font-size: 1.55rem;
    }
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


def render_html(markup):
    st.markdown(textwrap.dedent(markup).strip(), unsafe_allow_html=True)


def render_hero():
    render_html("""
        <div class="hero">
            <div class="hero-signs">🚦 <span>🛑</span> <span>⛔</span> <span>⚠️</span></div>
            <div class="hero-eyebrow"><span class="dot"></span> CNN-Powered Classifier</div>
            <h1 class="hero-title">Traffic Sign Recognition</h1>
            <p class="hero-subtitle">AI-Powered Road Sign Detection Using CNN</p>
            <div class="hero-badges">
                <span class="badge">5 Sign Classes</span>
                <span class="badge">32 × 32 Input</span>
                <span class="badge">Confidence Scoring</span>
            </div>
        </div>
    """)


def render_supported_classes():
    cards = "".join(
        f'<div class="sign-card">'
        f'<span class="sign-emoji">{CLASS_ICONS[name]}</span>'
        f'<span class="sign-name">{html.escape(name)}</span>'
        f'</div>'
        for name in CLASS_NAMES
    )
    render_html(f"""
        <div class="section-label">Supported Traffic Signs</div>
        <div class="sign-grid">{cards}</div>
    """)


def main():
    render_hero()

    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}. Please train the model first using train.py")
        st.stop()

    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    left_col, right_col = st.columns([1.05, 1], gap="large")

    with left_col:
        render_html('<div class="section-label"><span class="step">1</span> Upload Image</div>')
        uploaded_file = st.file_uploader(
            "Drag and drop a traffic sign image here",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            label_visibility="collapsed",
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", width=360)
            render_html(
                '<p class="upload-hint">'
                'Preview shown at its natural size. The model analyzes a 32 × 32 version.'
                '</p>'
            )
        else:
            render_html("""
                <div class="preview-placeholder">
                    <div class="placeholder-icon">🖼️</div>
                    <p><strong>No image uploaded yet</strong></p>
                    <p class="muted">Drag and drop a JPG, JPEG, PNG, or BMP file above.</p>
                </div>
            """)

        analyze_clicked = st.button(
            "🚦 Analyze Traffic Sign",
            type="primary",
            use_container_width=True,
            disabled=uploaded_file is None,
        )

    with right_col:
        render_html('<div class="section-label"><span class="step">2</span> Prediction Result</div>')

        if analyze_clicked and uploaded_file is not None:
            try:
                temp_path = os.path.join('outputs', 'temp_upload.jpg')
                os.makedirs('outputs', exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())

                img = preprocess_single_image(temp_path)
                predictions = model.predict(img)
                pred_class = np.argmax(predictions[0])
                confidence = float(predictions[0][pred_class]) * 100
                predicted_name = CLASS_NAMES[pred_class]

                render_html(f"""
                    <div class="result-card">
                        <div class="result-eyebrow">Predicted Class</div>
                        <div class="result-icon">{CLASS_ICONS.get(predicted_name, '🚦')}</div>
                        <div class="result-class">{html.escape(predicted_name)}</div>
                        <div class="result-confidence">Confidence: {confidence:.2f}%</div>
                    </div>
                """)

                st.progress(confidence / 100.0, text=f"Confidence: {confidence:.2f}%")

                try:
                    os.remove(temp_path)
                except Exception:
                    pass

            except Exception as e:
                st.error(f"Error processing image: {e}")
                st.info("Please ensure the image is a valid traffic sign image.")

        elif uploaded_file is None:
            render_html("""
                <div class="result-placeholder">
                    <div class="placeholder-icon">📊</div>
                    <p><strong>Waiting for an image</strong></p>
                    <p class="muted">Upload a sign to see the CNN prediction.</p>
                </div>
            """)
        else:
            render_html("""
                <div class="result-placeholder">
                    <div class="placeholder-icon">👈</div>
                    <p><strong>Ready to analyze</strong></p>
                    <p class="muted">Press “Analyze Traffic Sign” to run the CNN model.</p>
                </div>
            """)

    render_supported_classes()
    render_html('<div class="footer">Traffic Sign Recognition | CNN Mini Project</div>')


if __name__ == '__main__':
    main()
