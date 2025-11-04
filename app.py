# --- rice_analysis_app.py ---
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import os
from rice_analyzer import analyze_rice_image, MIN_PIXEL_AREA

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = 'rice_cnn_model.h5'
CLASS_NAMES_PATH = 'rice_class_names.json'

@st.cache_resource
def load_cnn_model(model_path, class_path):
    try:
        if not os.path.exists(model_path) or not os.path.exists(class_path):
            st.error("Model or class file missing. Ensure both files are in the project directory.")
            return None, []
        model = tf.keras.models.load_model(model_path)
        with open(class_path, 'r') as f:
            class_names = json.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, []

cnn_model, class_names = load_cnn_model(MODEL_PATH, CLASS_NAMES_PATH)

# -------------------------------
# PAGE CONFIG & GLOBAL STYLE
# -------------------------------
st.set_page_config(layout="wide", page_title="Rice Quality Analyzer")

st.markdown("""
<style>
body, .stApp {
    background-color: #ffffff !important;
    color: #000000 !important;
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif !important;
}

/* Headings and text */
h1, h2, h3, h4, h5, label, p, span, small {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e0e0e0 !important;
}
section[data-testid="stSidebar"] * {
    color: #000000 !important;
}

/* Buttons */
div.stButton > button, .stDownloadButton button, .stCameraInput button, .stFileUploader div div div button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5em 1.2em !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.15) !important;
}
div.stButton > button:hover, .stDownloadButton button:hover, .stCameraInput button:hover, .stFileUploader div div div button:hover {
    background-color: #f7f7f7 !important;
    transform: translateY(-1px);
}

/* Input boxes */
div[data-baseweb="input"] input, input[type=number], input[type=text], textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1.5px solid #000000 !important;
    border-radius: 6px !important;
    padding: 6px 8px !important;
}
div[data-baseweb="input"] svg {
    fill: #000000 !important;
}

/* File uploader */
.stFileUploader {
    background-color: #ffffff !important;
    border: 1.5px solid #000000 !important;
    border-radius: 8px !important;
    padding: 15px !important;
    color: #000000 !important;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.1) !important;
}
.stFileUploader div, 
.stFileUploader span, 
.stFileUploader p, 
.stFileUploader label, 
.stFileUploader small, 
.stFileUploader button, 
.stFileUploader svg {
    color: #000000 !important;
    fill: #000000 !important;
    font-weight: 500 !important;
}
.stFileUploader div[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #000000 !important;
    color: #000000 !important;
}

/* Camera input */
.stCameraInput {
    background-color: #ffffff !important;
    border: 1.5px solid #000000 !important;
    border-radius: 8px !important;
    padding: 15px !important;
}
.stCameraInput div, .stCameraInput span, .stCameraInput p, .stCameraInput button {
    color: #000000 !important;
}

/* Metrics and tables */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 700 !important;
}
.stDataFrame {
    border: 1px solid #00000020 !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# NAVIGATION
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "about"

def go_to(page):
    st.session_state.page = page

# -------------------------------
# PAGE 1: ABOUT (REDESIGNED)
# -------------------------------
if st.session_state.page == "about":
    st.markdown("""
        <div style="text-align:center; padding-top:20px; padding-bottom:10px;">
            <h1 style="color:#1A237E; font-size:40px; font-weight:700;">Qualty.Ai</h1>
            <p style="font-size:18px; color:#555;">
                Intelligent Rice Quality Inspection Powered by Deep Learning
            </p>
        </div>
        <hr style="border: 1px solid #e0e0e0; margin: 20px 0;">
        """, unsafe_allow_html=True)

    st.markdown("""
        <h3 style="color:#000000;">Overview</h3>
        <p style="font-size:16px; color:#000000; text-align:justify;">
        Qualty.Ai is a computer vision-powered rice quality inspection platform that automatically detects,
        measures, and classifies rice grains to enhance agricultural and industrial quality control.
        </p>

        <hr style="border: 1px solid #e0e0e0; margin: 20px 0;">

        <h3 style="color:#000000;">Core Features</h3>
        <ul style="color:#000000; font-size:15px; line-height:1.6;">
            <li>Accurate detection of each grain — even in dense or overlapping images.</li>
            <li>Automatic measurement of average grain <b>length</b>, <b>width</b>, and <b>ratio</b>.</li>
            <li>AI classification of grains into <b>Whole</b>, <b>Broken</b>, and <b>Discolored (DD)</b>.</li>
            <li>Instant visual output with detailed analytical data.</li>
        </ul>

        <hr style="border: 1px solid #e0e0e0; margin: 20px 0;">

        <h3 style="color:#000000;">Recent Improvements</h3>
        <ul style="color:#000000; font-size:15px; line-height:1.6;">
            <li>Improved broken rice detection and precision.</li>
            <li>Automatic handling of null or missing parameters.</li>
            <li>Optimized detection for images with extra spacing or multiple clusters.</li>
        </ul>

        <div style="text-align:center; padding-top:15px;">
            <p style="color:#000000; font-size:17px;">
                Ready to begin? Start analyzing your sample image.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("Proceed to Image Input & Calibration", on_click=lambda: go_to("input"))
    st.markdown("<hr><center><small>© 2025 Qualty.Ai    </small></center>", unsafe_allow_html=True)

# -------------------------------
# PAGE 2: IMAGE INPUT & CALIBRATION
# -------------------------------
elif st.session_state.page == "input":
    st.title("📸 Image Input & Calibration")
    st.markdown("Upload or capture your rice grain image for analysis below.")

    with st.sidebar:
        st.header("Calibration Settings")
        known_mm_length = st.number_input(
            "Known Grain Length (mm):",
            min_value=0.1, value=6.0, step=0.1, format="%.2f",
            help="Used to convert pixels into millimeters."
        )
        st.markdown("---")
        st.caption("Developed using OpenCV + TensorFlow")

    mode = st.radio("Select Image Source", ["Upload from Device", "Use Camera"], horizontal=True)
    selected_file = (
        st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if mode == "Upload from Device"
        else st.camera_input("Capture Using Camera")
    )

    if selected_file:
        st.session_state.selected_image = selected_file
        st.session_state.known_mm_length = known_mm_length
        st.success("✅ Image successfully loaded.")
        st.button("Run Analysis", on_click=lambda: go_to("analysis"))

    st.button("⬅ Back to Home", on_click=lambda: go_to("about"))

# -------------------------------
# PAGE 3: ANALYSIS RESULTS
# -------------------------------
elif st.session_state.page == "analysis":
    st.title("📊 Analysis Results")

    if "selected_image" not in st.session_state:
        st.error("Please upload or capture an image first.")
        st.button("Go Back", on_click=lambda: go_to("input"))
    else:
        try:
            selected_file = st.session_state.selected_image
            known_mm_length = st.session_state.known_mm_length

            file_bytes = np.asarray(bytearray(selected_file.read()), dtype=np.uint8)
            img_cv = cv2.imdecode(file_bytes, 1)

            # --- Calibration ---
            hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 70, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)
            blurred = cv2.GaussianBlur(mask, (5, 5), 0)
            _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            markers = cv2.connectedComponents(sure_fg)[1]
            markers += 1

            calibration_pixel_length = 0
            max_area = 0
            for marker_id in np.unique(markers):
                if marker_id <= 1:
                    continue
                mask_single = np.zeros(img_cv.shape[:2], dtype=np.uint8)
                mask_single[markers == marker_id] = 255
                contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours and cv2.contourArea(contours[0]) > MIN_PIXEL_AREA:
                    contour = contours[0]
                    area = cv2.contourArea(contour)
                    if area > max_area:
                        max_area = area
                        rect = cv2.minAreaRect(contour)
                        width, length = rect[1]
                        if width > length:
                            length, width = width, length
                        calibration_pixel_length = length

            if calibration_pixel_length > 0:
                scale_factor = known_mm_length / calibration_pixel_length
                st.sidebar.success(f"Scale Factor: {scale_factor:.4f} mm/pixel")
                calibration_successful = True
            else:
                calibration_successful = False
                st.error("Calibration failed. Please use a clear image.")

            # --- Analysis ---
            if calibration_successful:
                analysis_img, results_dict, total_grains = analyze_rice_image(
                    img_cv=img_cv,
                    scale_factor=scale_factor,
                    cnn_model=cnn_model,
                    class_names=class_names
                )

                st.subheader("Key Metrics")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Grains", f"{results_dict['total_grains']}")
                c2.metric("Avg Length", f"{results_dict['avg_length_mm']:.2f} mm")
                c3.metric("Broken %", f"{results_dict['broken_percent']:.2f} %")
                c4.metric("DD %", f"{results_dict['dd_percent']:.2f} %")

                st.markdown("---")
                st.subheader("Detailed Grain Statistics")

                table_data = {
                    "Parameter": [
                        "Total Grains", "Avg Ratio", "Avg Length (mm)", "Avg Width (mm)",
                        "Whole Count", "Broken Count", "DD Count", "Strip Count",
                        "Tip Count", "Foreign Count", "Broken %", "DD %"
                    ],
                    "Value": [
                        results_dict["total_grains"],
                        f"{results_dict['avg_ratio']:.2f}",
                        f"{results_dict['avg_length_mm']:.2f}",
                        f"{results_dict['avg_width_mm']:.2f}",
                        results_dict["good_rice"],
                        results_dict["broken_count"],
                        results_dict["dd_count"],
                        results_dict["strip_count"],
                        results_dict["tip_count"],
                        results_dict["foreign_count"],
                        f"{results_dict['broken_percent']:.2f}",
                        f"{results_dict['dd_percent']:.2f}"
                    ]
                }

                st.dataframe(pd.DataFrame(table_data))
                st.markdown("---")
                st.image(analysis_img, caption="Detected & Classified Grains", use_container_width=True, channels="BGR")

        except Exception as e:
            st.error(f"Error during analysis: {str(e)}")

        st.button("🔁 Back to Calibration", on_click=lambda: go_to("input"))
        st.button("🏠 Return Home", on_click=lambda: go_to("about"))
