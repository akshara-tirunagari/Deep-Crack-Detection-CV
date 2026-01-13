import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="DeepCrack Detector", page_icon="🛣️", layout="wide")

# --- CSS FOR STYLING ---
st.markdown("""
    <style>
    .critical {
        background-color: #ffcccc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #ff0000;
        color: #990000;
        font-weight: bold;
    }
    .safe {
        background-color: #ccffcc;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #009900;
        color: #006600;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- CONSTANTS ---
# Ensure these match the size you used in training (DeepCrack_Unet_Final_VGG16)
IMG_HEIGHT = 256
IMG_WIDTH = 256
CRACK_THRESHOLD_PERCENT = 1.5  # If > 1.5% of pixels are cracks, trigger warning

# --- LOAD MODEL ---
@st.cache_resource
def load_segmentation_model():
    # Update this filename to match your actual saved model file
    model_path = 'deepcrack_vgg16_unet.h5' 
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_segmentation_model()

# --- PREPROCESSING FUNCTION ---
def preprocess_image(image):
    # Resize to match model input
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img)
    
    # Normalize (0-1) as done in VGG16 preprocessing
    img_array = img_array.astype('float32') / 255.0
    
    # Expand dims to create batch of 1 (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, img

# --- MAIN APP UI ---
st.title("🛣️ DeepCrack: Road Defect Detection System")
st.markdown("Upload a road surface image to detect structural cracks using **U-Net + VGG16**.")

uploaded_file = st.file_uploader("Choose a road image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # 1. Read Image
    image = Image.open(uploaded_file)
    
    # 2. Preprocess
    input_tensor, resized_original = preprocess_image(image)
    
    # 3. Predict
    if st.button("Analyze Road Surface"):
        with st.spinner('Scanning for defects...'):
            pred_mask = model.predict(input_tensor)
            
            # The model outputs probabilities (0 to 1). We threshold at 0.5 to get binary mask.
            # Shape is likely (1, 256, 256, 1) -> squeeze to (256, 256)
            mask = (np.squeeze(pred_mask) > 0.5).astype(np.uint8)
            
            # 4. Calculate Severity
            total_pixels = mask.size
            crack_pixels = np.sum(mask)
            crack_percentage = (crack_pixels / total_pixels) * 100
            
            # 5. Display Results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(resized_original, use_column_width=True)
                
            with col2:
                st.subheader("Detected Cracks (Segmentation)")
                # Convert mask to visible image (0 -> black, 1 -> white)
                st.image(mask * 255, caption=f"Crack Density: {crack_percentage:.2f}%", use_column_width=True)
            
            # 6. Safety Logic / Warning
            st.divider()
            if crack_percentage > CRACK_THRESHOLD_PERCENT:
                st.markdown(f"""
                <div class="critical">
                    ⚠️ CRITICAL WARNING: SEVERE CRACKS DETECTED <br>
                    Crack Density: {crack_percentage:.2f}% (Threshold: {CRACK_THRESHOLD_PERCENT}%) <br>
                    Action Required: Schedule maintenance inspection immediately.
                </div>
                """, unsafe_allow_html=True)
            elif crack_percentage > 0:
                st.markdown(f"""
                <div class="safe">
                    ℹ️ MINOR DEFECTS DETECTED <br>
                    Crack Density: {crack_percentage:.2f}% <br>
                    Status: Monitor condition. No immediate action required.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ No cracks detected. Road surface is in good condition.")
