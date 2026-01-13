# Deep Crack Detection: Infrastructure Safety System 🛣️🔍

Automated road crack detection system using U-Net architecture with VGG16 encoder. Includes a Streamlit web app for real-time infrastructure analysis.

This project implements a computer vision system to automatically detect and segment cracks in road surfaces (concrete/asphalt). Using a **U-Net architecture with a VGG16 backbone**, the model provides pixel-level crack segmentation to assist in infrastructure maintenance and safety inspections.

## 🎥 Project Demo

*The system takes raw road images (left) and outputs a binary segmentation mask (right) highlighting the crack location.*

## 📂 Repository Structure
* **`app.py`**: The main Streamlit application script. Deploys the trained model in a web interface for real-time crack detection.
* **`Model_Training_UNet.ipynb`**: The training pipeline. Includes data augmentation, VGG16 transfer learning, and U-Net decoder implementation.
* **`Demo_Notebook.ipynb`**: Experimental notebook used to prototype the application logic.
* **`Project_Presentation.pdf`**: Executive summary, business use case, and performance metrics (IoU, F1 Score).
* **`prediction_metrics.csv`**: Log of model predictions and confidence scores on the test set.

## 🧠 Technical Approach

### 1. Data Source
* **Dataset:** [DeepCrack](https://github.com/yhlleo/DeepCrack)
* **Preprocessing:** Images were resized (256x256), normalized, and augmented (rotation, flipping) to improve model robustness.

### 2. Model Architecture: U-Net + VGG16
We utilized a **U-Net** architecture for semantic segmentation, replacing the standard encoder with a pre-trained **VGG16** network.
* **Encoder (Downsampling):** VGG16 (ImageNet weights) extracts deep features (edges $\rightarrow$ textures $\rightarrow$ object parts).
* **Decoder (Upsampling):** Reconstructs the segmentation mask using skip connections to preserve spatial details from the encoder.
* **Output:** A pixel-wise probability map where `1` = Crack and `0` = Background.

### 3. Business Logic (Streamlit App)
The `app.py` script applies a safety thresholding logic:
* **Detection:** The model predicts a segmentation mask.
* **Analysis:** The app calculates the percentage of "crack pixels" in the frame.
* ⚠️ **Alert:** If crack density > 1.5%, the system triggers a **"CRITICAL WARNING"** for immediate maintenance.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow/Keras, VGG16.
* **Computer Vision:** OpenCV, Scikit-Image.
* **Web App:** Streamlit.
* **Visualization:** Matplotlib, Seaborn.

## 🚀 How to Run the App
To run the dashboard locally:

1. **Clone the repository**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Deep-Crack-Detection-CV.git](https://github.com/YOUR_USERNAME/Deep-Crack-Detection-CV.git)

2. **Install dependencies**
   ```bash
   pip install tensorflow streamlit opencv-python matplotlib

3. **Download Model Weights**
   
   Note: The trained model file (`deepcrack_vgg16_unet.h5`) is required to run the app. If not present in the repo due to size limits, please train the model using `Model_Training_UNet.ipynb` to generate it.

5. **Launch the App**
   ```bash
   streamlit run app.py

------
