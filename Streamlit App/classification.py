import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2 as cv
import google.generativeai as genai

# Load trained DenseNet model
MODEL_PATH = "classification_model/densenet121.keras"
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# Configure Gemini API
API_KEY = ""
genai.configure(api_key=API_KEY)

# Define class labels
class_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
                'Emphysema', 'Fibrosis', 'Infiltration', 'Mass', 
                'Nodule', 'Pleural Thickening', 'Pneumothorax']

verify_image = True

# Image Preprocessing
def preprocess_image(image):
    image = image.convert('L')
    image_array = np.array(image)
    image_array = cv.GaussianBlur(image_array, (9, 9), 0)
    clahe = cv.createCLAHE(clipLimit=3, tileGridSize=(10, 10))
    clahe_image = clahe.apply(image_array)
    clahe_image = cv.cvtColor(clahe_image, cv.COLOR_GRAY2RGB)
    clahe_image = (clahe_image - clahe_image.min()) / (clahe_image.max() - clahe_image.min())
    image_resized = cv.resize(clahe_image, (320, 320))
    image_array = np.expand_dims(image_resized, axis=0).astype(np.float32)
    return image_array

def check_frontal_xray(image):
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(["Is this a frontal chest X-ray? Reply 'Yes' or 'No' only.", image])
    return response.text.strip().lower() == "yes"

# Grad-CAM Class
class Gradcam:
    def __init__(self, model, layer_name, pred_index=None):
        self.model = model
        self.layer_name = layer_name
        self.pred_index = pred_index

    def make_gradcam_heatmap(self, img_array):
        grad_model = tf.keras.models.Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            inputs = tf.cast(img_array, tf.float32)
            last_conv_layer_output, preds = grad_model(inputs)
            class_channel = preds[:, self.pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.math.divide_no_nan(tf.math.maximum(heatmap, 0), tf.reduce_max(heatmap))
        return heatmap.numpy()

# Overlay Heatmap Function
def overlay_heatmap(heatmap, img, alpha=0.4, cmap='jet'):
    heatmap = cv.resize(heatmap, (img.shape[1], img.shape[0]))  # Resize heatmap to match image (320x320)
    heatmap = np.uint8(255 * heatmap)  # Scale to 0-255
    color_map = cv.applyColorMap(heatmap, cv.COLORMAP_JET)  # Apply JET colormap
    
    # Denormalize image to 0-255 range
    img = img * 255
    superimposed_img = (color_map * alpha) + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    return superimposed_img

# Postprocess Function
def postprocess(cls_index, image_array):
    cam = Gradcam(model, pred_index=cls_index, layer_name='conv5_block16_concat')  # Adjust layer name if needed
    heatmap = cam.make_gradcam_heatmap(image_array)
    superimposed_image = overlay_heatmap(heatmap, image_array[0], alpha=0.4, cmap='jet')
    # Resize the final superimposed image to 200x200
    superimposed_image = cv.resize(superimposed_image, (200, 200))
    return superimposed_image

# Classification Page
def classification_page():
    st.title("Chest X-ray Classification")
    st.write("Classify a Chest X-ray image among 12 thoracic diseases: Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, and Pneumothorax.") 

    uploaded_file = st.file_uploader("Upload a Chest X-ray image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([250, 350])

        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_container_width=True)
        
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)

        with col2:
            # Check if image is a frontal chest X-ray
            if verify_image:
                with st.spinner('Verifying Image...'):
                    if not check_frontal_xray(image):
                        st.warning("Uploaded image is not a frontal chest X-ray. Please upload a valid image.")
                        return

            with st.spinner('Predicting Disease(s)'):
                # Predict diseases
                predictions = model.predict(preprocessed_image)[0]
                top_predictions = [(label, i) for i, (label, prob) in enumerate(zip(class_labels, predictions)) if prob > 0.8]
                
                # Display prediction results
                if top_predictions:
                    st.write("### **Predicted Disease(s):**")
                    for label, _ in top_predictions:
                        st.markdown(f'<div style="margin:10px 0; padding:10px; background-color:#FF2400; color:white; border-radius:10px; text-align:center; font-size:20px;">{label}</div>', unsafe_allow_html=True)
                else:
                    st.write("### **No diseases detected.**")

        # Display Grad-CAM Images
        if top_predictions:
                st.write("### **GradCAM Visualization:**")
                with st.spinner('Generating GradCAM Images...'):
                    cols = st.columns(len(top_predictions), gap="small")
                    for col, (label, cls_index) in zip(cols, top_predictions):
                        superimposed_image = postprocess(cls_index, preprocessed_image)
                        col.image(superimposed_image, caption=f'GradCAM for {label}', width=200)

if __name__ == "__main__":
    classification_page()