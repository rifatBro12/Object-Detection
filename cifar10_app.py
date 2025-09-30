import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import os

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU configured successfully")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Load class labels
@st.cache_data
def load_class_labels():
    try:
        with open('class_labels.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Load the trained model
@st.cache_resource
def load_model():
    try:
        # Try to load the lightweight model first
        if os.path.exists('cifar10_lightweight_model.h5'):
            model = tf.keras.models.load_model('cifar10_lightweight_model.h5')
            st.success("✅ Loaded best lightweight model (cifar10_lightweight_model.h5)")
        elif os.path.exists('cifar10_lightweight_final_model.h5'):
            model = tf.keras.models.load_model('cifar10_lightweight_final_model.h5')
            st.success("✅ Loaded final lightweight model (cifar10_lightweight_final_model.h5)")
        elif os.path.exists('cifar10_resnet50_model.h5'):
            model = tf.keras.models.load_model('cifar10_resnet50_model.h5')
            st.success("✅ Loaded ResNet50 model (cifar10_resnet50_model.h5)")
        elif os.path.exists('cifar10_resnet50_final_model.h5'):
            model = tf.keras.models.load_model('cifar10_resnet50_final_model.h5')
            st.success("✅ Loaded ResNet50 final model (cifar10_resnet50_final_model.h5)")
        else:
            st.error("❌ No trained model found. Please run the training script first.")
            return None
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image):
    """Preprocess uploaded image for prediction"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to 32x32 (CIFAR-10 size)
    image = image.resize((32, 32))
    
    # Convert to numpy array and normalize
    img_array = np.array(image) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_image(model, image_array, class_labels):
    """Make prediction on preprocessed image"""
    try:
        prediction = model.predict(image_array, verbose=0)
        predicted_class_idx = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        predicted_class = class_labels[predicted_class_idx]
        
        # Get top 3 predictions
        top_3_indices = np.argsort(prediction[0])[-3:][::-1]
        top_3_predictions = [(class_labels[idx], prediction[0][idx]) for idx in top_3_indices]
        
        return predicted_class, confidence, top_3_predictions
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        return None, None, None

def main():
    st.set_page_config(
        page_title="CIFAR-10 Object Detection",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 CIFAR-10 Object Detection System")
    st.markdown("**Upload an image to classify it into one of 10 CIFAR-10 categories**")
    
    # Load class labels
    class_labels = load_class_labels()
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # Display class labels
    st.sidebar.title("📋 Class Categories")
    for i, label in enumerate(class_labels):
        st.sidebar.write(f"{i}: {label}")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Upload an image to classify. Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Show image info
            st.write(f"**Image Size:** {image.size}")
            st.write(f"**Image Mode:** {image.mode}")
        
        with col2:
            st.subheader("🔮 Prediction Results")
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            # Make prediction
            predicted_class, confidence, top_3_predictions = predict_image(model, processed_image, class_labels)
            
            if predicted_class is not None:
                # Display main prediction
                st.success(f"**Predicted Class:** {predicted_class}")
                
                # Display top 3 predictions
                st.subheader("🏆 Top 3 Predictions")
                for i, (class_name, conf) in enumerate(top_3_predictions, 1):
                    st.write(f"{i}. {class_name}")
                
    
    # Add some sample images for testing
    st.sidebar.title("🧪 Test with Sample Images")
    st.sidebar.write("You can test the model with these sample images from the CIFAR-10 dataset:")
    
    # Check if we have sample images
    sample_dir = "cifar-10/train/train"
    if os.path.exists(sample_dir):
        sample_files = [f for f in os.listdir(sample_dir) if f.endswith('.png')][:5]
        for i, sample_file in enumerate(sample_files):
            if st.sidebar.button(f"Test Sample {i+1}"):
                sample_path = os.path.join(sample_dir, sample_file)
                sample_image = Image.open(sample_path)
                st.session_state.sample_image = sample_image
                st.session_state.sample_filename = sample_file
                st.rerun()
    
    # Handle sample image selection
    if hasattr(st.session_state, 'sample_image'):
        st.subheader("🧪 Sample Image Test")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(st.session_state.sample_image, caption=f"Sample: {st.session_state.sample_filename}", use_column_width=True)
        
        with col2:
            processed_sample = preprocess_image(st.session_state.sample_image)
            predicted_class, confidence, top_3_predictions = predict_image(model, processed_sample, class_labels)
            
            if predicted_class is not None:
                st.success(f"**Predicted Class:** {predicted_class}")
                
                st.subheader("Top 3 Predictions")
                for i, (class_name, conf) in enumerate(top_3_predictions, 1):
                    st.write(f"{i}. {class_name}")

if __name__ == "__main__":
    main()
