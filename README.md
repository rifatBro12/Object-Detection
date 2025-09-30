# 🔍 CIFAR-10 Object Detection System

A complete deep learning project for object detection and classification using the CIFAR-10 dataset. This project includes model training, optimization for GPU memory, and a user-friendly web interface for real-time predictions.

## 📋 Project Overview

This project implements a lightweight CNN model optimized for 6GB GPU memory to classify images into 10 CIFAR-10 categories:
- ✈️ Airplane
- 🚗 Automobile  
- 🐦 Bird
- 🐱 Cat
- 🦌 Deer
- 🐕 Dog
- 🐸 Frog
- 🐴 Horse
- 🚢 Ship
- 🚛 Truck

## 🚀 Features

- **GPU-Optimized Training**: Configured for 6GB RTX 4050 GPU with memory growth
- **Lightweight CNN Architecture**: 425K parameters for efficient inference
- **Real-time Web Interface**: Streamlit-based prediction app
- **Model Performance**: 68.35% test accuracy on CIFAR-10 dataset
- **Memory Efficient**: Optimized for 8GB RAM + 6GB GPU setup
- **Easy-to-Use**: Simple drag-and-drop image upload interface

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (6GB+ VRAM recommended)
- 8GB+ RAM

### Setup
1. **Clone or download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify GPU setup**:
   ```bash
   nvidia-smi
   ```

## 📁 Project Structure

```
deep learning projects/
├── 📊 10-object-detection.ipynb          # Original Jupyter notebook
├── 🐍 train_lightweight_model.py         # Training script (optimized)
├── 🐍 train_model.py                     # Alternative training script
├── 🌐 cifar10_app.py                     # Streamlit web app
├── 🌐 app.py                             # Alternative web app
├── 📋 requirements.txt                   # Dependencies
├── 📖 README.md                          # This file
├── 📁 cifar-10/                          # Dataset directory
│   ├── train/
│   └── trainLabels.csv
├── 🤖 cifar10_lightweight_model.h5       # Best trained model
├── 🤖 cifar10_lightweight_final_model.h5 # Final model
├── 📄 cifar10_lightweight_model.json     # Model architecture
└── 🏷️ class_labels.json                 # Class labels
```

## 🚀 Quick Start

### 1. Train the Model
```bash
python train_lightweight_model.py
```
This will:
- Load and preprocess CIFAR-10 data
- Train a lightweight CNN model
- Save the best model automatically
- Display training progress and final accuracy

### 2. Run the Web App
```bash
streamlit run cifar10_app.py
```
Then open your browser to `http://localhost:8501`

### 3. Make Predictions
1. Upload any image (PNG, JPG, JPEG)
2. Get instant classification results
3. View top 3 predictions

## 🧠 Model Architecture

The lightweight CNN model includes:
- **3 Convolutional Blocks** with BatchNormalization
- **Global Average Pooling** for memory efficiency
- **Dropout layers** for regularization
- **Dense layers** for final classification
- **Total Parameters**: 425,386 (optimized for 6GB GPU)

## 📊 Performance

- **Test Accuracy**: 68.35%
- **Training Time**: ~30-60 minutes (depending on GPU)
- **Model Size**: ~10MB
- **Inference Speed**: Real-time predictions

## 🎯 Usage Examples

### Training
```python
# Run the optimized training script
python train_lightweight_model.py
```

### Web Interface
```python
# Launch the prediction app
streamlit run cifar10_app.py
```

### Programmatic Usage
```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('cifar10_lightweight_model.h5')

# Preprocess image
image = Image.open('your_image.jpg')
image = image.resize((32, 32))
image_array = np.array(image) / 255.0
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array)
predicted_class = np.argmax(prediction[0])
```

## ⚙️ Configuration

### GPU Memory Optimization
The model is configured for 6GB GPU memory:
- Memory growth enabled
- Batch size: 32
- Model checkpointing
- Early stopping
- Learning rate reduction

### Hardware Requirements
- **Minimum**: 4GB GPU, 8GB RAM
- **Recommended**: 6GB+ GPU, 16GB+ RAM
- **Optimal**: RTX 4050/4060 or better

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory Error**:
   - Reduce batch size in training script
   - Close other GPU-intensive applications

2. **Model Loading Error**:
   - Ensure model files are in the same directory
   - Check TensorFlow version compatibility

3. **Streamlit App Issues**:
   - Clear browser cache
   - Try different port: `streamlit run cifar10_app.py --server.port 8502`

### Performance Tips
- Use GPU for training (CPU is much slower)
- Close unnecessary applications during training
- Monitor GPU memory usage with `nvidia-smi`

## 📈 Model Training Details

### Training Configuration
- **Epochs**: 50 (with early stopping)
- **Batch Size**: 32
- **Optimizer**: Adam
- **Learning Rate**: 0.001 (with reduction on plateau)
- **Validation Split**: 10%

### Callbacks
- **ModelCheckpoint**: Saves best model
- **EarlyStopping**: Prevents overfitting
- **ReduceLROnPlateau**: Adaptive learning rate

## 🤝 Contributing

Feel free to contribute by:
- Improving model architecture
- Adding new features to the web app
- Optimizing for different hardware setups
- Adding more datasets

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- TensorFlow/Keras team
- Streamlit team
- Deep learning community

## 📞 Support

If you encounter any issues:
1. Check the troubleshooting section
2. Verify your hardware meets requirements
3. Ensure all dependencies are installed correctly

---

**Happy Classifying! 🎉**
