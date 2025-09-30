#!/usr/bin/env python3
"""
Script to run the CIFAR-10 training notebook with GPU optimization
"""

import subprocess
import sys
import os

def run_notebook():
    """Run the Jupyter notebook to train the model"""
    print("Starting CIFAR-10 model training with GPU optimization...")
    print("This will train a ResNet50-based model on CIFAR-10 dataset")
    print("Model will be automatically saved after training")
    print("Training may take 30-60 minutes depending on your GPU")
    print("-" * 60)
    
    try:
        # Convert notebook to Python script and run it
        result = subprocess.run([
            sys.executable, "-m", "jupyter", "nbconvert", 
            "--to", "python", 
            "--execute", 
            "10-object-detection.ipynb"
        ], check=True, capture_output=True, text=True)
        
        print("Notebook executed successfully!")
        print("Model files should be saved in the current directory")
        
    except subprocess.CalledProcessError as e:
        print(f"Error running notebook: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

def check_model_files():
    """Check if model files were created"""
    model_files = [
        'cifar10_resnet50_model.h5',
        'cifar10_resnet50_final_model.h5',
        'cifar10_resnet50_model.json',
        'cifar10_resnet50_weights.h5',
        'class_labels.json'
    ]
    
    print("\nChecking for saved model files...")
    found_files = []
    missing_files = []
    
    for file in model_files:
        if os.path.exists(file):
            found_files.append(file)
            print(f"Found: {file}")
        else:
            missing_files.append(file)
            print(f"Missing: {file}")
    
    if missing_files:
        print(f"\nMissing files: {missing_files}")
        print("The training may not have completed successfully.")
        return False
    else:
        print("\nAll model files found! Ready to run the prediction app.")
        return True

def main():
    print("=" * 60)
    print("CIFAR-10 Object Detection Training & Prediction System")
    print("=" * 60)
    
    # Check if notebook exists
    if not os.path.exists("10-object-detection.ipynb"):
        print("Notebook '10-object-detection.ipynb' not found!")
        return
    
    # Run the training
    if run_notebook():
        # Check if model files were created
        if check_model_files():
            print("\nReady to launch the prediction app!")
            print("Run: streamlit run app.py")
        else:
            print("\nTraining may not have completed successfully.")
            print("Please check the notebook output for errors.")
    else:
        print("\nTraining failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
