#!/usr/bin/env python3
"""
Face Recognition System Setup Script
===================================

This script helps set up the Face Recognition System by creating
necessary directories and checking dependencies.
"""

import os
import sys
import subprocess

def create_directory(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✓ Created directory: {path}")
    else:
        print(f"✓ Directory already exists: {path}")

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 6):
        print("❌ Python 3.6 or higher is required.")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_requirements():
    """Install required packages."""
    try:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Successfully installed requirements")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements")
        return False

def check_model_files():
    """Check if required model files exist."""
    model_files = [
        "models/shape_predictor_68_face_landmarks.dat",
        "models/dlib_face_recognition_resnet_model_v1.dat"
    ]
    
    missing_files = []
    for file_path in model_files:
        if os.path.exists(file_path):
            print(f"✓ Model file found: {file_path}")
        else:
            print(f"❌ Model file missing: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print("\nPlease download the missing model files:")
        print("1. shape_predictor_68_face_landmarks.dat from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print("2. dlib_face_recognition_resnet_model_v1.dat from: http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2")
        print("Extract and place them in the 'models/' directory.")
        return False
    
    return True

def main():
    """Main setup function."""
    print("Face Recognition System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Create necessary directories
    directories = [
        "training-images",
        "test-images",
        "models"
    ]
    
    for directory in directories:
        create_directory(directory)
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        install_requirements()
    else:
        print("❌ requirements.txt not found")
    
    # Check model files
    check_model_files()
    
    print("\n" + "=" * 40)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Add training images to 'training-images/' directory")
    print("2. Run 'python create_encodings.py' to generate encodings")
    print("3. Run 'python train.py' to train the model")
    print("4. Add test images to 'test-images/' directory")
    print("5. Run 'python predict.py' to test recognition")

if __name__ == "__main__":
    main()
