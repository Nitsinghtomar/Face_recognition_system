# Face Recognition System

A comprehensive machine learning solution for facial recognition and identification using advanced computer vision techniques. This system employs state-of-the-art deep learning models to achieve high-accuracy face detection and recognition, similar to modern social media platforms.

The implementation leverages [dlib's](http://dlib.net/) robust face recognition framework powered by deep neural networks, achieving exceptional performance with 99.38% accuracy on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark dataset.

## Tech Stack

### Core Technologies:
- **Python 3.x** - Primary programming language
- **dlib** - Deep learning-based face recognition library
- **OpenCV** - Computer vision library for image processing
- **NumPy** - Numerical computing library
- **SciPy** - Scientific computing library
- **Scikit-learn** - Machine learning algorithms
- **Pandas** - Data manipulation and analysis

### Machine Learning Components:
- **Face Detection**: HOG-based facial detection
- **Landmark Detection**: 68-point facial landmark identification
- **Face Encoding**: Deep neural network-based feature extraction
- **Classification**: Support Vector Machine (SVM) or K-Nearest Neighbors (KNN)

## System Requirements

### Essential Dependencies:
- Python 3.6 or higher
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Scikit-learn >= 0.23.0
- Pandas >= 1.1.0
- OpenCV >= 4.0.0
- dlib >= 19.0.0

### Optional Components:
- **Openface** (for `./demo-python-files/projecting_faces.py`)

Note: Installing dlib can be challenging. For macOS or Linux systems, refer to [this installation guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).

## Installation Guide:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nitsinghtomar/Face_recognition_system.git
   cd Face_recognition_system
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download required models:**
   - Download `shape_predictor_68_face_landmarks.dat` from [dlib models](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Download `dlib_face_recognition_resnet_model_v1.dat` from [dlib models](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)
   - Extract and place both files in the `models/` directory

## Project Structure

```
Face_recognition_system/
├── create_encodings.py          # Generate facial encodings from training images
├── train.py                     # Train the classification model
├── predict.py                   # Perform face recognition on test images
├── webcam.py                    # Real-time face recognition via webcam
├── face_recognition_api.py      # Core face recognition API module
├── setup.py                     # Automated project setup script
├── requirements.txt             # Python dependencies
├── models/                      # Pre-trained model files directory
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
└── demo-python-files/          # Additional demonstration scripts
    ├── find_face.py
    ├── find_face_landmarks.py
    └── projecting_faces.py
```

## File Descriptions

### Core Processing Files:
- **`create_encodings.py`**: Processes training images and generates numerical face encodings. Creates `encoded-images-data.csv` and `labels.pkl` files.
- **`train.py`**: Trains machine learning classifier using generated encodings. Supports both SVM and KNN algorithms.
- **`predict.py`**: Performs face recognition on test images using the trained model.
- **`webcam.py`**: Real-time face recognition using webcam input with optimized performance settings.
- **`face_recognition_api.py`**: Core API module providing face detection, landmark detection, and encoding functions.

### Utility Files:
- **`setup.py`**: Automated setup script for project initialization and dependency checking.
- **`requirements.txt`**: List of required Python packages with version specifications.

### Demo Files:
- **`find_face.py`**: Basic face detection demonstration.
- **`find_face_landmarks.py`**: Facial landmark detection example.
- **`projecting_faces.py`**: Advanced face projection techniques (requires Openface).

## How to Use

### Step 1: Model Training Process
1. **Prepare Training Data:**
   - Create a directory named `training-images`
   - Organize training data by creating individual folders for each person within `training-images`
   - Directory structure example:
     ```
     training-images/
     ├── person1/
     │   ├── image1.jpg
     │   ├── image2.jpg
     │   └── image3.jpg
     └── person2/
         ├── image1.jpg
         └── image2.jpg
     ```
   - **Important**: Each training image should contain exactly one face

2. **Generate Facial Encodings:**
   ```bash
   python create_encodings.py
   ```
   - This creates `encoded-images-data.csv` and `labels.pkl` files
   - Processes all images in training directories
   - Generates numerical representations of faces

3. **Train the Classifier:**
   ```bash
   python train.py
   ```
   - Creates `classifier.pkl` file containing the trained model
   - Automatically creates backup if existing classifier found
   - Supports both SVM and KNN algorithms

### Step 2: Recognition and Testing
1. **Prepare Test Images:**
   - Create a `test-images` directory
   - Add images containing faces you want to identify

2. **Run Face Recognition:**
   ```bash
   python predict.py
   ```
   - Analyzes each test image for faces
   - Displays confidence scores and identity predictions
   - Handles multiple faces per image

### Step 3: Real-time Recognition
```bash
python webcam.py
```
- Starts real-time face recognition using webcam
- Press 'q' to quit the application
- Optimized for performance with frame downscaling

## Complete Usage Workflow

### Quick Start:
```bash
# 1. Setup project
python setup.py

# 2. Create training data structure
mkdir training-images
mkdir training-images/person1
mkdir training-images/person2
# Add images to respective folders

# 3. Generate encodings
python create_encodings.py

# 4. Train the model
python train.py

# 5. Test on images
mkdir test-images
# Add test images
python predict.py

# 6. Real-time recognition
python webcam.py
```

## Configuration Options

### Performance Tuning:
- **`CONFIDENCE_THRESHOLD`** in `webcam.py`: Adjust recognition sensitivity (default: 0.5)
- **`FRAME_SCALE_FACTOR`** in `webcam.py`: Modify processing resolution for performance (default: 0.25)

### Algorithm Selection:
- **SVM Classifier**: Higher accuracy, slower training
- **KNN Classifier**: Faster training, good for smaller datasets

### Model Parameters:
- **Face Detection**: HOG-based detector with configurable sensitivity
- **Encoding**: 128-dimensional face descriptors using ResNet model
- **Distance Metric**: Euclidean distance for face comparison

## Technical Architecture

### Processing Pipeline:
1. **Image Loading**: Support for JPG, JPEG, PNG formats
2. **Face Detection**: Identify face regions in images
3. **Landmark Detection**: 68-point facial feature mapping
4. **Feature Extraction**: Generate 128-dimensional face encodings
5. **Classification**: SVM/KNN-based identity prediction
6. **Confidence Assessment**: Distance-based recognition confidence

### Output Files:
- **`encoded-images-data.csv`**: Numerical face encodings with labels
- **`labels.pkl`**: Label encoder for class mapping
- **`classifier.pkl`**: Trained classification model

## Troubleshooting

### Common Issues:
- **No face detected**: Ensure images have clear, front-facing faces
- **Low accuracy**: Add more training images per person (minimum 5-10 recommended)
- **Performance issues**: Reduce frame processing rate or image resolution
- **Import errors**: Verify all dependencies are installed correctly

### Requirements:
- **Minimum training images**: 3-5 per person
- **Recommended training images**: 10-20 per person for optimal accuracy
- **Image quality**: Clear, well-lit faces preferred
- **Face size**: At least 100x100 pixels recommended

## Acknowledgments
- Appreciation to the dlib library developers for providing robust facial recognition capabilities
- Recognition to the machine learning community for advancing computer vision research and development
