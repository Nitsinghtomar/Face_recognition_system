# Face Recognition System

A comprehensive machine learning solution for facial recognition and identification using advanced computer vision techniques. This system employs state-of-the-art deep learning models to achieve high-accuracy face detection and recognition, similar to modern social media platforms.

The implementation leverages [dlib's](http://dlib.net/) robust face recognition framework powered by deep neural networks, achieving exceptional performance with 99.38% accuracy on the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) benchmark dataset.

## System Requirements:

- Python 3.x
- NumPy
- SciPy
- [Scikit-learn](http://scikit-learn.org/stable/install.html)
- [dlib](http://dlib.net/)

    Note: Installing dlib can be challenging. For macOS or Linux systems, you may refer to [this installation guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).

- Additional Components:

    - OpenCV (required for `webcam.py` to capture real-time video frames)

    - For utilizing `./demo-python-files/projecting_faces.py`, you'll need [Openface](https://cmusatyalab.github.io/openface/setup/) installed.

        Openface installation instructions:
        ```bash
            $ git clone https://github.com/cmusatyalab/openface.git
            $ cd openface
            $ pip install -r requirements.txt
            $ sudo python setup.py install
        ```

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

## Demo Output:
<img src='https://user-images.githubusercontent.com/17249362/28241776-a45a5eb0-69b8-11e7-9024-2a7a776914e6.gif' width='700px'>

## Project Structure:

```
Face_recognition_system/
├── create_encodings.py          # Generate facial encodings from training images
├── train.py                     # Train the classification model
├── predict.py                   # Perform face recognition on test images
├── webcam.py                    # Real-time face recognition via webcam
├── face_recognition_api.py      # Core face recognition API module
├── requirements.txt             # Python dependencies
├── models/                      # Pre-trained model files directory
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
└── demo-python-files/          # Additional demonstration scripts
    ├── find_face.py
    ├── find_face_landmarks.py
    └── projecting_faces.py
```

## Implementation Guide:

### Model Training Process:
- Create a directory named `training-images`.
- Organize training data by creating individual folders for each person within `training-images`.

    Directory Structure Example:
    ```bash
    $ mkdir training-images
    $ cd training-images
    $ mkdir Person_Name
    ```
    Place all images of the specific person in the `./training-images/Person_Name` directory.

    <img src='https://user-images.githubusercontent.com/17249362/28241803-2b6db474-69b9-11e7-9a70-43fd3e9b30a7.png' width='300px'>

- Execute `python create_encodings.py` to generate facial encodings and corresponding labels.
    This process creates `encoded-images-data.csv` and `labels.pkl` files containing the processed data.

    <img src='https://user-images.githubusercontent.com/17249362/28241799-1a848d7c-69b9-11e7-8572-dbac69631085.png' width='700px'>

    Important: Ensure each training image contains exactly one face, as the system will encode only the first detected face.

- Execute `python train.py` to train and persist the facial recognition classifier.
    This generates a `classifier.pkl` file containing the trained model.
    A backup file `classifier.pkl.bak` is automatically created if an existing classifier is found.

    <img src='https://user-images.githubusercontent.com/17249362/28241802-2894f456-69b9-11e7-91e8-341115fba605.png' width='700px'>

### Recognition and Testing:
- Create a `test-images` directory containing images for facial recognition testing.

    <img src='https://user-images.githubusercontent.com/17249362/28241801-25db4814-69b9-11e7-9c8e-c19f3e09499a.png' width='300px'>

- Execute `python predict.py` to perform facial recognition on test images.

    <img src='https://user-images.githubusercontent.com/17249362/28241800-21ecf69e-69b9-11e7-8564-6d9dcb067225.png' width='700px'>

### Real-time Recognition:
- Execute `python webcam.py` to start real-time face recognition using your webcam.
- Press 'q' to quit the webcam application.

## Usage Examples:

### Basic Workflow:
```bash
# Step 1: Create training data structure
mkdir training-images
mkdir training-images/person1
mkdir training-images/person2
# Add images to respective folders

# Step 2: Generate encodings
python create_encodings.py

# Step 3: Train the model
python train.py

# Step 4: Test on images
mkdir test-images
# Add test images
python predict.py

# Step 5: Real-time recognition
python webcam.py
```

### Advanced Configuration:
- Modify `CONFIDENCE_THRESHOLD` in `webcam.py` to adjust recognition sensitivity
- Switch between SVM and KNN classifiers in `train.py`
- Adjust performance settings in `webcam.py` for different hardware capabilities


## Acknowledgments
- Appreciation to the dlib library developers for providing robust facial recognition capabilities.
- Recognition to the machine learning community for advancing computer vision research and development.