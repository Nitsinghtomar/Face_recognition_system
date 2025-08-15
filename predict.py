"""
Face Recognition Prediction Module
=================================

This module performs facial recognition on test images using a pre-trained classifier.
It identifies faces in images and predicts the identity of recognized individuals
based on the training data.

Features:
    - Batch processing of test images
    - Confidence-based recognition filtering
    - Support for multiple faces per image
    - Detailed prediction results with locations

Requirements:
    - Trained classifier (classifier.pkl)
    - Test images in supported formats (jpg, jpeg, png)
"""

import os
import face_recognition_api
import pickle
import numpy as np
import pandas as pd


def get_test_image_paths(test_directory):
    """
    Retrieve paths of all valid image files in the test directory.
    
    Args:
        test_directory (str): Path to directory containing test images
        
    Returns:
        list: List of full paths to valid image files
    """
    directory_contents = [walk_result[2] for walk_result in os.walk(test_directory)][0]
    valid_image_paths = []
    supported_extensions = [".jpg", ".jpeg", ".png"]
    
    for filename in directory_contents:
        _, file_extension = os.path.splitext(filename)
        if file_extension.lower() in supported_extensions:
            valid_image_paths.append(os.path.join(test_directory, filename))

    return valid_image_paths


# Configuration
classifier_filename = 'classifier.pkl'
test_images_directory = './test-images'

# Load encoding data for reference (optional, used for additional validation)
encodings_data_file = './encoded-images-data.csv'
encodings_dataframe = pd.read_csv(encodings_data_file)
reference_data = np.array(encodings_dataframe.astype(float).values.tolist())

# Extract reference features and labels (for potential distance calculations)
reference_features = np.array(reference_data[:, 1:-1])
reference_labels = np.array(reference_data[:, -1:])

# Load trained classifier and label encoder
if os.path.isfile(classifier_filename):
    with open(classifier_filename, 'rb') as classifier_file:
        (label_encoder, trained_classifier) = pickle.load(classifier_file)
    print("Classifier loaded successfully.")
else:
    print('\x1b[0;37;43m' + "ERROR: Classifier '{}' not found. Train the model first.".format(classifier_filename) + '\x1b[0m')
    quit()

# Process each test image
test_image_paths = get_test_image_paths(test_images_directory)

if not test_image_paths:
    print('\x1b[0;37;43m' + "No valid images found in '{}'".format(test_images_directory) + '\x1b[0m')
    quit()

for image_path in test_image_paths:
    print('\x1b[6;30;42m' + "===== Analyzing faces in '{}' =====".format(image_path) + '\x1b[0m')

    # Load and process the image
    test_image = face_recognition_api.load_image_file(image_path)
    detected_face_locations = face_recognition_api.face_locations(test_image)

    # Generate encodings for detected faces
    face_encodings = face_recognition_api.face_encodings(test_image, known_face_locations=detected_face_locations)
    print("Detected {} face(s) in the image".format(len(face_encodings)))

    if len(face_encodings) == 0:
        print("No faces detected in this image.")
        continue

    # Calculate distances to nearest neighbors for confidence assessment
    nearest_distances = trained_classifier.kneighbors(face_encodings, n_neighbors=1)

    # Determine recognition confidence (threshold: 0.5)
    recognition_confidence = [nearest_distances[0][i][0] <= 0.5 for i in range(len(detected_face_locations))]

    # Generate predictions with confidence filtering
    face_predictions = []
    for prediction, location, is_confident in zip(
        trained_classifier.predict(face_encodings), 
        detected_face_locations, 
        recognition_confidence
    ):
        if is_confident:
            person_name = label_encoder.inverse_transform([int(prediction)])[0].title()
        else:
            person_name = "Unknown"
        face_predictions.append((person_name, location))

    # Display results
    for i, (name, location) in enumerate(face_predictions):
        top, right, bottom, left = location
        confidence_score = 1 - nearest_distances[0][i][0]  # Convert distance to confidence
        print("Face {}: {} (Confidence: {:.2f})".format(i+1, name, confidence_score))
        print("  Location: Top={}, Right={}, Bottom={}, Left={}".format(top, right, bottom, left))

    print("Prediction results:", face_predictions)
    print()
