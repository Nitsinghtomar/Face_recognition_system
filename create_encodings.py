"""
Face Recognition Encoding Generator
==================================

This module generates facial encodings from training images for use in machine learning models.
It processes organized directories of face images and creates numerical representations
suitable for classification algorithms.

Functions:
    - Directory processing utilities for training data organization
    - Image filtering and validation
    - Facial encoding generation using dlib models
    - Dataset creation and persistence

Output Files:
    - encoded-images-data.csv: Numerical face encodings with labels
    - labels.pkl: Label encoder for class mapping
"""

import os
import face_recognition_api
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import pandas as pd


def get_training_directories(training_directory_path):
    """
    Retrieve all subdirectories within the training directory.
    
    Args:
        training_directory_path (str): Path to the training images directory
        
    Returns:
        list: List of subdirectory paths containing training images
    """
    return [directory_info[0] for directory_info in os.walk(training_directory_path)][1:]


def get_training_class_labels(training_directory_path):
    """
    Extract class labels from subdirectory names in the training directory.
    
    Args:
        training_directory_path (str): Path to the training images directory
        
    Returns:
        list: List of class label names (subdirectory names)
    """
    return [directory_info[1] for directory_info in os.walk(training_directory_path)][0]


def get_files_per_class(training_directory_path):
    """
    Get lists of files for each class directory.
    
    Args:
        training_directory_path (str): Path to the training images directory
        
    Returns:
        list: List of file lists for each class directory
    """
    return [directory_info[2] for directory_info in os.walk(training_directory_path)][1:]


def filter_image_files(training_directory_path):
    """
    Filter and validate image files from training directories.
    
    Args:
        training_directory_path (str): Path to the training images directory
        
    Returns:
        list: List of valid image files for each class directory
    """
    supported_extensions = [".jpg", ".jpeg", ".png"]

    filtered_files_per_class = []
    for file_list in get_files_per_class(training_directory_path):
        valid_images = []
        for filename in file_list:
            name, extension = os.path.splitext(filename)
            if extension.lower() in supported_extensions:
                valid_images.append(filename)
        filtered_files_per_class.append(valid_images)

    return filtered_files_per_class


def create_class_directory_mapping(training_directory_path, class_labels):
    """
    Create mapping between directories, labels, and image files.
    
    Args:
        training_directory_path (str): Path to the training images directory
        class_labels (list): List of class labels
        
    Returns:
        list: Zipped tuples of (directory_path, label, image_files)
    """
    return list(zip(get_training_directories(training_directory_path),
                    class_labels,
                    filter_image_files(training_directory_path)))


def create_encoding_dataset(training_directory_path, encoded_labels):
    """
    Generate facial encodings dataset from training images.
    
    Args:
        training_directory_path (str): Path to the training images directory
        encoded_labels (array): Numerically encoded class labels
        
    Returns:
        list: Dataset containing facial encodings with corresponding labels
    """
    encoding_dataset = []
    directory_mapping = create_class_directory_mapping(training_directory_path, encoded_labels)
    
    for directory_info in directory_mapping:
        directory_path, label, image_files = directory_info
        
        for image_filename in image_files:
            image_file_path = os.path.join(directory_path, image_filename)
            loaded_image = face_recognition_api.load_image_file(image_file_path)
            image_encodings = face_recognition_api.face_encodings(loaded_image)

            if len(image_encodings) > 1:
                print('\x1b[0;37;43m' + 'Multiple faces detected in {}. Using first face only.'.format(image_file_path) + '\x1b[0m')
            elif len(image_encodings) == 0:
                print('\x1b[0;37;41m' + 'No face detected in {}. Skipping file.'.format(image_file_path) + '\x1b[0m')
            else:
                print('Successfully encoded {}.'.format(image_file_path))
                encoding_dataset.append(np.append(image_encodings[0], label))
                
    return encoding_dataset

# Configuration paths and filenames
output_encodings_file = './encoded-images-data.csv'
training_images_directory = './training-images'
label_encoder_filename = "labels.pkl"

# Extract class labels from training directory structure
# Encode labels numerically for machine learning compatibility
class_labels = get_training_class_labels(training_images_directory)
label_encoder = LabelEncoder().fit(class_labels)
encoded_class_labels = label_encoder.transform(class_labels)
total_classes = len(label_encoder.classes_)

# Generate facial encoding dataset
facial_encodings_dataset = create_encoding_dataset(training_images_directory, encoded_class_labels)
encodings_dataframe = pd.DataFrame(facial_encodings_dataset)

# Create backup if output file already exists
if os.path.isfile(output_encodings_file):
    print("{} already exists. Creating backup.".format(output_encodings_file))
    os.rename(output_encodings_file, "{}.bak".format(output_encodings_file))

# Save encodings to CSV file
encodings_dataframe.to_csv(output_encodings_file)

print("{} classes processed successfully.".format(total_classes))
print('\x1b[6;30;42m' + "Saving label encoder to '{}'".format(label_encoder_filename) + '\x1b[0m')

# Persist label encoder for future use
with open(label_encoder_filename, 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)
    
print('\x1b[6;30;42m' + "Facial encodings saved to {}".format(output_encodings_file) + '\x1b[0m')
