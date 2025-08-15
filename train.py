"""
Face Recognition Classifier Training Module
==========================================

This module trains machine learning classifiers for facial recognition using
preprocessed facial encodings. It supports multiple classification algorithms
and provides model persistence for deployment.

Key Features:
    - Data loading and validation
    - Multiple classifier options (SVM, KNN)
    - Model persistence and backup management
    - Comprehensive error handling

Output:
    - classifier.pkl: Trained model with label encoder
"""

import os
from random import shuffle
from sklearn import svm, neighbors
import pickle
import numpy as np
import pandas as pd

# File path configurations
encodings_data_file = './encoded-images-data.csv'
label_encoder_file = 'labels.pkl'

# Validate required input files exist
if os.path.isfile(encodings_data_file):
    encodings_dataframe = pd.read_csv(encodings_data_file)
else:
    print('\x1b[0;37;41m' + 'ERROR: {} not found. Run create_encodings.py first.'.format(encodings_data_file) + '\x1b[0m')
    quit()

if os.path.isfile(label_encoder_file):
    with open(label_encoder_file, 'rb') as file:
        label_encoder = pickle.load(file)
else:
    print('\x1b[0;37;41m' + 'ERROR: {} not found. Run create_encodings.py first.'.format(label_encoder_file) + '\x1b[0m')
    quit()

# Prepare training data
# Convert dataframe to numpy array and shuffle for better training distribution
training_data = np.array(encodings_dataframe.astype(float).values.tolist())
shuffle(training_data)

# Extract features and labels from the dataset
# Remove DataFrame index column (column 0) and separate features from labels
feature_vectors = np.array(training_data[:, 1:-1])  # All columns except first and last
class_labels = np.array(training_data[:, -1:])      # Last column contains labels

# Initialize and train the classifier
# Options: SVM or K-Nearest Neighbors
# classifier = svm.SVC(C=1, kernel='linear', probability=True)  # SVM option
classifier = neighbors.KNeighborsClassifier(
    n_neighbors=3, 
    algorithm='ball_tree', 
    weights='distance'
)

print("Training classifier with {} samples...".format(len(feature_vectors)))
classifier.fit(feature_vectors, class_labels.ravel())

# Model persistence with backup functionality
classifier_filename = "./classifier.pkl"

# Create backup if classifier already exists
if os.path.isfile(classifier_filename):
    print('\x1b[0;37;43m' + "Existing classifier found. Creating backup.".format(classifier_filename) + '\x1b[0m')
    os.rename(classifier_filename, "{}.bak".format(classifier_filename))

# Save the trained classifier along with label encoder
with open(classifier_filename, 'wb') as classifier_file:
    pickle.dump((label_encoder, classifier), classifier_file)

print('\x1b[6;30;42m' + "Classifier successfully saved to '{}'".format(classifier_filename) + '\x1b[0m')

