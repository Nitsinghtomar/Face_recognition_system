"""
Real-time Face Recognition via Webcam
=====================================

This module provides real-time facial recognition using a webcam feed.
It processes video frames in real-time and identifies faces based on
the trained classifier model.

Features:
    - Real-time video processing with optimized performance
    - Frame downscaling for faster processing
    - Confidence-based recognition filtering
    - Visual overlay with name labels and face boundaries

Performance Optimizations:
    1. Process frames at 1/4 resolution for speed
    2. Alternate frame processing to reduce computational load
    3. Efficient memory management for continuous operation

Controls:
    - Press 'q' to quit the application
"""

import face_recognition_api
import cv2
import os
import pickle
import numpy as np
import warnings

# Performance optimization settings
FRAME_SCALE_FACTOR = 0.25  # Process at 1/4 resolution
CONFIDENCE_THRESHOLD = 0.5  # Distance threshold for recognition

# Initialize webcam connection
webcam_capture = cv2.VideoCapture(0)

# Load the trained face recognition classifier
classifier_filename = 'classifier.pkl'
if os.path.isfile(classifier_filename):
    with open(classifier_filename, 'rb') as classifier_file:
        (label_encoder, face_classifier) = pickle.load(classifier_file)
    print("Face recognition classifier loaded successfully.")
else:
    print('\x1b[0;37;43m' + "ERROR: Classifier '{}' not found. Train the model first.".format(classifier_filename) + '\x1b[0m')
    quit()

# Initialize processing variables
detected_face_locations = []
face_encoding_vectors = []
identified_faces = []
process_current_frame = True


# Main video processing loop with warning suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    
    print("Starting real-time face recognition. Press 'q' to quit.")
    
    while True:
        # Capture frame from webcam
        frame_captured, current_frame = webcam_capture.read()
        
        if not frame_captured:
            print("Failed to capture frame from webcam.")
            break

        # Scale down frame for faster processing
        downscaled_frame = cv2.resize(current_frame, (0, 0), fx=FRAME_SCALE_FACTOR, fy=FRAME_SCALE_FACTOR)

        # Process every other frame to improve performance
        if process_current_frame:
            # Detect faces and generate encodings
            detected_face_locations = face_recognition_api.face_locations(downscaled_frame)
            face_encoding_vectors = face_recognition_api.face_encodings(downscaled_frame, detected_face_locations)

            identified_faces = []
            
            if len(face_encoding_vectors) > 0:
                # Calculate distances to nearest neighbors for confidence assessment
                nearest_neighbor_distances = face_classifier.kneighbors(face_encoding_vectors, n_neighbors=1)

                # Determine which faces meet confidence threshold
                recognition_confidence = [nearest_neighbor_distances[0][i][0] <= CONFIDENCE_THRESHOLD 
                                        for i in range(len(detected_face_locations))]

                # Generate predictions with confidence filtering
                identified_faces = []
                for prediction, location, is_confident in zip(
                    face_classifier.predict(face_encoding_vectors), 
                    detected_face_locations, 
                    recognition_confidence
                ):
                    if is_confident:
                        person_name = label_encoder.inverse_transform([int(prediction)])[0].title()
                    else:
                        person_name = "Unknown"
                    identified_faces.append((person_name, location))

        # Toggle frame processing flag
        process_current_frame = not process_current_frame

        # Display results on the video frame
        for person_name, (top, right, bottom, left) in identified_faces:
            # Scale coordinates back to original frame size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw bounding box around face
            cv2.rectangle(current_frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw name label below the face
            cv2.rectangle(current_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            text_font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(current_frame, person_name, (left + 6, bottom - 6), text_font, 1.0, (255, 255, 255), 1)

        # Display the processed frame
        cv2.imshow('Real-time Face Recognition', current_frame)

        # Check for quit command
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup resources
    webcam_capture.release()
    cv2.destroyAllWindows()
    print("Real-time face recognition stopped.")
