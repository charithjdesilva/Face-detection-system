# Face Recognition System Documentation

## Introduction

This project is a simple implementation of a face recognition system using Python and various libraries such as OpenCV, Numpy, Dlib, and face-recognition. The project utilizes the Histogram of Oriented Gradients (HOG) method to detect faces, followed by a Warp process to create a front-facing image. Afterward, the image is fed to a pre-trained neural network model that generates 128 measurements for each face. These measurements are then used to define a person and differentiate between different individuals. 

## System Workflow

1. Face Detection - The HOG method is used to detect faces in the image.

2. Face Alignment - The detected face is aligned using Dlib library to generate a front-facing image.

3. Face Encoding - The aligned image is fed into a pre-trained neural network model that generates 128 measurements.

4. Classification - A machine learning classifier (such as SVM) is used to differentiate between individuals by comparing the 128 measurements of each face.

## Implementation

### Requirements

The following libraries are required to run the project:

- OpenCV
- Numpy
- CMake
- Dlib
- face-recognition

### Workflow

1. Load the training and testing images, and convert them to RGB format.

2. Use face_recognition library to detect face locations in both the training and testing images.

3. Use face_recognition library to encode the faces detected in both the training and testing images.

4. Compare the encodings of the training and testing images using a machine learning classifier (e.g., SVM).

5. Calculate the distance between the encodings to determine the similarity between the two images.

6. Display the results on the testing image, including the similarity score and the name of the person.

## Conclusion

This project serves as a basic implementation of a face recognition system using Python and various libraries. By following the steps outlined in this documentation, you can create your own face recognition system to differentiate between individuals in images or videos.