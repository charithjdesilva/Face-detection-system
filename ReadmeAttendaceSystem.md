# Attendance Marking System with Face Recognition

This project is a facial recognition-based attendance marking system that uses OpenCV, NumPy, and the face_recognition library to identify and mark the attendance of known individuals.

## Usage

### Requirements

- Python 3.x
- OpenCV
- NumPy
- face_recognition

### Installation

You can install OpenCV, NumPy, and face_recognition using pip. To install, simply run the following commands in your terminal:

```python
pip install opencv-python
pip install numpy
pip install face-recognition
```

### How to Run

1. Clone the repository to your local machine.
2. Place the images of the known individuals in the "People" folder.
3. Run the program using the following command:

```python
python attendance.py
```

4. The program will open your webcam and start recognizing the individuals in the frame.
5. The program will mark the attendance of the recognized individuals in the "Attendance.csv" file.

### How it Works

1. The program reads the images of the known individuals from the "People" folder.
2. It generates the encodings for the known individuals using the face_recognition library.
3. It captures the video stream from the webcam and processes each frame.
4. It identifies the faces in the frame using the face_recognition library.
5. It compares the encodings of the recognized faces with the encodings of the known individuals to find the best match.
6. It marks the attendance of the recognized individuals in the "Attendance.csv" file.

## Conclusion

This project demonstrates how to use facial recognition technology to create an attendance marking system. By combining OpenCV, NumPy, and the face_recognition library, we can easily create a system that can recognize known individuals and mark their attendance.