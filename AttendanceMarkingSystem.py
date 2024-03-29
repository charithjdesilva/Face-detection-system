import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'People'
images = []

# create a list of names from a images inside the folder
imageNames = []

# first we will grab the list of images inside the image folder
imagesList = os.listdir(path)
print(imagesList)

# use the names of imagesList to import images one by one
for imgName in imagesList:
    currentImg = cv2.imread(f'{path}/{imgName}')    # read image and save it to currentImg
    images.append(currentImg)    # append current image to images list
    imageNames.append(os.path.splitext(imgName)[0])    # also append imageNames, we only need the name not with file extension
print(imageNames)

# find the encodings of the images
def findEncodings(images):
    encodeList = []    # a list to save all the encoded images

    # loop through all the images
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB
        encodeOfImg = face_recognition.face_encodings(img)[0]    # find encoding of the image
        encodeList.append(encodeOfImg)  # append the encode value to the encodeList
    return encodeList

# Function to mark the attendance
def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:     # open file for read and write at the same Time
        prevAttendedList = f.readlines()    # if someone has already arrived we dont want to repeat it
        nameList = []
        for line in prevAttendedList:
            entry = line.split(',')     # to seperate name and time
            nameList.append(entry[0])   # will get us the all name list

        if name not in nameList:
            now = datetime.now()
            dateTimeString = now.strftime('%H:%M%S')
            f.writelines(f'\n{name},{dateTimeString}')      # write name, date time in a new line

markAttendance('a')

# run encoding function
encodeListForKnownFaces = findEncodings(images)
print('Encoding completed!')

# find the image match between encodings
cap = cv2.VideoCapture(0)

while True:     # to get each frame one by one
    success, img = cap.read()

    # we are reducing the image size because it will help us by speeding the process
    imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)     # small image will be 1/4 th of the size
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)    # convert to RGB

    # in webcam we may find multiple faces, so we are first find the location of the faces
    facesLocationsInCurrFrame = face_recognition.face_locations(imgSmall)  # capture all the face location in the current frame

    # then we send these location to the encoding function
    encodeFacesInCurrFrame =  face_recognition.face_encodings(imgSmall, facesLocationsInCurrFrame)

    # find matches
    # iterate through all the faces that we have found in the current frame
    for encodeFace, faceLocation in zip(encodeFacesInCurrFrame, facesLocationsInCurrFrame):
        matches = face_recognition.compare_faces(encodeListForKnownFaces, encodeFace)   # compare with all the encoding that we found before
        faceDistance = face_recognition.face_distance(encodeListForKnownFaces, encodeFace)    # find the distance, this will give us distance according to the all the knownFaces
        print(faceDistance)

        # we need to get the best match which is the lowest distance in faceDistance List
        matchIndex = np.argmin(faceDistance)    # will have the index of the best matched face

        if matches[matchIndex]:
            name = imageNames[matchIndex]
            print(name)

            # display a bounding box around that person and display their name
            y1, x2, y2, x1 = faceLocation    # we have the location of the face in the imageSmall, we destruct the list
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4    # location of the face in the original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255,255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (255,255,0), 2)  # rectangle to display name
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            markAttendance(name)

    cv2.imshow('webcam', img)   # show the original image
    cv2.waitKey(1)
