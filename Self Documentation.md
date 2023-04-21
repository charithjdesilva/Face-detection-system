Introduction to the System

First we have to detect the face. TO that we use HOG(Histogram of Oriented Gradients) method.

Then we draw a bounding box around the face.

Then we apply a Warp. Because it will be hard to identify otherwise if the face is tilted. To that it uses Dlib library at the backend. 
    They find the facial ladmarks first, and with that they try to generate the front face with them.

Once they have that image they send it to a neural network previously trained.

This NN model will give us the Encoded features.

So, we feed the image to this model and it will generate 128 different measurements for us.

Using these 128 measurements we can define a person. We can diffrentiate between differnt people as well.

To find the match we can use a Machine learning (classifier) method in here.

-----Steps I followed to make it.

First I had to download the C compiler. (Downloaded the Visual Studio 2022 community edtion -> from the installer -> Desktop development with C++)

Then follwing packages.
    opencv-python
    numpy
    cmake
    dlib
    face-recognition

-------Implementing the project

First we import packages cv2, numpy, face_recognition

then we load training image and convert it to RGB
then we load testing image and convert it to RGB

After that we find the face location of the Training image, face location returns a list of 4 values. They are Top, right,bottom, and left
Also, we encode the image with face_encoding()

After that we find the face location of the Testing image
Also, we encode the image with face_encoding()

Then we compare the encodings, we use Linear SVM to find out whether they match or not
results = face_recognition.compare_faces(known_face_encodings, face_encoding_to_check) we can give a list of face encoidngs

We will get a results either true or false. But we want to find out how similar these images are. Then we can find a best match. To do that we can find the distance 'face_distance'. The lower the distanc, the better the match is.

Then we display the value rounded up to 2 decimal places on the actual results image, alongside the name of the person.



