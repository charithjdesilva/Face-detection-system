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