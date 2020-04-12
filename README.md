# Digit-recognition

In this project I have made a deep learning model to predict the handwritten number using keras.
In the number recongnize using keras file, I have used MNIST Handwritten images Datasets to train the model, after training the model 
we can save the model trained so that we dont need to train the model every time we run the program.
Then in project.py file using opencv i have built a GUI through which you can draw an number and save it by pressing 's' or 'S'
the function then reads the saved image and then returns to the main program. The image is then converted to required size for the model to predict the number. The image is then fed to the model and it returns the predicted number. The number is then written on 
the Predicted number.
You can see prediction.mp4 how it works.
