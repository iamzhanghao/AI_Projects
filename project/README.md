# Instructions
## Task 1. code that prepares the dataset 
 use the prepare() function in project.utils.py
 specify the split, size, and directory to split files and image files by changing the params

## Task 2. 
code for training - the code should report a validation set accuracy at least once every epoch. epoch means here: when you are one time through all your training images
 - Haonet.train is used in training the CNN weights
 - Haonet.validate in mode = 'val' is used to measure the accuracy of the prediction of training images

## Task 3. 
code for 
A. predicting on all the test set images of at least one split without a GUI
B. reporting the accuracy on the test set images. note: test set, not validation set. You use the validation set during training.
 - Haonet.validate in mode = 'test' is used to predict test images and measure its test accuracy
 

## Task 4. 
code for running a GUI that allows a user to select one image of the test set, runs a prediction on it
 - run GUI.py and select folder where test set is stored
 - once done, press run classification

# Note
Read report.pdf for details
