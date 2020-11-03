# Facemask_detector
The aim of the project is to build a system with a user interface including multi-class CNN that can detect the type of mask worn by the visitor(s) using a dedicated video camera, and control the door access over the internet to provide access to the visitor(s) if mask conditions are satisfied.

Project was developed in following steps:

- Environment Setup
- Creating Artificial Images
- Preparing Training Folders
- Training Mask – No Mask Classifier
- Training Mask Type Classifier
- Setting up and Running Website Interface

## 1. Setting up Correct Environment:

The project has been developed on the current versions of deep learning libraries (especially with Tensorflow). Errors might occur if a different tensorflow version is used. In addition to most popular and common python packages, the following packages along with their required version are as follows:
- Tensorflow >= 2.3.0 - Pyserial
- Flask
- CV2
- Face-detection - Keras
- sqlalchemy
- Sklearn
- matplot - pandas - numpy -os
- haarcascade_frontalface_default.xml model file (available at GitHub repository)

## 2. Creating Artificial Images:
Real Images are present in the real folder with further sub-folders for images belonging to each of the 4 classifications: no_mask, fabric, n95, and surgical. There is an additional folder no_mask_unused which contains images without masks that are not used for creating artificial images (to maintain differences between training data).
The code artificial_image_generator.ipynb is used to create artificial images. Please ensure a folder called artificial is present in the current directory with 3 sub-folders: fabric, n95, and surgical is present. The generated artificial images will be stored in this folder. Please note: the GitHub repository already contains these folders loaded with artificial images generated.

## 3. Preparing Training Folders:
The images in the real and artificial folders are combined together into a folder called combined with a similar 4 sub-folder structure. The not used without mask real images are put into the no_mask folder at this point.
For each classifier, a different folder is created: dataset_mask_no_mask for mask – no mask classifier and dataset_mask_type for mask type classifier. This is a manual process that requires the following to be done:
  1. For Mask – No Mask Classifier:
30
- Copy the no_mask real image folder within the dataset_mask_no_mask folder
- Create a folder called mask inside dataset_mask_no_mask folder
- Copy random images (300-350 each) from the fabric, n95, and surgical folders into this mask folder - Make sure to include all real images in the above step to increase variation in data
Use the Train_data_generator_mask_no_mask.ipynb code to crop the images in this folder and save them in a similar structure folder by the name train_mask_no_mask folder (ensure proper folder structure already exists for saving). This will now act as the final dataset for training and testing the mask – no mask model
2. For Mask Type Classifier:
- Copy all except no_mask folder within the dataset_mask_type folder
- Ensure you have 3 different folders inside dataset_mask_type folder corresponding to each mask type
Use the Train_data_generator_mask_type .ipynb code to crop the images in this folder and save them in a similar structure folder by the name train_mask_type folder (ensure proper folder structure already exists for saving). This will now act as the final dataset for training and testing the mask type model
The folder structure for the above has already been placed and uploaded with images to allow for easier execution of the project if required. Additionally, any noise or incorrect images are deleted at this point by visual inspection.

## 4. Training Mask – No Mask Classifier
The code mask_no_mask_classifier.ipynb is used to extract data from dataset_mask_no_mask folder and stored in a numpy array along with sub-folder names as data label. This code trains the CNN as per the user defined parameters of epoch, learning rate, and batch-size. The generated model is stored under the name mask_classifier_v1

## 5. Training Mask Type Classifier
The code mask_type_classifier.ipynb is used to extract data from dataset_mask_type folder and stored in a NumPy array along with sub-folder names as data label. This code trains the CNN as per the user defined parameters of epoch, learning rate, and batch-size. The generated model is stored under the name mask_type_classifier_v1

## 6. The code Interface
Interface.ipynb is used to setup and run the web interface. Please ensure camera.py and the classifier models are all in the same directory. This code sets up the communication with Arduino and takes the live camera feed to run the entire system. Note: this code requires a connected Arduino.
If an Arduino is not available, Interface_without_Arduino.ipynb can be used to run and test the system. The system will lack any ability to update the visitor count in this mode, but still presents the framework of the website and the predictions made by the system.
