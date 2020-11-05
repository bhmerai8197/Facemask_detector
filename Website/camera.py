import cv2# defining face detector
from imutils import paths
import face_detection
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model

mask_classifier = load_model("mask_classifier_v1")
type_classifier = load_model("mask_type_model_v1")

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

<<<<<<< HEAD:Website/camera.py
=======

labels_dict={0:'without_mask',1:'fabric',2:'n95',3:'surgical'}
>>>>>>> 31411c4afff194cc40c423192742092cba8df5ed:Website/webcam.py

labels_dict={0:'without_mask',1:'fabric',2:'n95',3:'surgical'}
color_dict={0:(0,0,255),1:(0,0,0),2:(255,255,255),3:(255,0,0)}
classifier_dict={-1:'no_person',0:'without_mask',1:'fabric',2:'n95',3:'surgical'}

ds_factor=1
size = 0.6

class Camera(object):
    def __init__(self):
       #capturing video
       self.webcam = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.webcam.release()
    
    def closest_face_finder(faces):
        closest_face =[0,0,0,0]
        face_size = 0
        found_flag = 0
        if(len(faces)>0):
            for face in faces:
                (x, y, w, h) = [v for v in face] #Scale the shapesize backup
                size = w*h
                if(size > face_size):
                    closest_face = face                
                    face_size = size
                    #print(closest_face)
                    found_flag = 1
        return closest_face, found_flag, face_size
    
    def get_frame(self,prediction_history):
       
        # Number of frames to capture
        num_frames = 1;
        if(len(prediction_history)>20):
            prediction_history = prediction_history[-9:]
            
        #extracting frames
        (rval, im) = self.webcam.read()
        #ret, frame
        im=cv2.flip(im,1,1)
        #mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        # Draw rectangles around each face
       
        
        frame=cv2.resize(im,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)   
       
        faces = face_clsfr.detectMultiScale(frame)
        
        closest_face, flag, face_size = Camera.closest_face_finder(faces)
        
        if flag == 0: 
            classifier_output = -1
            prediction_history.append(classifier_output)
            # Draw rectangles around each face
        if flag == 1 and face_size >= 60000:
            (x, y, w, h) = [v * ds_factor for v in closest_face]
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(224,224))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,224,224,3))
            reshaped = np.vstack([reshaped])
            result=mask_classifier.predict(reshaped)
            
            if (np.argmax(result,axis=1)[0] == 0):
                classifier_output = 0
                prediction_history.append(classifier_output)
                label = 0
                prediction_history.append(classifier_output)
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            elif (np.argmax(result,axis=1)[0] == 1):
               
                
                type_result = type_classifier.predict(reshaped)
                label=np.argmax(type_result,axis=1)[0]
                classifier_output = label + 1
                prediction_history.append(classifier_output)
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label + 1],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label + 1],-1)
                cv2.putText(im, labels_dict[label +1], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
        _, index, count, = np.unique(prediction_history[-9:], return_index= True, return_counts = True)
        mask_flag = prediction_history[index[np.argmax(count)]]
        ret, jpeg = cv2.imencode('.jpg', im)  # encode OpenCV raw frame to jpg and displaying it      
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes(),mask_flag,prediction_history
    
<<<<<<< HEAD:Website/camera.py
 
=======
>>>>>>> 31411c4afff194cc40c423192742092cba8df5ed:Website/webcam.py
