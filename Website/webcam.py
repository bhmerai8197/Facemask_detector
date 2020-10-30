import cv2# defining face detector
from imutils import paths
import face_detection
import cv2
import numpy as np
from tensorflow.keras.models import Model, load_model

mask_classifier = load_model("mask_classifier_v1")
type_classifier = load_model("mask_type_model_v1")

face_clsfr=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_type={0:'fabric',1:'n95',2:'surgical'}
labels_dict={0:'without_mask',1:'mask'}
color_dict={0:(0,0,255),1:(0,255,0),2:(255,0,0)}

ds_factor=1
size = 0.6
class Camera(object):
    def __init__(self):
       #capturing video
       self.webcam = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.webcam.release()
    
    def get_frame(self):
        #extracting frames
        (rval, im) = self.webcam.read()
        #ret, frame
        im=cv2.flip(im,1,1)
        #mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
        
        
        frame=cv2.resize(im,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)   
       
        faces = face_clsfr.detectMultiScale(frame)
        
        #for (x,y,w,h) in face_rects:
        for f in faces:
            (x, y, w, h) = [v * ds_factor for v in f]
            face_img = im[y:y+h, x:x+w]
            resized=cv2.resize(face_img,(224,224))
            normalized=resized/255.0
            reshaped=np.reshape(normalized,(1,224,224,3))
            reshaped = np.vstack([reshaped])
            result=mask_classifier.predict(reshaped)
            
            if (np.argmax(result,axis=1)[0] == 0):
               
                label=np.argmax(result,axis=1)[0]
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
            elif (np.argmax(result,axis=1)[0] == 1):
            
                type_result = type_classifier.predict(reshaped)
                label=np.argmax(type_result,axis=1)[0]
                cv2.rectangle(im,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(im,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(im, labels_type[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
               
   
            break   
              
        ret, jpeg = cv2.imencode('.jpg', im)
        return jpeg.tobytes()
    
   
    




