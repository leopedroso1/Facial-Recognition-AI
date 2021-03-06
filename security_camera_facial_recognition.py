# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:38:50 2020
@title: Face Recognition from video using face_recognition library
@author: Leonardo
"""

# -*- coding: utf-8 -*-

import face_recognition 
import os
import cv2
import pickle
import time

# WARNING: LINUX OR MACOS ONLY!!! For more information, please check the following link: https://pypi.org/project/face-recognition/

# Main algorithm steps
# 1. CREATE A FOLDER WITH KNOWN FACES
# 2. CREATE ANOTHER FOLDER WITH UNKNOWN FACES
# 3. SET THEM AT THE SAME DIRECTORY


KNOWN_FACES_DIR = "known_faces"
#UNKNOWN_FACES_DIR = "unknown_faces"  >> Now the unkwon will be our video
TOLERANCE = 0.7
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #convolution neural network


video = cv2.VideoCapture("facerecvideo.mp4") # Tip: You can use your webcam (2) or other external video source in a final project


print("Processing known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}/{filename}"):

        encoding = pickle.load(open(f"{name}/{filename}", "rb"))
        known_faces.append(encoding)
        known_names.append(int(name))
    
if len(known_names) > 0:
    next_id = max(known_names) + 1
else:
    next_id = 0


        
print("Processing unknown faces")
        
while True:

    ret, image = video.read()
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    
#    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found! {match}")
        
        else:
            match = str(next_id)
            next_id += 1
            known_names.append(match)
            known_faces.append(face_encoding)
            os.mkdir(f"{KNOWN_FACES_DIR}/{match}")
            pickle.dump(face_encoding, open(f"{KNOWN_FACES_DIR}/{match}/{match}.{int(time.time())}.pkl", "wb"))
            
        # draw a rectangle where the face was detected
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])
        
        color = [0, 255, 0]
        cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
        
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)

        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow("", image)

    if cv2.waitKey(1) or 0xFF == ord("q"):
        break

#   cv2.waitKey(10000)
#   cv2.destroyWindow(filename)

                  