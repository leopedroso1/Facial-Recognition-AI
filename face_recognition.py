# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 15:50:58 2020

@title: Face Recognition from images using face_recognition library
@author: Leonardo

"""

import face_recognition 
import os
import cv2


# WARNING: LINUX OR MACOS ONLY!!!

# 1. CREATE A FOLDER WITH KNOWN FACES
# 2. CREATE ANOTHER FOLDER WITH UNKNOWN FACES
# 3. SET THEM AT THE SAME DIRECTORY


KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces" 
TOLERANCE = 0.7
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn" #convolution neural network


known_faces = []
unknown_faces = []

print("Processing known faces")

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}/{filename}"):

        image = face_recognition.load_image_file()
        encoding = face_recognition.face_encodings(image)[0]

        known_faces.append(encoding)
        known_names.append(name)
        
print("Processing unknown faces")
        
for filename in os.listdir(UNKNOWN_FACES_DIR):
    
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for face_encoding, face_location in zip(encodings, locations):

        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None

        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found! {match}")
            
            # draw a rectangle where the face was detected
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            
            color = [0, 255, 0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2] + 22)

            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), FONT_THICKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(10000)
#   cv2.destroyWindow(filename)

                  