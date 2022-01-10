import cv2
from deepface import DeepFace
import numpy as np  #this will be used later in the process


imgpath = "/home/paco/Pictures/20220107142647.jpg"

#analyze = DeepFace.analyze(imgpath,actions=['emotion', 'age', 'gender', 'race'],models={}, enforce_detection=True)
analyze = DeepFace.analyze(imgpath, actions=['emotion', 'age', 'gender', 'race'], models={}, enforce_detection=False)

print(analyze)
