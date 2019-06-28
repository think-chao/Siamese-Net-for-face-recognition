#!/usr/bin/env python  
# -*- coding:utf-8 _*-  
""" 
@author:wchao118
@license: Apache Licence 
@file: read_camera.py 
@time: 2019/06/26
@contact: wchao118@gmail.com
@software: PyCharm 
"""

import cv2
from main import api
import os
from PIL import Image
import numpy as np

save_path = r'E:\file\data\face\wangchao'
detector = cv2.CascadeClassifier(r'E:\conda\pkgs\libopencv-3.4.2-h20b85fd_0\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
print('ok')
id = 0
while True:
    id += 1
    ret, img = cap.read()
    if id % 10 == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            crop_face = img[x:x+w, y:y+h]
            # api(Image.fromarray(crop_face))
            cv2.imwrite(os.path.join(save_path, str(id) + '.jpg'), crop_face)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.imshow('test', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break