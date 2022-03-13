from djitellopy import tello
from time import sleep
import cv2
import KeyPressModule as kp
from time import sleep
import pygame
import math
import numpy as np

# Second : connect to the drone and turn the stream on, see what the drone sees but on the computer
me = tello.Tello()
me.connect()
print(me.get_battery())

# the lines below turns on the camera of the drone, where you'll be able to stream
me.streamon()

while True:
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)
