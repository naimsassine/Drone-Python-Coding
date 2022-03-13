from djitellopy import tello
from time import sleep
import cv2
import KeyPressModule as kp
from time import sleep
import pygame
import math
import numpy as np

# First section : just connect to the drone, make it fly then land using only code

# connect to the drone, don't forget to connect via wifi to the drone iself
# from your computer
me = tello.Tello()
me.connect()

print(me.get_battery())

# take coff and controls
me.takeoff()
me.send_rc_control(0, 50, 0, 0)

sleep(2)

me.send_rc_control(0, 0, 0, 30)

sleep(2)

me.send_rc_control(0, 0, 0, 0)

# land
me.land()
