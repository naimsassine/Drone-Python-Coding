# importing all the needed libraries
from djitellopy import tello
from time import sleep
import cv2
import KeyPressModule as kp
from time import sleep
import pygame
import math
import numpy as np

# First section

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


# Second section

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


# Third section

kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50
    if kp.getKey("LEFT"):
        lr = -speed
    elif kp.getKey("RIGHT"):
        lr = speed
    if kp.getKey("UP"):
        fb = speed
    elif kp.getKey("DOWN"):
        fb = -speed
    if kp.getKey("w"):
        ud = speed
    elif kp.getKey("s"):
        ud = -speed
    if kp.getKey("a"):
        yv = -speed
    elif kp.getKey("d"):
        yv = speed
    if kp.getKey("q"):
        me.land()
        sleep(3)
    if kp.getKey("e"):
        me.takeoff()
    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    sleep(0.05)


def init():
    pygame.init()
    win = pygame.display.set_mode((400, 400))


def getKey(keyName):
    ans = False
    for eve in pygame.event.get():
        pass
    keyInput = pygame.key.get_pressed()
    myKey = getattr(pygame, "K_{}".format(keyName))
    print("K_{}".format(keyName))

    if keyInput[myKey]:
        ans = True
    pygame.display.update()
    return ans


def main():
    if getKey("LEFT"):
        print("Left key pressed")

    if getKey("RIGHT"):
        print("Right key Pressed")


if __name__ == "__main__":
    init()
    while True:
        main()


# Surveillance section


kp.init()
me = tello.Tello()
me.connect()
print(me.get_battery())

global img

me.streamon()


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    if kp.getKey("LEFT"):
        lr = -speed

    elif kp.getKey("RIGHT"):
        lr = speed

    if kp.getKey("UP"):
        fb = speed

    elif kp.getKey("DOWN"):
        fb = -speed

    if kp.getKey("w"):
        ud = speed

    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -speed

    elif kp.getKey("d"):
        yv = speed

    if kp.getKey("q"):
        me.land()
        time.sleep(3)

    if kp.getKey("e"):
        me.takeoff()

    if kp.getKey("z"):
        cv2.imwrite(f"Resources/Images/{time.time()}.jpg", img)

        time.sleep(0.3)

    return [lr, fb, ud, yv]


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = me.get_frame_read().frame
    img = cv2.resize(img, (360, 240))
    cv2.imshow("Image", img)
    cv2.waitKey(1)


# Mapping section


######## PARAMETERS ###########

fSpeed = 117 / 10  # Forward Speed in cm/s   (15cm/s)
aSpeed = 360 / 10  # Angular Speed Degrees/s  (50d/s)
interval = 0.25
dInterval = fSpeed * interval
aInterval = aSpeed * interval

###############################################

x, y = 500, 500

a = 0

yaw = 0

kp.init()
me = tello.Tello()
me.connect()

print(me.get_battery())

points = [(0, 0), (0, 0)]


def getKeyboardInput():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 15
    aspeed = 50
    global x, y, yaw, a
    d = 0

    if kp.getKey("LEFT"):
        lr = -speed
        d = dInterval
        a = -180

    elif kp.getKey("RIGHT"):
        lr = speed
        d = -dInterval
        a = 180

    if kp.getKey("UP"):
        fb = speed
        d = dInterval
        a = 270

    elif kp.getKey("DOWN"):
        fb = -speed
        d = -dInterval
        a = -90

    if kp.getKey("w"):
        ud = speed

    elif kp.getKey("s"):
        ud = -speed

    if kp.getKey("a"):
        yv = -aspeed
        yaw -= aInterval

    elif kp.getKey("d"):
        yv = aspeed
        yaw += aInterval

    if kp.getKey("q"):
        me.land()
        sleep(3)

    if kp.getKey("e"):
        me.takeoff()

    sleep(interval)

    a += yaw
    x += int(d * math.cos(math.radians(a)))
    y += int(d * math.sin(math.radians(a)))

    return [lr, fb, ud, yv, x, y]


def drawPoints(img, points):
    for point in points:
        cv2.circle(img, point, 5, (0, 0, 255), cv2.FILLED)

    cv2.circle(img, points[-1], 8, (0, 255, 0), cv2.FILLED)

    cv2.putText(
        img,
        f"({(points[-1][0] - 500) / 100},{(points[-1][1] - 500) / 100})m",
        (points[-1][0] + 10, points[-1][1] + 30),
        cv2.FONT_HERSHEY_PLAIN,
        1,
        (255, 0, 255),
        1,
    )


while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    img = np.zeros((1000, 1000, 3), np.uint8)

    if points[-1][0] != vals[4] or points[-1][1] != vals[5]:
        points.append((vals[4], vals[5]))

    drawPoints(img, points)
    cv2.imshow("Output", img)
    cv2.waitKey(1)


# Face tracking


me = tello.Tello()

me.connect()

print(me.get_battery())

me.streamon()

me.takeoff()

me.send_rc_control(0, 0, 25, 0)

time.sleep(2.2)

w, h = 360, 240

fbRange = [6200, 6800]

pid = [0.4, 0.4, 0]

pError = 0


def findFace(img):

    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.2, 8)
    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cx = x + w // 2
        cy = y + h // 2
        area = w * h
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        myFaceListC.append([cx, cy])
        myFaceListArea.append(area)

    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        return img, [myFaceListC[i], myFaceListArea[i]]

    else:

        return img, [[0, 0], 0]


def trackFace(info, w, pid, pError):

    area = info[1]

    x, y = info[0]
    fb = 0
    error = x - w // 2
    speed = pid[0] * error + pid[1] * (error - pError)
    speed = int(np.clip(speed, -100, 100))

    if area > fbRange[0] and area < fbRange[1]:
        fb = 0

    elif area > fbRange[1]:
        fb = -20

    elif area < fbRange[0] and area != 0:
        fb = 20

    if x == 0:
        speed = 0
        error = 0

    # print(speed, fb)

    me.send_rc_control(0, fb, 0, speed)

    return error


# cap = cv2.VideoCapture(1)

while True:

    # _, img = cap.read()

    img = me.get_frame_read().frame
    img = cv2.resize(img, (w, h))
    img, info = findFace(img)

    pError = trackFace(info, w, pid, pError)
    # print("Center", info[0], "Area", info[1])
    cv2.imshow("Output", img)

    if cv2.waitkey(1) & 0xFF == ord("q"):
        me.land()
        break


# Line follower


me = tello.Tello()
me.connect()
print(me.get_battery())
me.streamon()

# me.takeoff()

cap = cv2.VideoCapture(1)
hsvVals = [0, 0, 188, 179, 33, 245]
sensors = 3
threshold = 0.2
width, height = 480, 360
senstivity = 3  # if number is high less sensitive

weights = [-25, -15, 0, 15, 25]

fSpeed = 15

curve = 0


def thresholding(img):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsvVals[0], hsvVals[1], hsvVals[2]])
    upper = np.array([hsvVals[3], hsvVals[4], hsvVals[5]])
    mask = cv2.inRange(hsv, lower, upper)

    return mask


def getContours(imgThres, img):

    cx = 0

    contours, hieracrhy = cv2.findContours(
        imgThres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )

    if len(contours) != 0:

        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)

        cx = x + w // 2
        cy = y + h // 2

        cv2.drawContours(img, biggest, -1, (255, 0, 255), 7)
        cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    return cx


def getSensorOutput(imgThres, sensors):

    imgs = np.hsplit(imgThres, sensors)
    totalPixels = (img.shape[1] // sensors) * img.shape[0]
    senOut = []

    for x, im in enumerate(imgs):
        pixelCount = cv2.countNonZero(im)
        if pixelCount > threshold * totalPixels:
            senOut.append(1)

        else:
            senOut.append(0)

        # cv2.imshow(str(x), im)

    # print(senOut)

    return senOut


def sendCommands(senOut, cx):

    global curve

    ## TRANSLATION

    lr = (cx - width // 2) // senstivity
    lr = int(np.clip(lr, -10, 10))

    if 2 > lr > -2:
        lr = 0

    ## Rotation

    if senOut == [1, 0, 0]:
        curve = weights[0]

    elif senOut == [1, 1, 0]:
        curve = weights[1]

    elif senOut == [0, 1, 0]:
        curve = weights[2]

    elif senOut == [0, 1, 1]:
        curve = weights[3]

    elif senOut == [0, 0, 1]:
        curve = weights[4]

    elif senOut == [0, 0, 0]:
        curve = weights[2]

    elif senOut == [1, 1, 1]:
        curve = weights[2]

    elif senOut == [1, 0, 1]:
        curve = weights[2]

    me.send_rc_control(lr, fSpeed, 0, curve)


while True:

    # _, img = cap.read()

    img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    img = cv2.flip(img, 0)
    imgThres = thresholding(img)

    cx = getContours(imgThres, img)  ## For Translation
    senOut = getSensorOutput(imgThres, sensors)  ## Rotation
    sendCommands(senOut, cx)

    cv2.imshow("Output", img)
    cv2.imshow("Path", imgThres)
    cv2.waitKey(1)
