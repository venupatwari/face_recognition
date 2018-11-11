import cv2

#just a shortcut
import numpy as np

import sqlite3

from PIL import Image

import ST7735 as TFT
import Adafruit_GPIO as GPIO
import Adafruit_GPIO.SPI as SPI


WIDTH = 128
HEIGHT = 160
SPEED_HZ = 4000000


# Raspberry Pi configuration.
DC = 24
RST = 25
SPI_PORT = 0
SPI_DEVICE = 0

# Create TFT LCD display class.
disp = TFT.ST7735(
    DC,
    rst=RST,
    spi=SPI.SpiDev(
        SPI_PORT,
        SPI_DEVICE,
        max_speed_hz=SPEED_HZ))

#using defualt haarcascade from opencv library (opencv->sources->data->haarcascades)

#calling cascade classifier with file name with file extension
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');

def getProfile(id):
    conn =sqlite3.connect("facedatabase.db");
    cmd ="SELECT * FROM details WHERE id="+str(id);
    cursor = conn.execute(cmd);
    profile=None;
    for row in cursor:
        profile=row;
    conn.close();
    return profile;

def display():
    disp_img = numpy.ones((128,128,1),numpy.unit8)*0;
    font_ = cv2.FONT_HERSHEY_SIMPLEX;
    cv2.putText(disp_img,'ID: '+str(profile[0]),(5,40),font_,1,(255,255,255),2,cv2.CV_AA);
    cv2.putText(disp_img,str(profile[1]),(5,40),font_,1,(255,255,255),2,cv2.CV_AA);
    cv2.imshow('display image',disp_img);
    cv2.imwrite('temp.jpg',disp_img);
    return temp.jpg
    

#video capture object to capture images for webcam value (video capture id) is 0 or else try 1
cam = cv2.VideoCapture(0);
#capture frames one by one and detect faces and show them in a window

#creating recognizer
recognizer = cv2.createLBPHFaceRecognizer();
recognizer.load('recognizer\\trainingdata.yml');
id = 0;
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4);
# parameters first one font size last on thickness

#creating loop and braking it with key
while(True):
    ret,img = cam.read();
    #returns status variable and one captured image

    #for classifier to work converting color image into grayscale one
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);

    #list to store the faces
    faces = faceDetect.detectMultiScale(gray,1.3,5);
    #this will detect all the faces in the frames and return co-ordinates of the faces in the frame

    #we can get multiple f(x+waces so drawing rectangles around faces
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2);
        # parameters variablles, starting point, ending point, BGR, thickness or width

        id,conf = recognizer.predict(gray[y:y+h,x:x+w])
        profile = getProfile(id);
        if(profile!=None):
            cv2.cv.PutText(cv2.cv.fromarray(img),str(profile[1]),(x,y+h+60),font,255);
        cv2.waitKey(1);

    #displaying the faces
    cv2.imshow("Face",img);

    # Initialize display.
    disp.begin()
    display();

    # Load an image.
    print('Loading image...')
    disp_image = Image.open('temp.jpg')

    # Resize the image and rotate it so matches the display.
    disp_image = image.rotate(90).resize((WIDTH, HEIGHT))

    # Draw the image on the display hardware.
    print('Drawing image')
    disp.display(disp_image)

    #wait command or else it wont work and exiting the loop on if waitKey value is 2
    if(cv2.waitKey(1)==ord('q')):
        break;
    
#freeing up camera
cam.release();

#closing all windows
cv2.destroyAllWindows();
