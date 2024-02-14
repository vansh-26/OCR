import cv2
import pytesseract
import numpy as np

pytesseract.pytesseract.tesseract_cmd='/opt/homebrew/bin/tesseract'
cascade=cv2.CascadeClassifier("/Users/vanshtrivedi/Desktop/MySpace/Development/NumberPlateBRL/haarcascade_russian_plate_number.xml")

def extract_num(img_name):
    global read
    img =cv2.imread(img_name)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    nplate= cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in nplate:
         
         # Cropping number plate
         a,b=(int(0.02*img.shape[0]),int(0.025*img.shape[1]))
         plate = img[y+a:y+h-a,x+b:x+w-b,:]
         
         # image processing  
         kernel = np.ones((1,1),np.uint8)
         plate = cv2.dilate(plate,kernel,iterations=1)
         plate = cv2.erode(plate,kernel,iterations=1)
         plate_gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
         (thresh,plate) = cv2.threshold(plate_gray,127,255,cv2.THRESH_BINARY)
         read = pytesseract.image_to_string(plate)
         read = ''.join(e for e in read if e.isalnum())
    print(read)


extract_num('/Users/vanshtrivedi/Desktop/MySpace/Development/NumberPlateBRL/images.jpeg')