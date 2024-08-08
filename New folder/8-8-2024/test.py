import cv2
import numpy as np

#read the image with the particular library 
image =cv2.imread('test.jpg')

if image is None:
    print("could not read the image")
    exit()

#original image    
#cv2.imshow('original image',image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

#convert to greyscale
grey_image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#we apply a gaussin blur into the above image
blurred_image =cv2.GaussianBlur(grey_image,(5,5),0)

#edge dectection by using canny in cv2
edges =cv2.Canny(blurred_image,50,150)

#original image
cv2.imshow('original image',image)
cv2.waitKey(0)

cv2.imshow('greyscale image',image)
cv2.waitKey(0)

cv2.imshow('blurred image',image)
cv2.waitKey(0)

cv2.imshow('edges image',image)
cv2.waitKey(0)

cv2.destroyAllWindows()