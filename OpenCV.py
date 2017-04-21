# https://github.com/mciantyre/scipy-opencv-notebook
# OpenCV - computer vision and machine learning software library
import cv2
import matplotlib.pyplot as plt

webget "http://www.atpworldtour.com/-/media/images/verdasco-dubai-2017-thursday.jpg"

# getting the downloaded image
image_path = './verdasco-dubai-2017-thursday.jpg'
img = read(image_path)
create_plot(img)



def create_ball_mask(image):
    
    green_lower = (20, 100, 180)
    green_upper = (60, 255, 255)
    
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # to get rid of the extra noise
    mask = cv2.erode(mask, None, iterations=2) # .erode = makes the not needed parts smaller
    mask = cv2.dilate(mask, None, iterations=2) # .dilate = maked the shape clearer
    
    return mask

create_plot(create_ball_mask(img))




def mark_object(image, mask):
    # gives a copy of mask for conturing = list of coordinates
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    
    # find the largest contour in the mask, then use
    # it to compute the minimum enclosing circle and
    # centroid
    c = max(contours, key=cv2.contourArea) # finding the biggest area
    ((x, y), radius) = cv2.minEnclosingCircle(c) # makes nice circle with centr and radius
    
    # draw the circle and centroid on the frame,
    # then update the list of tracked points
    cv2.circle(image, (int(x), int(y)), int(radius), (255, 0, 0), 5)

    return image

img = mark_object(img, mask)
cv2.imwrite('./verdasco-obj-detected.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

