#!/usr/bin/env python
# coding: utf-8

# In[1]:


#To import required libraries to solve q2
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import imutils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# To read two images 
image1 = cv2.imread("p2.jpg")
image2 = cv2.imread("p1.jpg")


# In[3]:


# To convert images color to Gray and show images:
figure, ax = plt.subplots(1, 2, figsize=(16, 16))


ax[0].imshow(image1)
ax[1].imshow(image2)


# In[4]:


# To convert images into Gray Color:
image1GRAY = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2GRAY = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


#To Show images in Gray scale:
figure, ax = plt.subplots(1, 2, figsize=(16, 16))


ax[0].imshow(image1GRAY)
ax[1].imshow(image2GRAY)


# In[5]:


#To initiate the SIFT detector:
sift = cv2.xfeatures2d.SIFT_create()


#To compute the keypoints and descriptors for images using SIFT
keyPoints1, descriptors1 = sift.detectAndCompute(image1GRAY, None)
keyPoints2, descriptors2 = sift.detectAndCompute(image2GRAY, None)


# In[6]:


#To draw detected keypoints on the first image and show them:
plt.figure(figsize = (13,13))
image1_KD = cv2.drawKeypoints(image1GRAY, keyPoints1, image1GRAY)
plt.imshow(image1_KD, interpolation='none', aspect='auto')


# In[7]:


# To draw detected keypoints on image 2 and show them:

plt.figure(figsize = (13,13))
image2_KD = cv2.drawKeypoints(image2GRAY, keyPoints2, image2GRAY)
plt.imshow(image2_KD, interpolation='none', aspect='auto')


# In[8]:


#To setup feature matching tool: I used BF Matcher NORM_L2
sift_match = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


# To sort matches based on the distance
matchesList = sift_match.match(descriptors1,descriptors2)
matchesList = sorted(matchesList, key = lambda x:x.distance)


# In[9]:


#To draw the matching result:
plt.figure(figsize = (13,13))


#Figure below only shows 250 matching 
matchingResult = cv2.drawMatches(image1_KD, keyPoints1, image2_KD, keyPoints2, matchesList[:250], image1_KD, flags=2)
plt.imshow(matchingResult, interpolation='none', aspect='auto'),plt.show()


# In[11]:


#To construct two sets of points from detected Key points
sourcePoints = np.float32([ keyPoints1[m.queryIdx].pt for m in matchesList ]).reshape(-1,1,2)
distinationPoints = np.float32([ keyPoints2[m.trainIdx].pt for m in matchesList ]).reshape(-1,1,2)



#To estimate the homography between the constructed sets of points
homographyResult, mask = cv2.findHomography(sourcePoints, distinationPoints, cv2.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

h,w = image1GRAY.shape

points = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
destination = cv2.perspectiveTransform(points,homographyResult)


image2GRAY = cv2.polylines(image1GRAY,[np.int32(destination)],True,255,3, cv2.LINE_AA)


# In[12]:


#To get the width and height values:
width = image1GRAY.shape[1] + image2GRAY.shape[1]
height = image1GRAY.shape[0] + image2GRAY.shape[0]


# To Apply warp Perspective on the result
warpResult = cv2.warpPerspective(image1GRAY, homographyResult, (width, height))
warpResult[0:image1GRAY.shape[0], 0:image2GRAY.shape[1]] = image2GRAY


# To show the result
plt.figure(figsize=(20,10))
plt.imshow(warpResult)
plt.show()


# In[14]:


# To threshold the warpped image:
threshold = cv2.threshold(warpResult, 0, 255, cv2.THRESH_BINARY)[1]

# To find contours from the binary image
contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# get the maximum contour area
c = max(contours, key=cv2.contourArea)

# get a bbox from the contour area
(x, y, w, h) = cv2.boundingRect(c)

# crop the image to the bbox coordinates
warpResult = warpResult[y:y + h, x:x + w]

# show the cropped image
plt.figure(figsize=(20,10))
plt.imshow(warpResult)


# In[ ]:




