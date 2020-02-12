#!/usr/bin/env python
# coding: utf-8

# In[1]:


#To import required libraries to solve q2

import cv2 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# To read two images 
image1 = cv2.imread("p1.jpg")
image2 = cv2.imread("p2.jpg")


# In[3]:


# To show images:
figure, ax = plt.subplots(1, 2, figsize=(16, 16))


ax[0].imshow(image1)
ax[1].imshow(image2)


# In[6]:


#To initiate the ORB detector:
sift = cv2.xfeatures2d.SIFT_create()


image1GRAY = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image2GRAY = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


#To compute the keypoints and descriptors for images using SIFT
keyPoints1, descriptors1 = sift.detectAndCompute(image1, None)
keyPoints2, descriptors2 = sift.detectAndCompute(image2, None)





# In[7]:


#To draw detected keypoints on images and show them:
image1_KD = cv2.drawKeypoints(image1GRAY, keyPoints1, image1)

plt.imshow(image1_KD)


# In[8]:


# To draw detected keypoints on image 2 and show them:
image2_KD = cv2.drawKeypoints(image2GRAY, keyPoints2, image2)
plt.imshow(image2_KD)


# In[22]:


#To setup feature matching tool:
import numpy as np
sift_match = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = sift_match.match(descriptors1,descriptors2)
matches = sorted(matches, key = lambda x:x.distance)


# In[23]:


#To draw the matching result:
finalResult = cv2.drawMatches(image1_KD, keyPoints1, image2_KD, keyPoints2, matches[:50], image2_KD, flags=2)
plt.imshow(finalResult),plt.show()


# In[ ]:




