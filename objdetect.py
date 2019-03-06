#!/usr/bin/env python
# coding: utf-8

# In[70]:


# from google.colab import drive
# drive.mount('/gdrive')


# In[ ]:


import cv2
import numpy as np


# In[ ]:


# ! cd /gdrive/My\ Drive && wget https://pjreddie.com/media/files/yolov3.weights


# In[ ]:


# !ls


# In[ ]:


# config = '/gdrive/My Drive/Colab_ML/Object Recognition/yolov3.cfg'
config = 'yolo3.cfg'
weights = 'yolo-small.weights'
classes = 'yolov3.txt'
net = ''


# In[ ]:


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img,class_id,confidence,x,y,x_plus_w,y_plus_h):
    label = str(classes[class_id])
    color = colors[class_id]
    cv2.rectangle(img,(x,y),(x_plus_w,y_plus_h),color,2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# In[ ]:


imagepath = 'testImages/4.png'
image = cv2.imread(imagepath)
Width,Height = image.shape[1],image.shape[0]
scale = .00392

classes = None
f = open('yolov3.txt','r')
classes = [line.strip() for line in f.readlines()]

colors = np.random.uniform(0,255,size = (len(classes),3))
net = cv2.dnn.readNet(weights,config)
blob = cv2.dnn.blobFromImage(image,scale,(416,416), (0,0,0), True, crop=False)
net.setInput(blob)


# In[ ]:


outs = net.forward(get_output_layers(net))
class_ids = []
confidences = []
boxes = []
conf_threshold = 0.5
nms_threshold = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence >0.5:
            center_x = int(detection[0]*Width)
            center_y = int(detection[1]*Height)
            w = int(detection[2]*Width)
            h = int(detection[3]*Height)
            x = center_x -w/2
            y = center_y-h/2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x,y,w,h])
            
# Non max suppresion
indices = cv2.dnn.NMSBoxes(boxes,confidences,conf_threshold,nms_threshold)
for i in indices:
    i = i[0]
    box = boxes[i]
    x,y,w,h = box[0],box[1],box[2],box[3]

    draw_bounding_box(image,class_ids[i],confidences[i],round(x),round(y),round(x+y),round(y+h))


# In[78]:


# cv2.imshow('Objects',image)
imname = imagepath.split('/')[-1]
cv2.imwrite("{}.jpg".format(imname),image)

