﻿
#### Face Detection and Obfuscation in Video

### How to use this repo:


### DETECTION
First clone the tensorflow/models repo [here](https://github.com/tensorflow/models).

Then follow the installation functions found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

If you want to retrain the model, we have provided several files to make the process easy. First download the WIDER FACE Dataset from their [website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). Then use convert.py to convert the WIDER FACE annotations to Pascal VOC XML format annotations. After, use convert_wider_tf_record.py to create tfrecord files to be used by the object detection API. Read the following [tutorial on how to use a custom dataset with the API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md). We have provided the needed files that are outlined in the tutorial. Just change paths defined in those files to reflect the configuration on you machine.


### OBFUSCATION
To perform the face obfuscation, install the following libraries 
```bash
python -m pip install numpy opencv-python dlib imutils dippykit scipy tensorflow pillow
```

Next, download the VGG19 convolutional neural network into the final directory from https://github.com/tensorlayer/pretrained-models/blob/master/models/vgg19.npy

Face swap can be done on a webcam feed with faceSwapLive.py by changing line 176: filename1 = 'pratt.jpg' to whatever source image you want to superimpose on faces in the webcam feed.

Face swap can be done on an image with multiple people with faceSwapMultiple.py. Similarly, the source and destination images can be changed on lines 196 and 197.

Face swap can be done on a video frame-by-frame using faceSwapVideo.py. The wrapping function provided is called swapVideo(imagename='pratt.jpg', videoname='input1.mp4', outname='output.mp4'), and an example call in a loop is on line 317

Face swap can also be done on a static image using the VGG19 CNN by using transfer_gatys_tf.py and putting the content and style images in the files content_img.jpg and style_img.jpg, respectively. The output will be written to faceswap.jpg

