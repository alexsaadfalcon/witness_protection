
#### Face Detection and Obfuscation in Video

### How to use this repo:

First clone the tensorflow/models repo [here](https://github.com/tensorflow/models).

Then follow the installation functions found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).

If you want to retrain the model, we have provided several files to make the process easy. First download the WIDER FACE Dataset from their [website](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/). Then use convert.py to convert the WIDER FACE annotations to Pascal VOC XML format annotations. After, use convert_wider_tf_record.py to create tfrecord files to be used by the object detection API. Read the following [tutorial on how to use a custom dataset with the API](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md). We have provided the needed files that are outlined in the tutorial. Just change paths defined in those files to reflect the configuration on you machine.
