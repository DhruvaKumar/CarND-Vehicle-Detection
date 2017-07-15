# Vehicle Detection

This project is a part of Udacity's self-driving nanodegree program. The goal of this project is to detect vehicles in a video. The code is in [main.ipynb](./main.ipynb). The final video is [here](./project_video_result.mp4)

[//]: # (Image References)

[image1]: ./output_images/outputtensor.png "output_tensor"
[image2]: ./output_images/result_testimgs.png "result"
[image3]: ./output_images/yolo.png "yolo-tiny"


Introduction
---
Traditionally, vehicle/object detection has been treated as a classification problem. A classifier (SVM, AdaBoost, etc) trained on object features (HoG, haar) evaluates the image at various locations and scales. Bounding boxes are allocated to locations that are classified with a high probability. Recent approaches like [YOLO](https://arxiv.org/pdf/1612.08242.pdf) and [SSD](https://arxiv.org/pdf/1512.02325.pdf) have started treating object detection as a unified regression problem where bounding boxes and object class probablities are detected directly from image pixels.


YOLO
---
In this project, we implement the tiny model of [YOLOv1](https://pjreddie.com/media/files/papers/yolo.pdf) which process images at 155 frames per second. 

The network consists of 9 convolution layers and 3 fully connected layers. Each convolution layer is followed by a leaky ReLU and a max pooling layer.

![alt text][image3]

The final output is a 7x7x30 tensor of class probability and bounding box predictions. 

### Output tensor

Each input image is split into SxS grid cells. Each cell predicts B bounding boxes coordinates(x,y,w,h) and a confidence scores for each box. The confidence score is a measure of how confident the model is about the object being in the box. It's defined as P(object) * IOU (intersection over union between the ground truth and predicted box). Each grid cell also predicts the conditional class probabilites P(class|object). 

Therefore, the output is a `S x S x (B x 5 + C)` tensor. 
```
S = grid cells
B = bounding boxes
C = # of classes
5: [x, y, w, h, P(object)]
```
S=7. B=2. Since the model has been trained on PASCAL VOC, we have 20 classes. C=20.

Output tensor: `7x7x(2x5+20) = 1470`

At test time, the box and class probablities are combined that gives us class specific confidence scored in each predicted bounding box P(object|class).

![alt text][image1]
Image taken from https://docs.google.com/presentation/d/1kAa7NOamBt4calBU9iHgT8a86RRHz9Yz2oh4-GTdX6M/edit#slide=id.p

### Training

We use pretrained weights from [here](https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU).

The grid cell which has the center of the ground truth bounding box is responsible for predicting it. During training, each ground truth bounding box is matched to its cell, its grid cell class probabilities P(class|object) are adjusted. For all the predicted bounding box of that cell, it increases the condidence of that box which closely matches the ground truth and decreases the confidence of the other bounding boxes. Some grid cells won't be matched with any ground truth boxes. The confidence of predicted boxes by those cells are decreased. 

The model is initially trained on Imagenet for classification and then on PASCAL VOC for detection.

### Testing

Each image is cropped to its region of interest, resized to 448x448 since the model's trained with that input size, normalized between -1 to 1 and passed through the network. 
The output tensor from the model is converted to bounding boxes that are predicted with a confidence > 0.16. Overlapping bounding boxes whose IOU > threshold are combined. 

### Results

![alt text][image2]

Pipeline on test video: [project_video_result.mp4](./project_video_result.mp4)


houghts and future work
---
It's exciting to see recent progress in real time, end-end object detection using CNNs. It alleviates some of the pains of fine tuning each module in classical pipelines. It also allows the model to look at the context of the entire image while detecting objects, rather than fixed regions/windows. Tiny-YOLO v1 works largely well except while localizing bounding boxes. It took some time in finding the right thresholds to correctly retain and combine bounding boxes during post processing. Even then, at times, it fails to accurately localize the boxes. [YOLOv2](https://arxiv.org/pdf/1612.08242.pdf) which uses only convolution layers and a pass through layer addresses these issues by using anchor boxes and constraining the location prediction. It has other improvements over version 1 like using batch normalization, high resolution classifiers and multi-scale features. On testing with the full blown YOLOv2, the bounding box prediction were far better. In some cases, it would also detect vehicles on the opposite side of the lane, even though they were partially occluded by the lane divider. 

**Transfer learning**
It would be interesting to apply transfer learning to these models for the sole purpose of vehicle detection. By trimming down the model, retaining early convolution layers and its weights (since the low level features they have learnt can be applicable to vehicle detection) and retraining the final layers with vehicle images, it would likely detect vehicles more accuractely. The full blown YOLOv2 seems like an overkill for vehicle detection. A scaled down version, fine-tuned with the vehicles and non-vehicles may both speed up testing and increase accuracy.

**Temporal information**
These models detect static images well. For the purpose of vehicle detection and tracking, we can use temporal information of detected vehicles over a couple of frames to strengthen our detection and tracking. In some frames, we completely lose the detection or have false positives. Even if we do get near 100% accuracy on static images, detection would fail if the camera is blinded for some reason (sunlight, snow, etc) for some time. We humans tend to rely on what was happening in the last observable frame and extrapolate the intentions over a forseeable future. If the model can be architected in such a way that it can learn temporal association of vehicles over frames like how LSTMs do, it would not only increase the accuracy of detections but help detect vehicles for some time even when the camera is blinded. 


