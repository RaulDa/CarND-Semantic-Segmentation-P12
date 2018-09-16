# **Semantic Segmentation**

---

**Semantic Segmentation Project**

The goals / steps of this project are the following:

* Detect road pixels of individual frames taken from an automotive video.
* Use a Fully Convolutiontal Network, so that the spatial information is preserved.

[//]: # (Image References)

[image1]: ./image_output/umm_000013.png "Image detection"
[image2]: ./output_images/undistort_test_image.png "Road Transformed"
[image3]: ./output_images/threshold.png "Binary Example"
[image4]: ./output_images/warped.png "Warp Example"
[image5]: ./output_images/lane_detection.png "Fit Visual"
[image6]: ./output_images/view.png "Output"
[video1]: ./project_video.mp4 "Video"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/989/view) individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

### Build the neural network

#### 1. Does the project load the pretrained vgg model?

The pretrained vgg model is loaded in the function `load_vgg` (line `34` of `main.py`), via `load` Tensorflow function. Additionally, the `input_image` and `keep_prob` placeholders as well as the tensors `layer3_out`, `layer4_out` and `layer7_out` are extracted from the pretrained model for creating a fully convolutional network and building skip connections according to the `FCN-8s` architecture.

#### 2. Does the project learn the correct features from the images?

The pretrained model extracts the features from the images with the original configuration, since the feature layers and the `1x1` convolutions are already defined in the provided network. The fully convolutional network is built in the function `layers`(line `69` of `main.py`) via application of transposed convolutions to `layer7_out` and skip connections with `layer4_out` and `layer3_out`.

Two transposed convolutions of `kernel_size=4` and `strides=2` are applied to `layer7_out` and one to `layer4_out` before connecting them. A transposed convolution of `kernel_size=16` and `strides=8` is applied to the result, as well as to `layer3_out`. The sum of both represents the final output tensor (logits), which has the shape of the original image and where the pixels are correctly classified thanks to a successful feature learn.

#### 3. Does the project optimize the neural network?

The function `optimize` (line `97` of `main.py`) implements the optimization operations and calculates the loss. An adam optimizer is used to minimize it and the total loss also includes the loss obtained via L2 regularization with scale of `1e-3`, applied to the last transposed convolution.

#### 4. Does the project train the neural network?

The function `train_nn` (line `120` of `main.py`) performs the training. The training tensor defined in the `optimize` function is evaluated, as well as the loss one. The evaluation is done for `epochs=15` and taking `batch_size=20`. A `keep_prob` of `0.5` is defined for the dropout layers of the vgg, as well as a `learning_rate` of `0.00075`. The loss is monitored while the training is performed.

### Neural network training

#### 5. Does the project train the model correctly?

The model decreases the loss on average for each epoch. For the training corresponding to the attached image and video results the following loss was obtained:

| Epoch         		|     Loss	        					|
|:---------------------:|:---------------------------------------------:|
| 1         		| 0.47   							|
| 2     	| 0.40 	|
| 3					|	0.39											|
| 4	      	| 0.36 				|
|	5					|	0.34											|
| 6	    | 0.37     									|
| 7					|	0.30											|
| 8	      	| 0.25 				|
|	9					|	0.24											|
| 10		| 0.18        									|
| 11		| 0.19       									|
| 12				| 0.18        									|
|	13					|	0.19											|
| 14		| 0.18        									|
| 15		| 0.16       									|




#### 6. Does the project use reasonable hyperparameters?

As mentioned in previous sections, the evaluation is done for `epochs=15` and taking `batch_size=20`. The mentioned number of epochs was selected because it was the limit number by which the loss stops being reduced. The mentioned batch size ensured that the used GPU does not run out of memory. Both values ensured also a good detection of both the image test set and the video.

#### 7. Does the project correctly label the road?

Most pixels are labeled close to the best solutions. The following image is an example of the detection:

![alt text][image1]

Additionally the logits were applied to a random video (`project_video.mp4`). The result is shown in the following [video](https://www.youtube.com/watch?v=5nY5wKeFI98).
