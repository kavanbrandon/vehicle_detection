# **Advanced Lane Finding**

## Kavan Brandon

**Advanced Lane Finding**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/example_car_and_noncar.png
[image2]: ./examples/example_car_and_noncar_with_hog.png
[image3]: ./examples/sliding_search_example.jpg
[video1]: ./output_video.mp4

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the fourth code cell of the IPython notebook called vehicle_detection_notebook.ipynb.  

I started by reading in all the `vehicle` and `non-vehicle` images from provided datasets. The dataset consisted of 8,792 vehicle images and 8,968 non-vehicle images.

Below is an example of one `vehicle` and `non-vehicle` image:

![alt text][image1]

In the third code cell of the IPython notebook, I defined the functions used in the Object Detection lesson including a function for extracting HOG features, spatial binning, and computing color histograms. The function single_img_features is used to extract the combined features from a given image.

#### 2. Explain how you settled on your final choice of HOG parameters.

As described in the lesson, the orientations , pixels_per_cell , and the cells_per_block descriptors control the dimensions of the resulting feature vector. After conducting research online, it seems the generally recommended amount of orientations should be between the range of 9 to 12 to improve accuracy. I ended up settling on 9 orientations which was also the value used in the lesson.

A cell is essentially a region defined by a certain amount of pixels within the cell. As described in the lesson, the pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. Since the cells are recommended to be square, I chose 8 x 8.

Lastly, cells_per_block specifies the local area over which the histogram counts in a given cell will be normalized. Dalal and Triggs, the authors who introduced HOG features in 2005, recommend using either 2 x 2 or 3 x 3 cells_per_block in order to obtain reasonable accuracy. I ended up choosing 2 x 2.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)`. In addition, I used `hist_bins = 32` for applying the color histogram and `spatial_size = (32,32)` for spatial binning:

![alt text][image2]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training a classifier can be found in the fifth code cell of the IPython notebook called vehicle_detection_notebook.ipynb.  

I used the following HOG parameters when training the classifier:

```python
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (32,32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
```
Using a color_space value of 'YCrCb' seemed to produce the best accuracy results (98.14%). Using an 'RGB' value resulted in an accuracy of 96% while 'HLS' resulted in an accuracy that was slightly lower than the best result.

I used the function extract_features, which is similar to the single_img_features function, but returns features for a list of images instead of one image. To use StandardScaler(), I needed to create a numpy array where each row is a single feature vector. The array included the extracted car_features and notcar_features. After providing the right format, the data can then be fit to the scaler and transformed.

The next step was splitting the training and test sets before training the classifier. A LinearSVC() was used to fit the classifier with the training data. Here were the results:

```python
124.88881874084473 Seconds to compute features...
Using: 9 orientations, 8 pixels per cell, 2 cells per block, 32 histogram bins, and (32, 32) spatial sampling
Feature vector length: 8460
30.27 Seconds to train SVC...
Test Accurary of SVC =  0.9814
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for implementing a sliding window search can be found in the 8th code cell of the IPython notebook called vehicle_detection_notebook.ipynb.  

I used a cells_per_step of 6, which results in an overlap of 75% considering 64 pixels is the size of the original sampling rate, with 8 cells and 8 pix per cell. This overlap seemed to work better than a 25% or 50% overlap, however, false positives still surface. I kept a scale of 1.5 which was used in the lesson. This resulted in a total of 32 windows. Additionally, I applied a ystart value of 400 and ystop of 656 which only allows searching to be conducted within these bounds (ignores anything above the road).

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I used an `RGB2YCrCb` conversion when searching through an image. The image was further separated into three channels. HOG features, spatial binning, and color histagrams were extracted from the image to develop a robust feature vector pipeline. This was necessary to optimize the performance of the classifier.

Here is an image showing the output of the sliding window search with HOG images included:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The code for processing each video frame can be found in the 12th code cell of the IPython notebook called vehicle_detection_notebook.ipynb.  

The process_image function uses the find_cars to make predictions on each image frame. Furthermore, I used the apply_threshold function to slightly remove false positives from the frame. Lastly, the draw_labeled_bboxes function is used to add overlapping bounding boxes.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Applying a threshold higher than .5 seemed to remove false positives almost too well and started to impact the true positives. It seems like it would be necessary to adjust the overlap and scale parameters to ensure this doesn't happen. Furthermore, the false positives tended to appear around the middle barrier to the left of the video, in addition to the yellow left lane lines. It seems more color extractions could be made to ensure the training data is normalized more robustly. Lastly, the pipeline could be made more robust by ensuring the ordering of the feature extractions results in the most optimal classifications.
