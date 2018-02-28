## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/figure1.png "Perspective Transform"
[image2]: ./output_images/figure2.png "Binary Mask"
[image3]: ./output_images/figure3.png "Lane Drawing"
[image4]: ./output_images/figure4.png "Chessboard"
[video1]: ./output_images/project_video.mp4 "Project Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

[README file can be found here](https://github.com/esouliot/CarND-Advanced-Lane-Lines/blob/master/README.md)

[Jupyter notebook with necessary code can be found here](https://github.com/esouliot/CarND-Advanced-Lane-Lines/blob/master/examples/P4_Advanced_Lane_Lines.ipynb)

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the second code cell of the IPython notebook located in "./examples/P4_Advanced_Lane_Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Chessboard Undistorted][image4]

### Pipeline

#### I used the following steps to take a raw image from a video stream, and return it as an undistorted, annotated image with demarcated lane lines and numerical approximations for lane curvature and vehicle distance from the center of the lane

1.) Used the distortion matrices calculated using the provided chessboard images to undistort the image

2.) Defined a region of interest based on four points centered around the vertical midpoint (640px)

3.) Calculated a perspective transform to give an image that appears to be taken from above

#### These three steps can be seen in the following code from section 3 of the Jupyter notebook

```
    #Undistort using our global undistortion values
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    #Region of interest
    #To be used in "birds-eye" perspective transform
    pts1 = np.float32([[540, 450], [740, 450], [100, 720], [1180, 720]])
    pts2 = np.float32([[0, 0], [1280, 0], [300, 720], [980, 720]])

    #Get perspective transform
    #Output should look as though the image was taken from above
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst2 = cv2.warpPerspective(dst, M, (1280, 720))

```

#### As well as in the following images

![Original to Perspective Transform][image1]

#### Having reached the perspective transform, I then performed the following

4.) Searched for yellow and white pixels in the images. From testing throughout the project, it was apparent that the most robust method for finding lane line pixels in a given image was to search by color, and not necessarily by Sobel gradient. This became apparent in the project video in portions when the road surface contained shadows from trees and other objects in the periphery. 

The functions for finding yellow and white lines are as follows

```
#Function uses hue and saturation values to find yellow-ish pixels
def yellow_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h = hls[:,:,0]
    s = hls[:,:,2]
    binary = np.zeros_like(h)
    binary[(h >= 15) & (h <= 25) & (s >= 125)] = 1
    return binary

#White lane lines are easily found using the HLS lightness filter
def white_thresh(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l = hls[:,:,1]
    binary = np.zeros_like(l)
    binary[(l >= 220)] = 1
    return binary
```

And the binary mask was created within the pipeline with the following

```
    #Looking for yellow and white lines        
    white = white_thresh(dst2)
    yellow = yellow_thresh(dst2)

    #Creating a blank image the same dimensions as our video
    binary = np.zeros_like(white)
    
    #If a pixel was found to be part of a yellow or white line
    #Light up that pixel
    binary[(yellow == 1) | (white == 1)] = 255
```

The binary masks created by the yellow and white line functions is illustrated as follows

![Binary Masks][image2]

5.) After creating a binary mask for the two lane lines, I then used modified versions of the line drawing functions provided in sections 33 and 35 of the "Advanced Techniques for Finding Lane Lines" lesson. The function includes the following steps. (For the sake of brevity, the code for this function has been omitted. It can be found in section 2 of the Jupyter notebook).

- Performing a sliding window search on the left and right lines
- Saving the indices of any nonzero pixels found in the left and right lines
- Using these indices with the NumPy polyfit method to find coefficients for two second-order polynomials (ax^2 + bx + c)
- Using the second-order polynomials to annotate the lane between the two lane lines
- Scaling the polynomials to give a numerical approximation of the lane's curve radius
- Using the scaled distance between the two lane lines to approximate the vehicle's deviation from center

6.) Having returned the warped image with the lane markings overlayed, I unwarped the image, and used the calculated radius and off-center values to annotate the image. This final image is then returned at the end of the pipeline

```
    #Function looks for nonzero pixels on the left and right side
    #It uses them to create an order two polynomial fit
    #Using the polynomial, we can also calculate curve radius
    #As well as distance from the center of the lane
    lines, radius, width = draw_lines(binary)
    
    #Display radius in kilometers and round it to one decimal place
    radius /= 1000
    radius = np.around(radius, decimals = 1)
    
    #Display off-center measure in meters, rounded to one decimal place
    width = np.around(width, decimals = 1)
    
    #To be drawn onto the output image
    radius_string = 'Radius of curve = ' + str(radius) + ' km'
    width_string = 'Car is ' + str(width) + 'm off center'

    #Reversing the perspective transform to give us our annotated image
    M_unwarp = cv2.getPerspectiveTransform(pts2, pts1)
    unwarped = cv2.warpPerspective(lines, M_unwarp, (1280, 720))
    weighted = weighted_img(unwarped, img)

    #Writing curve radius and distance from center on each frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(weighted, radius_string, (100, 100), font, 2, (255, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(weighted, width_string, (100, 150), font, 2, (255, 0, 255), 4, cv2.LINE_AA)

    return weighted
```

![Drawing Lane Lines][image3]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

[Here's a link to my video result](./output_images/project_video.mp4)

I'm not a highway engineer, so I can't comment with much certainty about the accuracy of the road measurements, but with regards to the lane annotation, the pipeline performs consistently, experiencing minor wobble in sections with spotty shadows.

However, the pipeline is not perfect, as can be seen in the annotation of the ["even harder challenge" video](./output_images/harder_challenge_video.mp4). The pipeline finds the lanes for some of the frames, but is very easily confused by objects on the side of the road, leading to catastrophic failure. 

This may be remedied by various modifications to the pipeline, like shortening the region of interest, or perhaps even changing the order of pipeline steps in order to find an effective ROI computationally. Additionally, with the debris covering the right side of the road for certain portions of the video, it may be effective to process the video in sections rather than feeding each image into a static pipeline. More to come on this in later updates...

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The overall point of the project was to apply various new computer vision methods in upgrading our "Lane Lines" repertoire, from the Canny Edge Detection and Hough Lines of Project 1 to using new methods like Sobel Gradient detection, color thresholding in HSL colorspace, sliding window searches, and polynomial fitting of curves. But, even with these new skills, my pipeline has its limitations, stemming from the approach I took overall.

In testing the pipeline using still images, I only managed to use the [provided test images](./test_images/), all of which appeared to come from the [main project video](./project_video.mp4). Of course, the pipeline works successfully on said project video, but it seems that my pipeline only really works under the conditions found in that video (lighting, road type, road conditions).

For future iterations of this project, I will have to modify my line detection methods, and quite possibly my preprocessing methods (including the perspective transform) in order to process the individual frames within the context of their video. Admittedly, my approach included a large amount of manual parameter tuning and use of ["magic numbers"](https://en.wikipedia.org/wiki/Magic_number_(programming)). As such, I will have to revisit my approach, and see where I can tune for parameters computationally instead of manually. Additionally, from discussions with other students, I can also include methods such as [moving averages](https://en.wikipedia.org/wiki/Moving_average) to smooth the lines, and potentially even deep learning in detecting the lane lines instead of conventional computer vision techniques.
