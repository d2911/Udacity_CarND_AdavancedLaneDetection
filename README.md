# ADVANECED LANE FINDING Project

## Requirement 1: Camera Calibration:
Computed the camera matrix and distortion coefficients using openCV on 9x6 chessboard images provided. Use the determined values on an example image.

## Requirement 2: Colour transform and Gradients
Create a binary image of each image containing likely lane pixels using colour transform, Gradients or other methods.

## Requirement 3: Perspective transformation
Create "bird’s eye view" image of each test image using perspective transformation or other methods.

## Requirement 4: Identified lane-line pixels and fit their positions with a polynomial
Identify lane-lines pixels from rectified binary image, then identify left and right lane and fit with curved function form.

## Requirement 5: Calculate radius of curvature of lane line and position w.r.t center
The radius of curvature may be given in meters assuming the curve of the road follows a circle.
The position of the vehicle, assume the camera is mounted at the center of the car and the deviation of the midpoint of the lane from the center of the image is the offset.

As with polynomial fitting convert pixels to meters.

Hint : curvature is 1km

## Requirement 6: plot the result on image
Mark the lane between two polynomial and inverse perspective transform lane back to original image and put text on output image to show the results radius of curvature and car position w.r.t center of lane.

# Reflection
Video is nothing but a sequence of images.
Pipeline is build using the helper functions and/or concepts discussed during this course to process the image. Once all the images in ".../test_images" folder is tested for proper working, the pipeline is used to test the video “project_video.mp4". During the process the parameter in the pipeline are adjusted to make pipeline work properly for video.
One of the test image shown below is used to explain the pipeline.

## 1. Camera Calibration
20 images of chess board with 9 x 6 corners provided with the project is used to calibrate the camera. Of the 20 images where ever 9x6 corners are identified image points [2D] and object points [3D] is constructed and appended into single array. With image and object points cv2.calibrateCamera determines “camera Matrix” and “distortion Coeffs”. These values for 20 images is shown below.

With determined camera Matrix and distortion coefficients the test image is undistorted.

## 2. Create Binary Image
One of the simplest method was to use canny edge detection on grayscaled image and construct a binary image by applying a threshold on the strength of gradient at each pixel. Here we are analysing image deeper by considering gradient in any one direction x or y-axis and by considering different colour spaces [RGB, HLS, HSV] where analysis can be done on each channel R, G, B, H, L, S or V. Threshold can be applied on the most useful channel and binary image can be formed by combining selected components.

In this implementation, from HLS colour space L-channel and s-channel are selected. On L-channel gradient in x-axis is calculated and Threshold [20 <= x <= 100 ] is applied to form 1st binary image. On S-channel threshold [170=x<=255] is applied to form 2nd binary image. Both images are combined to form final binary image as shown below.
Green ones represent pixels from 1st binary image and blue ones represent 2nd binary image.

## 3. Perspective Transformation “Bird Eye View”
In order to find the lane lines and to determine curvature of the lane lines it would be helpful to have a top view of the binary image formed. Hence perspective transformation on the points selected as per below image is done using cv2.getPerspectiveTransform and cv2.warpPerspective.

Top view of the binary image as per above selection, we can clearly see lane lines.

## 4. Find left and right lane lines & fit a polynomial of 2nd degree
  
  4.1. apply histogram across x-axis on bottom half of image
  
  4.2. Identify two peaks, one on the left side of bottom image and second on right side of bottom image. When we plot the histogram value we can see one peak on left and one on the right, both at the exact x-axis point where lane lines are starting.
  
  4.3. SLIDING WINDOW : consider 9 window on each lane line. Create 1st window on both lane lines where the peak is identified. Create next window above first and adjust position across x-axis based on the median value of pixels inside the window. With this approach all 9 window is positioned on left and right lanes.
  
  4.4. Identify all x-axis indices and y-axis indices inside the windows where pixel value is non-zero separately for left and right line.
  
  4.5. Using these x & y indices of each line determine the polynomial coefficients using np.polyfit(). Hence polynomial of 2nd degree is derived for left and right lane lines.
Window, pixels inside windows [red-left, blue right] and derived polynomial is drawn on the image for visualization.

  4.6. using cv2.fillPoly() all points between two lane lines [polynomials ] can be filled which is helpful to visualize complete lane on which vehicle is travelling.

## 5. Perspective Transformation “Bird Eye View” to normal view
Inverse perspective transform the image from step 4 to normal camera view using the same points used in step 3.
Merge it with calibrated image to form output image.

## 6. Determine radius of curvature and position of car
  6.1. Use below formula to calculate radius of curvature of left and right lane lines.
    
  For 2nd degree polynomial f(y)=Ay2+By+C
  
  6.2. Average of radius of curvature of left and right lanes is radius of curvature of the lane.
  
  6.3. From step 4.5, average of x-points between left and right lines at bottom of the image [y = 719] is center point of lane.
  
  6.4. Considering camera is mounted on the middle of car, midpoint of total image length on x-axis gives position of center of car.
  
  6.5. Center point of lane and position of car is used to determine relative car position from center of lane.
Radius of curvature of lane and position of car w.r.t center of lane is printed on image.

## 7. Testing the pipeline on other test images

Final output of all test images are shown below.

## 8. Video processing

![](/ReadMe_Images/project_video_0.gif)
![](/ReadMe_Images/project_video_1.gif)
