# -*- coding: utf-8 -*-

import numpy as np
import cv2
from pathlib import Path
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def calibrateCamera(nx,ny,imgPath):
    objPts = [] #3D points in object plane
    imgPts = [] #2D points in image plane
    
    objP = np.zeros((6*9,3),np.float32)
    objP[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    for path in Path(imgPath).glob('calibration*.jpg'):
        img = cv2.imread(str(path))
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        
        # If found, draw corners
        if ret == True:
            imgPts.append(corners)
            objPts.append(objP)
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    
    

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objPts, imgPts, gray.shape[::-1], None, None)
    
    return mtx,dist
print("calibrating...")
mtx, dist = calibrateCamera(9, 6, 'camera_cal')
print("calibration done")

def binaryImage(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    #combined = np.zeros_like(sxbinary)
    combined = sxbinary | s_binary
    return combined, color_binary

def corners_unwarp(img, points, offset=100):
    
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    #binaryImg = binaryImage(undist)
    
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    #src = np.float32([[600,450], [780,450], [1100,700], [250,700]])
    src = np.float32(points)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                                 [img_size[0]-offset, img_size[1]], 
                                 [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    
    return warped

def corners_unwarp_inverse(img, points, offset=100):
    
    #undist = cv2.undistort(img, mtx, dist, None, mtx)
    
    #binaryImg = binaryImage(undist)
    
    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners
    #src = np.float32([[600,450], [780,450], [1100,700], [250,700]])
    src = np.float32(points)
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    
    dst = np.float32([[offset, 0], [img_size[0]-offset, 0], 
                                 [img_size[0]-offset, img_size[1]], 
                                 [offset, img_size[1]]])
    # Given src and dst points, calculate the perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, Minv, img_size)
    
    return warped


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    window_img = np.zeros_like(np.dstack((binary_warped, binary_warped, binary_warped))*255)

    left_line_pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_pts = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                              ploty])))])
    all_lane_points = np.hstack((left_line_pts, right_line_pts))

    cv2.fillPoly(window_img, np.int_([all_lane_points]), (0,255, 255))

    return window_img, left_fit, right_fit, left_fitx, right_fitx

def measure_curvature_real(image,left_fit_cr, right_fit_cr):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Start by generating our fake example data
    # Make sure to feed in your real data instead in your project!
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

"pipeline"
def pipeline(ipImage):
        
    dstImg = cv2.undistort(ipImage, mtx, dist, None, mtx)
    
    binaryImg, color_binary = binaryImage(dstImg)
    
    pts = [[590,450], [710,450], [1150,700], [200,700]]
    
    topViewImg = corners_unwarp(binaryImg, pts, 150 )
    
    
    imgWindow, left_fit, right_fit,  left_fitx, right_fitx = fit_polynomial(topViewImg)
    
    #result = search_around_poly(topViewImg, left_fit, right_fit)
    
    unwarpedImg = corners_unwarp_inverse(imgWindow, pts, 150 )
    
    left_curverad, right_curverad = measure_curvature_real(imgWindow, left_fit, right_fit)
    center = (left_fitx[(imgWindow.shape[0])-1] + right_fitx[(imgWindow.shape[0])-1])/2
    position = dstImg.shape[1]/2
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    #opImage = color_binary
    #opImage = np.dstack(( topViewImg, topViewImg, topViewImg)) * 255
    #opImage = imgWindow
    #opImage = result
    #opImage = unwarpedImg
    opImage = cv2.addWeighted(dstImg, 1, unwarpedImg, 0.4, 0)
    
    opImage = cv2.putText(opImage, "Radius of Curvature = %.2f m" % ((left_curverad+right_curverad)/2), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3,cv2.LINE_AA)
    if position > center:
        opImage = cv2.putText(opImage, "Vehicle is %.2f m right of center" % ((position-center)*xm_per_pix), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3,cv2.LINE_AA)
    elif position < center:
        opImage = cv2.putText(opImage, "Vehicle is %.2f m left of center" % ((center-position)*xm_per_pix), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3,cv2.LINE_AA)
    else:
        opImage = cv2.putText(opImage, "Vehicle is in center" , (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),3,cv2.LINE_AA)
    
    return opImage



white_output = 'Output1_6.mp4'
clip1 = VideoFileClip("../P2-Proj-CarND-Advanced-Lane-Lines/project_video.mp4")
#clip1 = VideoFileClip("../P2-Proj-CarND-Advanced-Lane-Lines/challenge_video.mp4")
#clip1 = VideoFileClip("../P2-Proj-CarND-Advanced-Lane-Lines/harder_challenge_video.mp4")
white_clip = clip1.fl_image(pipeline) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)