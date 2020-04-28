import numpy as np
import cv2

class LaneDetector:
    '''
    Our lane detection class
    '''
    def __init__(self, image, camMatrix, distCoeffi):
        '''
        Default constructor.
        Defination of parameters.
        '''
        self.imageSize = image.shape[0:2]
        self.roi = np.int32([
            [self.imageSize[1] * 0.47, self.imageSize[0] * 0.63], 
            [self.imageSize[1] - self.imageSize[1] * 0.45, self.imageSize[0] * 0.63], 
            [self.imageSize[1] - self.imageSize[1] * 0.12, self.imageSize[0]],
            [self.imageSize[1] * 0.18, self.imageSize[0]]])
        self.warpOffset = 300
        # perspective transformation computation function
        self.computeWarpTransform()
        # calibration data
        self.camMatrix = camMatrix
        self.distCoeffi = distCoeffi
        # thresholding parameters
        self.bThreshold = [150, 255]
        self.lowerWhite = np.array([185, 185, 185])
        self.upperWhite = np.array([255, 255, 255])
        self.lowerYellow = np.array([150, 150, 0])
        self.upperYellow = np.array([255, 255, 120])

        # buffer for the last five poly fit
        self.leftFits = []
        self.rightFits = []
        # buffer for numbers of valid pixels for the last five lane search
        self.leftNumPix = []
        self.rightNumPix = []

        self.lanePixelRange = [500, 50000]
        self.laneDistRange = [250, 800]

        self.ym_per_pix = 35/600 # meters per pixel in y dimension
        self.xm_per_pix = 3.7/600 # meters per pixel in x dimension
        # state
        self.state = 'sliding window'

    def computeWarpTransform(self):
        img_size = (self.imageSize[1], self.imageSize[0])
        src = np.float32(self.roi)
        dst = np.float32([
            [self.warpOffset, 0], 
            [img_size[0]-self.warpOffset, 0], 
            [img_size[0]-self.warpOffset, img_size[1]], 
            [self.warpOffset, img_size[1]]])
        # Given src and dst points, calculate the perspective transform matrix
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.MInverse = np.linalg.inv(self.M)

    def warpImage(self, image, M):
        '''
        Warp image with perspective transformation
        '''
        img_size = (image.shape[1], image.shape[0])
        imageWarped = cv2.warpPerspective(image, M, img_size)
        return imageWarped

    def preProcess(self, image):
        '''
        Image preprocessing before lane detection.
        Steps including distortion correction, thresholding, perspective transform
        '''
        # correct image
        corrected = cv2.undistort(image, self.camMatrix, self.distCoeffi, None, None)
        # compute LAB threshold mask
        imageB = cv2.cvtColor(corrected, cv2.COLOR_RGB2Lab)[:,:,2]
        maskB = np.zeros_like(imageB)
        maskB[(imageB >= self.bThreshold[0]) & (imageB <= self.bThreshold[1])] = 255

        # compute yellow white color threshold mask
        maskW = cv2.inRange(corrected, self.lowerWhite, self.upperWhite)
        maskY = cv2.inRange(corrected, self.lowerYellow, self.upperYellow)

        # combine mask
        maskCombined = cv2.add(maskW, maskY)
        maskCombined = cv2.add(maskCombined, maskB)

        warped = self.warpImage(maskCombined, self.M)

        return warped

    def fit_poly(self, leftx, lefty, rightx, righty):
        '''
        Fit a second order polynomial to each using `np.polyfit`.
        '''
        if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
            left_fit = [1, 1, 0]
            right_fit = [1, 1, 0]
        else:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        return left_fit, right_fit

    def find_lane_pixels(self, binary_warped):
        '''
        Scliding window fine lane.
        '''
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        nwindows = 10
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 40

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
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current -margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
            
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) 
                & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if(len(good_left_inds) > minpix):
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if(len(good_right_inds) > minpix):
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # the out put image is for debugging purpose only. Final lane is draw
        # from fitted lane.
        return leftx, lefty, rightx, righty, out_img

    def find_lane_init(self, binary_warped):
        '''
        Scliding window fine lane.
        '''
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(binary_warped)

        left_fit, right_fit = self.fit_poly(leftx, lefty, rightx, righty)

        numLeft = len(leftx)
        numRight = len(rightx)
        dist = self.leftRightDist(binary_warped, left_fit, right_fit)
        
        valid = True
        if numLeft < self.lanePixelRange[0] or numLeft > self.lanePixelRange[1] or \
           numRight < self.lanePixelRange[0] or numRight > self.lanePixelRange[1] or \
           dist < self.laneDistRange[0] or dist > self.laneDistRange[1]:
            left_fit = self.leftFits[np.argmax(self.leftNumPix)]
            right_fit = self.rightFits[np.argmax(self.rightNumPix)]
            valid = False

        ## Visualization ##
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        leftPoints = np.int32(np.dstack((left_fitx, ploty)))
        rightPoints = np.int32(np.dstack((right_fitx, ploty)))
        cv2.polylines(out_img, leftPoints, False, color=[255, 255, 0], thickness = 5)
        cv2.polylines(out_img, rightPoints, False, color=[255, 255, 0], thickness = 5)

        bw = np.dstack((binary_warped, binary_warped, binary_warped))
        out_img = cv2.addWeighted(bw, 1, out_img, 1, 0)

        text                   = 'Left pixel: ' + str(numLeft) + ' Right pixel: ' + str(numRight) + ' Dist: ' + str(dist)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20,20)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        cv2.putText(out_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
        ## End visualization steps ##

        # The out put image is for debugging purpose only. Final lane is draw
        # from fitted lane.
        return valid, left_fit, right_fit, numLeft, numRight, dist, out_img

    def search_around_poly(self, binary_warped, left_fit, right_fit):
        '''
        Search lane around the polynominal.
        '''
        # HYPERPARAMETER
        # Choose the width of the margin around the previous polynomial to search
        # The quiz grader expects 100 here, but feel free to tune on your own!
        margin = 100

        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        ### TO-DO: Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        ### Hint: consider the window areas for the similarly named variables ###
        ### in the previous quiz, but change the windows to our new search area ###
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                        left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                        right_fit[1]*nonzeroy + right_fit[2] + margin)))
        
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit new polynomials
        left_fit, right_fit = self.fit_poly(leftx, lefty, rightx, righty)

        numLeft = len(leftx)
        numRight = len(rightx)
        dist = self.leftRightDist(binary_warped, left_fit, right_fit)

        valid = True
        if numLeft < self.lanePixelRange[0] or numLeft > self.lanePixelRange[1] or \
           numRight < self.lanePixelRange[0] or numRight > self.lanePixelRange[1] or \
           dist < self.laneDistRange[0] or dist > self.laneDistRange[1]:
            left_fit = self.leftFits[np.argmax(self.leftNumPix)]
            right_fit = self.rightFits[np.argmax(self.rightNumPix)]
            valid = False

        ## visualization
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
                
        leftPoints = np.int32(np.dstack((left_fitx, ploty)))
        rightPoints = np.int32(np.dstack((right_fitx, ploty)))
        
        cv2.polylines(out_img, leftPoints, False, color=[255, 0, 0], thickness = 10)
        cv2.polylines(out_img, rightPoints, False, color=[0, 0, 255], thickness = 10)

        text                   = 'Left pixel: ' + str(numLeft) + ' Right pixel: ' + str(numRight) + ' Dist: ' + str(dist)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (20,20)
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        cv2.putText(out_img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

        bw = np.dstack((binary_warped, binary_warped, binary_warped))
        out_img = cv2.addWeighted(bw, 1, out_img, 1, 0)
        ## End visualization steps ##

        # The out put image is for debugging purpose only. Final lane is draw
        # from fitted lane.
        return valid, left_fit, right_fit, numLeft, numRight, dist, out_img

    def leftRightDist(self, image, left_fit, right_fit):
        '''
        Compute the average distance between two lanes.
        '''
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        return sum(right_fitx - left_fitx) / len(ploty)

    def measureCurAndPos(self, image, left_fit, right_fit):
        '''
        Estimate curvature and position of the car.
        '''
        y_eval = image.shape[0] - 1

        # convert into real world coordinate
        left_fit_world = []
        left_fit_world.append(left_fit[0] * self.xm_per_pix / (self.ym_per_pix**2))
        left_fit_world.append(left_fit[1] * self.xm_per_pix / self.ym_per_pix)
        left_fit_world.append(left_fit[2])

        right_fit_world = []
        right_fit_world.append(right_fit[0] * self.xm_per_pix / (self.ym_per_pix**2))
        right_fit_world.append(right_fit[1] * self.xm_per_pix / self.ym_per_pix)
        right_fit_world.append(right_fit[2])

        left_curverad = (1 + (2 * left_fit_world[0] * y_eval + left_fit_world[1])**2)**1.5 / np.abs(2 * left_fit_world[0])
        right_curverad = (1 + (2 * right_fit_world[0] * y_eval + right_fit_world[1])**2)**1.5 / np.abs(2 * right_fit_world[0])

        position = image.shape[1] / 2 - (left_fit[0] * (y_eval ** 2) + left_fit[1] * y_eval + left_fit[2] + 
            right_fit[0] * (y_eval ** 2) + right_fit[1] * y_eval + right_fit[2]) / 2
        position = position * self.xm_per_pix

        return left_curverad, right_curverad, position

    def processImage(self, image):
        '''
        Perform the main pipeline and return result image.
        '''
        # process image detect lane
        preProcessed = self.preProcess(image)

        current_state = self.state

        if current_state == 'sliding window':
            valid, left_fit, right_fit, left_num, right_num, dist, laneImage = \
                self.find_lane_init(preProcessed)

            if valid:
                self.leftFits.append(left_fit)
                self.leftNumPix.append(left_num)
                self.rightFits.append(right_fit)
                self.rightNumPix.append(right_num)
                self.state = 'polylines neighor'

        if current_state == 'polylines neighor':
            left_fit = self.leftFits[-1]
            right_fit = self.rightFits[-1]

            valid, left_fit, right_fit, left_num, right_num, dist, laneImage = \
                self.search_around_poly(preProcessed, left_fit, right_fit)

            if not valid:
                self.state = 'sliding window'
            else:
                self.leftFits.append(left_fit)
                self.rightFits.append(right_fit)
                self.leftNumPix.append(left_num)
                self.rightNumPix.append(right_num)

        if(len(self.leftFits) > 5):
            self.leftFits.pop(0)
            self.rightFits.pop(0)
            self.leftNumPix.pop(0)
            self.rightNumPix.pop(0)

        left_curverad, right_curverad, position = \
            self.measureCurAndPos(preProcessed, left_fit, right_fit)

        ## final visualization
        start = int(preProcessed.shape[0] * 0.2)
        end = preProcessed.shape[0]-1
        ploty = np.linspace(start, end, end - start + 1)

        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.zeros_like(image)
        window_img = np.zeros_like(out_img)

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        out_img = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
                
        leftPoints = np.int32(np.dstack((left_fitx, ploty)))
        rightPoints = np.int32(np.dstack((right_fitx, ploty)))
        
        cv2.polylines(out_img, leftPoints, False, color=[255, 0, 0], thickness = 10)
        cv2.polylines(out_img, rightPoints, False, color=[0, 0, 255], thickness = 10)

        out_img = self.warpImage(out_img, self.MInverse)
        out_img = cv2.addWeighted(image, 1, out_img, 1, 0)

        #text                   = 'Left pixel: ' + str(left_num) + ' Right pixel: ' + str(right_num) + ' Dist: ' + str(dist)
        text = 'Left Curvature: %d(meter) Right Curvature: %d(meter) \nPosition: %.2f(meter)' % (left_curverad, right_curverad, position)
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        fontScale              = 1
        fontColor              = (255,255,255)
        lineType               = 2
        text = 'Left Curvature: %d(meter)' % (left_curverad)
        cv2.putText(out_img, text, (100,100), font, fontScale, fontColor, lineType)
        text = 'Right Curvature: %d(meter)' % (right_curverad)
        cv2.putText(out_img, text, (100,140), font, fontScale, fontColor, lineType)
        text = 'Position: %.2f(meter)' % (position)
        cv2.putText(out_img, text, (100,180), font, fontScale, fontColor, lineType)

        return out_img
