# Package Import
import cv2
from skimage import measure
from skimage.measure import compare_ssim, compare_mse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imutils.object_detection import non_max_suppression
import argparse
import time
from PIL import Image

#package version numbers included in supporting documentation

ap = argparse.ArgumentParser() #initialize arg parser
ap.add_argument("-v", "--video", type=str,
    help="path to input video") #accepts video file path as parameter
args = vars(ap.parse_args()) #argparse parse input parameter

vid_name=args["video"] #captures input video as vid_name variable
#vid_name will be input for detect_duplicates function
start = time.time() #start timer 

def is_info_card(image,frame_count):
    'Determines if frame is part of either static intro or end card sequence with text box detection.'
    #accepts image that is currently being processed in detect_duplicates() and its corresponding frame number

    ##If frame is in a sequence of frames greater than two we know deterministically it is a card.
    ##This is because duplication only happens in pairs
    
    #compare image's previous frame and subsequent frame
    
    #if scores indicate similarity, we must check if current frame is part of an information card
    
    i = frame_count #change variable name
    info_card = False #initialize info_card true false decision
    curr_text_box_count = 0 #initialize current images text box count

    if frame_count == 0: #if first frame we return false and zero values
        return info_card, curr_text_box_count
    #else, we compare an images previous and subsequent frames to each other
    prev_image = cv2.imread('frame'+str(i-1)+'.jpg') #vectorize previous image
    next_image = cv2.imread('frame'+str(i+1)+'.jpg') #vectorize subsequent image
            
    #convert images to grayscale
    #use sci-image to normalize image vectors on gray scale
    grey_prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    grey_next_image = cv2.cvtColor(next_image, cv2.COLOR_BGR2GRAY) 
        
    #compute mean squared error and mse for previous and subsequent images
    mse,sSim = compare_images(grey_prev_image,grey_next_image)
        
    if mse <.22 or sSim > .99: #Chosen thresholds for whether previous and subsequent images are considered similar
        similar = True
        #compare by text box count with previous image
        #if similar score is true, the current image is part of info sequence

        #if not similar we begin calculation to determine if at the tail end of a sequence
        #if an image the final image in a sequence it will have the same number text boxes as the previous text box
        #if an image does not have any text boxes it is not part of an info sequence
        ##we rely on this probabilistic technique that compares text between sequential images

        prev_text_box_count = count_text_box('frame'+str(i-1)+'.jpg')
        
        curr_text_box_count = count_text_box('frame'+str(i)+'.jpg')
        
        if prev_text_box_count == curr_text_box_count:
            similar = True
            duplicate = False
            info_card = True #non real duplicate established as info card due to amount of and presence of text boxes. 
        if curr_text_box_count == 0:
            duplicate = True
            info_card = False
    
    #if previous and subsequent images are not similar we indicate that the image is not part of an info sequence.
    #text box count will remain at zero as any text detection will be computation expensive
    #additional text box counting will also not be informational for the purposes of this application
    else:
        similar = False
        duplicate = False
        info_card = False
        
    return info_card, curr_text_box_count



##Begins text detection functionality. 
##The following function is called only within is_info_card() if an image's prevous and subsequent frame are determined similar
##Disclaimer: Borrowed function. See documentation for citation on text detection functionality.
def count_text_box(image):
    'Detects Text Boxes in input frame and returns count of text items.'

    #read in image
    image = cv2.imread(image)
    orig = image.copy() #create copy
    (H, W) = image.shape[:2] #store height and weight dimensions

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    #requires multiples of 32.
    ##Default Dimensions of 512x288 allow for videos with 16:9 aspect ratio only to be resized without distortion.
    
    (newW, newH) = (512, 288) ##both multiples of 32 and retains common 16:9 video aspect ratio
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # defines the two output layer names for the EAST detector model
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",   # the first is the output probabilities
        "feature_fusion/concat_3"]     # the second can be used to derive the bounding box coordinates of text
    
    # loads the pre-trained EAST text detector
    net = cv2.dnn.readNet('frozen_east_text_detection.pb') #ensure this is within the same root directory as the application
    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()
    
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < .5:
                continue

            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
    
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    text_item_count = 0 #initialize text box count score that will be returned

    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        text_item_count = text_item_count + 1 #count each box as it is drawn
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #print('Box count', box_count)
    #if box_count > 0: text_detected = True
    #else: text_detected = False
    
    return text_item_count #returns total number of boxes that are drawn

def compare_images(imageA, imageB): #this function will be called in the detect_duplicates() function and the is_info_card() function
    'Computes the mean squared error and structural similarity between two images.'
    #computations use measures from the sci-image library
    mse = compare_mse(imageA, imageB) ##lower values indicate similarity
    ##ssim higher values indicate similarity, but normalized bewteen 0-1
    ssim = measure.compare_ssim(imageA, imageB) 
    return mse,ssim
    

#this is the main function that the application will
#it in turn will call other functions as needed
def detect_duplicates(vid_name):
    'Creates image frames from input video and detects erroneous duplications.'
    
    # summary_chart = pd.DataFrame(columns=['Frame ID','mse','sSim','Text Boxes','In Info Sequence','Duplicate Detected','Comparison Frame'])
    #summary chart commented out. This dataframe is only used for QA purposes.

    ##See citation in documentation for reading in video as frames
    vidcap = cv2.VideoCapture(vid_name)
    success,image = vidcap.read()
    
    count = 0
    success = True
    
    while success: #
        cv2.imwrite("frame%d.jpg" % count, image) #writes frame as jpeg image in working directory
        success,image = vidcap.read()
      #  print('Read a new frame: ', success)
        count += 1
        
        #if count > 5: #option to limit amount of frames in a vid 
         #  break   #useful for QA purposes
   # print(count)
    dup_count = 0 #initialize duplicate counter
 
    for i in range(count):
        dup = 'No' #initialize determinatino of whether frame is duplicate
        if i == 0: #first frame will have no previous image to compare, skip to next frame
            continue
        else:
            #load the sequential images
  
            prev_image = cv2.imread('frame'+str(i-1)+'.jpg') #vectorize frame image
            curr_image = cv2.imread('frame'+str(i)+'.jpg') #vectorize frame image
             
            #convert images to grayscale
            grey_prev_image = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
            grey_curr_image = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        
        mse,sSim = compare_images(grey_prev_image,grey_curr_image) #compute mse and structured similarity measures
        
        #checks if frame is part of an intro or end information card sequence.
        
        frame_c = i-1 #gets frame count input
        
        if mse < .22 or sSim > .99:
            in_info_sequence, text_boxes = is_info_card(prev_image,frame_c) #calls info card fucntion to check if similar frames part of an info sequence
            if in_info_sequence == True: #if both similar and info sequence detected, we say it is not an erroneously duplicated frame
                dup = 'No'
            else:
                dup_count += 1 #if similar and NOT determined to be in an info sequence we say it is an erroneously duplicated frame
                dup = 'Yes'
       # summary_chart.loc[i] = ['frame'+str(i-1)+'.jpg',mse,sSim,text_boxes,in_info_sequence,dup,'frame'+str(i)+'.jpg']
        #appends record to summary chart

    end = time.time() #end function calculation time
    print('Total Duplicate Count:', dup_count) # return erroneously duplicated frame count
    print('Time Elapse:', end-start) # return time elapsed

    #return summary_chart[40:88] #return portions of summary_chart dataframe to view comparison scores

detect_duplicates(vid_name) #call main function