# Package Import
import cv2
from skimage import measure
from skimage.measure import compare_ssim, compare_mse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from imutils.object_detection import non_max_suppression
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

def compare_images(imageA, imageB): #this function will be called in the detect_duplicates() function and the is_info_card() function
    'Computes the mean squared error and structural similarity between two images.'
    #computations use measures from the sci-image library
    #mse = compare_mse(imageA, imageB) ##lower values indicate similarity
    ##ssim higher values indicate similarity, but normalized bewteen 0-1
    ssim = measure.compare_ssim(imageA, imageB) 
    return ssim

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
        
        sSim = compare_images(grey_prev_image,grey_curr_image) #compute mse and structured similarity measures
        
        #checks if frame is part of an intro or end information card sequence.
        
        frame_c = i-1 #gets frame count input
        
        #if mse < .22 or sSim > .99:
        if sSim > .99:
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