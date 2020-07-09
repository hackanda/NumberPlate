import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Import all necessary libraries.
import os
import cv2
import numpy as np
import sys
import json
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# change this property
NOMEROFF_NET_DIR = os.path.abspath('../')

# specify the path to Mask_RCNN if you placed it outside Nomeroff-net project
MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')

sys.path.append(NOMEROFF_NET_DIR)


# Import license plate recognition tools.
from NomeroffNet import  filters
from NomeroffNet import  RectDetector
from NomeroffNet import  TextDetector
from NomeroffNet import  OptionsDetector
from NomeroffNet import  Detector
from NomeroffNet import  textPostprocessing
from NomeroffNet import  textPostprocessingAsync

# Initialize npdetector with default configuration file.
nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
nnet.loadModel("latest")

rectDetector = RectDetector()

optionsDetector = OptionsDetector()
optionsDetector.load("latest")

# Initialize text detector.
textDetector = TextDetector({
    "eu_ua_2004_2015": {
        "for_regions": ["eu_ua_2015", "eu_ua_2004"],
        "model_path": "latest"
    },
    "eu_ua_1995": {
        "for_regions": ["eu_ua_1995"],
        "model_path": "latest"
    },
    "eu": {
        "for_regions": ["eu"],
        "model_path": "latest"
    },
    "ru": {
        "for_regions": ["ru", "eu-ua-fake-lnr", "eu-ua-fake-dnr"],
        "model_path": "latest" 
    },
    "kz": {
        "for_regions": ["kz"],
        "model_path": "latest"
    },
    "ge": {
        "for_regions": ["ge"],
        "model_path": "latest"
    }
})

max_img_w = 1600
async def print_predictions(img_path , cordinates): 
    cordinates = cordinates.strip().split(' ')
    x ,y , w , h , region_name = cordinates
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)

    img = cv2.imread(img_path)

    # Getting the cropped Image
    crop_img = img[x:w , y:h]
    cv2.imshow('cropped Image' , crop_img)

    img_w = img.shape[1]
    img_h = img.shape[0]
    img_w_r = 1
    img_h_r = 1
    if img_w > max_img_w:
        resized_img = cv2.resize(img, (max_img_w, int(max_img_w/img_w*img_h)))
        img_w_r = img_w/max_img_w
        img_h_r = img_h/(max_img_w/img_w*img_h)
    else:
        resized_img = img

    NP = nnet.detect([resized_img]) 

    # Generate image mask.
    cv_img_masks = await filters.cv_img_mask_async(NP)

    # Detect points.
    arrPoints = await rectDetector.detectAsync(cv_img_masks, outboundHeightOffset=0, fixGeometry=True, fixRectangleAngle=10)
    #print(arrPoints)
    arrPoints[..., 1:2] = arrPoints[..., 1:2]*img_h_r
    arrPoints[..., 0:1] = arrPoints[..., 0:1]*img_w_r

    # # cut zones
    zones = await rectDetector.get_cv_zonesBGR_async(img, arrPoints)
    toShowZones = await rectDetector.get_cv_zonesRGB_async(img, arrPoints)
    for zone, points in zip(toShowZones, arrPoints):
        plt.axis("off")
        plt.imshow(zone)
        plt.show()

    # # find standart
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)
    #print(regionNames)
    #print(countLines)
    print ("State name : ", regionNames)

    # # find text with postprocessing by standart  
    textArr = textDetector.predict(zones, regionNames, countLines)
    textArr = await textPostprocessingAsync(textArr, regionNames)
    #print(textArr)
    print ("Detected licence number : ", textArr)

    left_upper_corner  = arrPoints[0][0]
    right_upper_corner = arrPoints[0][1]
    right_lower_corner = arrPoints[0][2]
    left_lower_corner  = arrPoints[0][3]

    print ("Lefr upper corner coordinates  : ",left_upper_corner)
    print ("left lower corner coordinates  : ",left_lower_corner)
    print ("Right upper corner coordinates : ",right_upper_corner)
    print ("Right lower corner coordinates : ",right_lower_corner)
   
async def main(img_path , cordinates):
    await print_predictions(img_path , cordinates)


if __name__ == "__main__":
    import asyncio
    import argparse 
    import os.path
    from os import path

    print('\n\n')
    parser = argparse.ArgumentParser(description ='Search some files') 
    
    parser.add_argument('--image_dir',
                        action ='store', 
                        default ='', help ='image_dir')  

    parser.add_argument('--cordinates',
    					action = 'store',
    					default='',help = 'Enter cordinates')
 
    args = parser.parse_args() 


    img_path = args.image_dir
    cordinates = args.cordinates
    if  path.exists("guru99.txt"): 
        print("Image path: ", img_path)

        loop = asyncio.get_event_loop()
        loop.run_until_complete(main(img_path , cordinates))
        loop.close()
        print('\n\n')
    else: 
        print('Please enter a valid Path to an image')
    



