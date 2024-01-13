# Input path to data
data_path =  "C:/Development/task3_duetsch_sodamin/data/training/alwaysTurnRight/IMG"

# Input path to output
output_path = "C:/Development/task3_duetsch_sodamin/data/training/alwaysTurnRight/traffic_light_training_data"

# open or create a csv file in the output path
# if the file already exists, append to it

# Wait for the user to press a button called start



import os
import pandas as pd
from PIL import Image
import cv2
import numpy as np

def crop_out_traffic_lights(image, model, confidence_threshold):
    # Change image to numpy array
    image = np.array(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original = image.copy()
    results = model(image)


    # crop the image to create multiple images of traffic lights
    traffic_lights = []
    for i in range(len(results.xyxy[0])):
        xmin, ymin, xmax, ymax, confidence, class_id = results.xyxy[0][i]
        if class_id == 9 and confidence > confidence_threshold:
            
            image = original[int(ymin):int(ymax), int(xmin):int(xmax)]
            traffic_lights.append({"image" : image, "xmin" : xmin, "ymin" : ymin, "width" : xmax - xmin, "height" : ymax - ymin})
        print("adding element to traffic_lights")
    
    return traffic_lights

import torch
# Create a dataframe with the columns path, xmin, ymin, widht, height, direction, color
try:
    information = pd.read_csv(output_path + "/traffic_light_training_data.csv")
except:
    print("no file found")
    information = pd.DataFrame(columns=['path', 'xmin', 'ymin', 'width', 'height', 'direction', 'color'])   


# show each image one by one

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

start = int(input("Enter start: "))

import random

dirs = os.listdir(data_path)
length = len(dirs) - start

print("length: " + str(length))

random.seed(42)

random.shuffle(dirs)



for i in range(start, length):
    filename = dirs[i]
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # open image
        path = os.path.join(data_path, filename)
        image = Image.open(path)
        print("opening image: " + path)
        traffic_lights = crop_out_traffic_lights(image, model, confidence_threshold=0.3)
        # show image
        idx = 0
        for traffic_light in traffic_lights:

            idx += 1
            # Show the image
            zoomed = np.zeros((traffic_light["image"].shape[0]*10, traffic_light["image"].shape[1]*10, 3))
            zoomed[:,:,0] = np.kron(traffic_light["image"][:,:,0], np.ones((10,10)))
            zoomed[:,:,1] = np.kron(traffic_light["image"][:,:,1], np.ones((10,10)))
            zoomed[:,:,2] = np.kron(traffic_light["image"][:,:,2], np.ones((10,10)))
            zoomed = zoomed.astype(np.uint8)
            Image.fromarray(zoomed).show()

            # wait for commandline input
            # input direction
            direction = input("Input direction: ")
            if direction == "back":
                color = "na"
            else:  
                # input color
                color = input("Input Color: ")

            #save image to output path
            path = output_path + "/" + "image_" + str(i) + "_traffic_light_" + str(idx) + ".png"
            cv2.imwrite(path, traffic_light["image"])

            information.loc[len(information)] = [path, float(traffic_light["xmin"]), float(traffic_light["ymin"]), float(traffic_light["width"]), float(traffic_light["height"]), direction, color]

            # save to csv
            information.to_csv(output_path + "/traffic_light_training_data.csv", index=False)

            # close the image window
            Image.fromarray(zoomed).close()
    else:
        continue