import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import json
from pycocotools import mask
from skimage import measure, img_as_ubyte



image_path_annon = './data/Image/' # Image Directory
mask_path_annon =  './data/Mask/' # Binary Mark Directory
image_path_clean = './data/Clean/Image/' # Only Required If Present in Different Directories, Otherwise leave blank

out_path = './data/image_format/' # 2D image slices will be saved here
if not os.path.exists(out_path):
    os.makedir(out_path)

df = pd.read_csv("./meta.csv") # Metadata CSV path


#Initializing Image and Annotation Lists
images_test = []
images_train= []
images_val = []
images_test_bn = []
images_train_bn = []
images_val_bn = []
images_test_ml = []
images_train_ml = []
images_val_ml = []
annotations = []

count = 1 # For progress Print
for i in df.itertuples(index=False):
    print("Processing Annoted Data: ", count, "/", len(df), end = "\r")

    count = count + 1

    img = np.load(image_path_annon + str(i[5]) + '.npy')
    imageio.imwrite(out_path + i[5][15:] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )


    img = np.load(mask_path_annon + str(i[6]) + '.npy')

    ground_truth_binary_mask = img

    fortran_ground_truth_binary_mask = np.asfortranarray(ground_truth_binary_mask)
    encoded_ground_truth = mask.encode(fortran_ground_truth_binary_mask)
    ground_truth_area = mask.area(encoded_ground_truth)
    ground_truth_bounding_box = mask.toBbox(encoded_ground_truth)
    contours = measure.find_contours(ground_truth_binary_mask, 0.5)

    name = str(i[5])
    s_id = int( name[15:19] + name[22:25] + name[31:])

    image = {
            "id": s_id,
            "width": img.shape[1],
            "height": img.shape[0],
            "file_name": "{}.jpg".format(str(i[5][15:]))
    }

    annon_class = 0

    if i[7] >= 3:
    	annon_class = 2
    elif i[7] > 0:
    	annon_class = 1


    annotation = {
            "segmentation": [],
            "area": ground_truth_area.tolist(),
            "iscrowd": 0,
            "image_id": s_id,
            "bbox": ground_truth_bounding_box.tolist(),
            "category_id": annon_class,
            "id": s_id
        }

    for contour in contours:
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        annotation["segmentation"].append(segmentation)


    if annon_class > 0:
        annotations.append(annotation)

    if i[11] == 'Train':
    	if(annon_class == 1):
    		images_train_bn.append(image)
    	if(annon_class == 2):
    		images_train_ml.append(image)
    elif i[11] == 'Test':
    	  	images_test.append(image)
    else:
    	if(annon_class == 1):
    		images_val_bn.append(image)
    	if(annon_class == 2):
    		images_val_ml.append(image)


# Balancing Malignant and Benign slices for train and val sets

min_cnt = min(len(images_train_ml), len(images_train_bn))
images_train.extend(images_train_ml[:min_cnt])
images_train.extend(images_train_bn[:min_cnt])

min_cnt = min(len(images_val_ml), len(images_val_bn))
images_val.extend(images_val_ml[:min_cnt])
images_val.extend(images_val_bn[:min_cnt])


if(image_path_clean != ''):
    df = pd.read_csv("./clean_meta.csv") # for Clean data
    count = 1
    for i in df.itertuples(index=False):
        print("Processing Clean Data: ", count, "/", len(df), end = "\r")
        count = count + 1

        img = np.load(image_path_clean + i[5] + '.npy')
        imageio.imwrite(out_path + i[5][15:] + '.jpg', img_as_ubyte(img/max(np.max(img), -1*np.min(img) ) ) )

        name = i[5]
        s_id = int( name[15:19] + name[22:25] + name[31:])

        image = {
                "id": s_id,
                "width": img.shape[1],
                "height": img.shape[0],
                "file_name": "{}.jpg".format(i[5][15:])
        }

        if i[11] == 'Train':
        	images_train.append(image)
        elif i[11] == 'Validation':
        	# images_test.append(image)
        	images_val.append(image)
        # else:
        # 	images_val.append(image)


# Dumping Image and annotation data into 3 json files for Train, Validation and Test
json_object1 = json.dumps(annotations, indent=4)

with open("annotations.json", "w") as outfile:
    outfile.write(json_object1)

json_object2 = json.dumps(images_train, indent=4)

json_object3 = json.dumps(images_test, indent=4)
json_object4 = json.dumps(images_val, indent=4)

with open("images_train.json","w") as outfile:
    outfile.write(json_object2)

with open("images_test.json","w") as outfile:
    outfile.write(json_object3)

with open("images_val.json","w") as outfile:
    outfile.write(json_object4)

f2data = ""
with open('annotations.json') as outfile:
    f2data= '\n' + outfile.read()

with open('images_test.json','a+') as outfile:
    outfile.write(f2data)

with open('images_train.json','a+') as outfile:
    outfile.write(f2data)

with open('images_val.json','a+') as outfile:
    outfile.write(f2data)
