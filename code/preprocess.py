"""
Data preprocessing, including images cropping, training set and verification set partition,
and data augumentation.
Note: This scirpt need to be executed first to prepare training and validation set.
"""
import cv2
import os
import shutil
import random
import pandas as pd
from modules import img_crop, ellip_fill, img_augumentation

# Clear '../data/' folder
data_folder = '../data/'
dirs = os.listdir(data_folder)
for f in dirs:
    f_path = data_folder + f
    if f != 'HC18_dataset':
        if os.path.isdir(f_path):
            shutil.rmtree(f_path)
        else:
            os.remove(f_path)

# Original HC18 training_set data
HC18_training_set_folder = '../data/HC18_dataset/training_set/'

# To save preprocess images
train_img_folder = '../data/train/images/'
os.makedirs(train_img_folder)

# To save preprocess labels
train_label_folder = '../data/train/labels/'
os.makedirs(train_label_folder)

# Croping images and labels in training_set to（512，768）, then divide them to images and labels
dirs = os.listdir(HC18_training_set_folder)
for i in range(len(dirs)):
    print("Dividing images and labels: %d / %d" % (i+1,len(dirs)))
    img_name = dirs[i]
    img_path = HC18_training_set_folder + img_name
    img = cv2.imread(img_path,0)

    # Crop images
    crop_img = img_crop(img)

    # Save images
    if img_name[-14:] == 'Annotation.png':
        save_path = train_label_folder + img_name
    else:
        save_path = train_img_folder + img_name

    cv2.imwrite(save_path, crop_img)

# Fill ellipse to create segmentation mask for model training
ellip_fill(train_label_folder, train_label_folder)

# Divide data for train and val from '../data/train/images' and '../data/train/labels' with a ratio of 8:2
val_img_folder = '../data/validation/images/'
val_label_folder = '../data/validation/labels/'
os.makedirs(val_img_folder)
os.makedirs(val_label_folder)

dirs = os.listdir(train_img_folder)
l = len(dirs)
val_l= round(0.2*l)
random.shuffle(dirs)

for i in range(val_l):
    print("Dividing the train and val set: %d / %d" % (i+1, val_l))
    img_name = dirs[i]
    label_name = img_name[:-4] + '_Annotation.png'

    img_path = train_img_folder + img_name
    label_path = train_label_folder + label_name

    img_save_path = val_img_folder + img_name
    label_save_path = val_label_folder + label_name

    img = cv2.imread(img_path, 0)
    label = cv2.imread(label_path, 0)

    cv2.imwrite(img_save_path, img)
    cv2.imwrite(label_save_path, label)

    # Remove validation images and labels from '../data/train/images' and '../data/train/labels'
    os.remove(img_path)
    os.remove(label_path)

# Create .csv file for validation set to save pixel size and HC
pixel_size_hc_file = '../data/HC18_dataset/training_set_pixel_size_and_HC.csv'
val_pixel_size_hc_save = '../data/validation/val_set_pixel_size_and_HC.csv'
p_size_hc = pd.read_csv(pixel_size_hc_file)

fn = p_size_hc['filename'].values
ps = p_size_hc['pixel size(mm)'].values
hc = p_size_hc['head circumference (mm)'].values

new_psize_hc = pd.DataFrame({'pixel size(mm)':ps,'head circumference (mm)':hc},index=fn)

val_psize_hc = pd.DataFrame(new_psize_hc,index=dirs[0:val_l])

val_fn = val_psize_hc.index
val_ps = val_psize_hc['pixel size(mm)'].values
val_hc = val_psize_hc['head circumference (mm)'].values

val_psize_hc = pd.DataFrame({'filename':val_fn, 'pixel size(mm)':val_ps, 'head circumference (mm)':val_hc})
val_psize_hc.to_csv(val_pixel_size_hc_save,index=False)

# Data augumentation in train set
train_augu_img_folder = '../data/train/augu_images/'
train_augu_label_folder = '../data/train/augu_labels/'
os.makedirs(train_augu_img_folder)
os.makedirs(train_augu_label_folder)

# Image augumentation
img_augumentation(train_img_folder,train_augu_img_folder,data="image")

# Label augumentation
img_augumentation(train_label_folder, train_augu_label_folder, data="label")

# Finished
print("Data preprocessing finish!")






