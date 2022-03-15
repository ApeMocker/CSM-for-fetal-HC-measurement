"""
This script is used for extracting edge images from prediction results.
Requirement: predict.py has been executed so prediction results exist.
"""
import os
import cv2
from modules import mcc_edge
import shutil

# Folder of model predictions
input_folder = '../results/predictions/'

#  Folder to save postprocess results
edge_folder = '../results/predictions_edge/'
if os.path.isdir(edge_folder):
    shutil.rmtree(edge_folder)
os.makedirs(edge_folder)

# Extract fetal contour
dirs = os.listdir(input_folder)
for i in range(len(dirs)):
    print('Extracting max connect component edge: Image = %d / %d' % (i + 1, len(dirs)))
    img_name = dirs[i]
    img_path = input_folder + img_name

    img = cv2.imread(img_path, 0)
    edge_img = mcc_edge(img)

    save_path = edge_folder + img_name
    cv2.imwrite(save_path, edge_img)

