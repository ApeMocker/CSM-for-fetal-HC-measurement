"""
Prediction on validation set.
"""
import torch
import os
import shutil
from modules import CSM, predict

# Clear '../results/' folder
data_folder = '../results/'
dirs = os.listdir(data_folder)
for f in dirs:
    f_path = data_folder + f
    if os.path.isdir(f_path):
        shutil.rmtree(f_path)
    else:
        os.remove(f_path)

# Validation images folder
input_folder = '../data/validation/images/'

# Prediction folder for results saving
predict_folder = '../results/predictions/'
if os.path.isdir(predict_folder):
    shutil.rmtree(predict_folder)
os.makedirs(predict_folder)

# Load the network
net_dict_file = '../models/test_model.pth'
net = CSM()
net.load_state_dict(torch.load(net_dict_file))

# Predict
predict(net,input_folder,predict_folder,device='cuda')









