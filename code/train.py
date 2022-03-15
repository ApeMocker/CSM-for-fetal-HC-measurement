"""
Train CSM model on training set and save.
"""
from modules import CSM, train_model,HcDataset
from torch.utils.data import DataLoader

input_folder = '../data/train/augu_images/'
label_folder = '../data/train/augu_labels/'

#  Training data
train_data = HcDataset(input_folder,label_folder,[192,128])
dataloader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)

# CSM model
net = CSM()

# File to save the trained model
save_model_name='../models/test_model.pth'

# Model training
train_model(model=net,dataloader=dataloader,epoches=10,lr=0.001,device='cuda',save_model_name=save_model_name)









