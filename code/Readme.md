# Introduction

This project is created for automatic fetal head circumference(HC) measurement from 2D ultrasound images using a deep learning based method. A very lightweight deep convolutional neural network is proposed for efficient and accurate fetal head segmentation and  then post-processing, including morphological processing and least-squares ellipse fitting, was applied to obtain the fetal HC.  

![image-20220228212702366](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220228212702366.png)



# Algorithm flowchart

Flow chart of automatic fetal HC measurement from two-dimensional ultrasound image in this work is shown below. (a) The original fetal ultrasound image; (b) the predicted fetal head segmentation from deep model CSM; (c) the fetal head contour extracted by the use of morphological processing; (d) the target ellipse obtained after using least-squares ellipse fitting. 

![image-20220303165305610](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220303165305610.png)



# Deep model architecture

The very lightweight architecture of the proposed Convolutional Segmentation Machine (CSM) for fetal ultrasound image segmentation is shown below. 



![深度神经网络](C:\Users\69115\Desktop\软件著作权\图例\深度神经网络.png)

# Algorithm performance

The performance of the proposed method on **HC18 test set** is shown below. Four most widely  used indices for accuracy evaluation including  dice similarity coefficient (DSC) , absolute difference( ADF), difference(DF), and hausdorff distance(HD)  and two indices for efficiency evaluation including inference time and model parameters(Params) are computed.

![image-20220303171552640](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220303171552640.png)

# Requirements

This project is totally built  by python and several packages are required as below. It should be noted that all of our codes are tested and run successfully on Windows system.

* python = 3.6
* pytorch = 1.10.2
* opencv-python = 3.4.2
* pandas = 1.1.5
* scikit-learn = 0.24.



# Build directories

First of all, you should choose a directory for this project and create some empty directories as below.

```
.
├─codes
├─data
│  └─HC18_dataset
│
├─models
└─results
```



Then copy  the python scripts in this repository into "/codes/", and download HC18(https://hc18.grand-challenge.org/). dataset into "/data/HC18_dataset/".




# Description of directories and files

The directories and files contained in this project are described below. The project directory path is represented as **".../"**. 

* **.../codes/** : Directory to save the codes. All the codes are with brief  comments.
  * **modules.py :** Components for other script file.
  * **preprocess.py :** Perform data preprocessing, including  images cropping, training set and validation set partition and data augumentation.
  * **train.py :** Used for training deep model on training set and save the model.
  * **predict.py :** Used for predicting on validation set.
  * **postprocess.py :** Used for extracting edge images from prediction results.
  * **ellip_fit.py :** Used for ellipse fitting from edge images to create predicted fetal head contours.
  * **visualization.py :** Prediction results visualization.
  * **evalution.py** : Used for model evalutation. Four evaluation metrics including mean difference, mean absolule difference, mean dice coeffient and mean hausdorff distance are calculated.
* **.../data/** : Directory to save original HC18 dataset and preprocess results.
  * **HC18_dataset/ :** Original HC18 dataset. You can  download  the dataset  from https://hc18.grand-challenge.org/.
  * **train/ :** Directory to save train data. *NOTE: This directory only be created after **".../codes/preprocess.py"** is executed.* 
  * **validation/ :** Directory to save validation data. *NOTE: This directory only be created after **".../codes/preprocess.py"** is executed.* 
* **.../models/** : Directory to save trained model.
* **.../results/** : Directory to save the results of prediction, visualization and evaluation. This directories is empty originally unless several corresponding python scripts are executed.
  * **predictions/** : Model prediction results on validation set. *NOTE: This directory only be created after **".../codes/predict.py"** is executed.* 
  * **predictions_edge/** : Postprocessing results to create edge images from prediction results. *NOTE: This directory only be created after **".../codes/postrocess.py"** is executed.* 
  * **visualizations/** : Visualization results on validation set.  *NOTE: This directory only be created after **".../codes/ellip_fit.py"** and  **".../codes/visualization.py"** is executed.* 
  * **ellip_params.csv** : Ellipse parameters using ellipse fitting on predicted edge images. *NOTE: This file only be created after **".../codes/ellip_fit.py"** is executed.* 
  * **eval_results.csv** : Evaluation results of trained model on validation set. *NOTE: This file only be created after **".../codes/evaluation.py"** is executed.* 

# Usages

## Preprocessing 

The original HC18 dataset need to be preprocessed firstly before training and testing. Only the data of training_set is used because the labels of HC18 test set are not provided. We firstly crop the images to the same size of 768*512. Then divide the data of original training_set with the ratio of 8:2 to create train and validation set in this project. Finally, several data augumentation processes are performed to obtain more data for model training. 

```python
python preprocess.py
```



## Training

Create a deep neural network for fetal ultrasound segmentation and then train the network using augumentation training set from preprocessing. The trained model will be save in **".../models/"**

```pythpon
python train.py
```



## Prediction

Using the trained mode to predict on validation set. The prediction results will be saved in ".../results/predictions/". Illustration of prediction result is as below.

```python
python predict.py
```



![image-20220228231357125](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220228231357125.png)

## Postprocessing

The prediction results need to be postprocessing to obtain final head circumference. First, max connected component extraction and edge detection are perform to obtain the fetal contour. The results will be saved in ".../results/predictions_edge"

```python
python postprocess.py
```

![image-20220228231242836](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220228231242836.png)

## Ellipse fitting

Least square method is used to fit edge images into ellipse and the ellipse parameters are saved in ".../results/ellip_params.csv". Five ellipse parameters are defined as below.

```python
python ellip_fit.py
```

![image-20220228232930740](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220228232930740.png)

## Visualization

Ellipse fitting result can be visually presented by executed ".../codes/visualization.py".

```python
python visualization.py
```

![image-20220228232304020](C:\Users\69115\AppData\Roaming\Typora\typora-user-images\image-20220228232304020.png)

## Evaluation

Four evaluation metrics can be calculated includingmean difference, mean absolule difference, mean dice coeffient and mean hausdorff distance. The results will be saved in **".../results/eval_results.csv"**

```python
python evaluation.py
```



# Contact

Email: zengw5@mail2.sysu.edu.cn







