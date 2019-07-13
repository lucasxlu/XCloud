# Context-Aware_Crowd_Counting-pytorch
This is an simple and clean unoffical implemention of CVPR 2019 paper ["Context-Aware Crowd Counting"](https://arxiv.org/pdf/1811.10452.pdf).  
# Installation
&emsp;1. Install pytorch 1.0.0 later and python 3.6 later  
&emsp;2. Install visdom    
```pip
pip install visdom
```
&emsp;3. Install tqdm
```pip
pip install tqdm
```  
&emsp;4. Clone this repository  
```git
git clone https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch.git
```
We'll call the directory that you cloned Context-Aware_Crowd_Counting-pytorch as ROOT.
# Data Setup
&emsp;1. Download ShanghaiTech Dataset from
Dropbox: [link](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or Baidu Disk: [link](http://pan.baidu.com/s/1nuAYslz)  
&emsp;2. Put ShanghaiTech Dataset in ROOT and use "data_preparation/k_nearest_gaussian_kernel.py" to generate ground truth density-map. (Mind that you need modify the root_path in the main function of "data_preparation/k_nearest_gaussian_kernel.py")  
# Training
&emsp;1. Modify the root path in "train.py" according to your dataset position.  
&emsp;2. In command line:
```
python -m visdom.server
```  
&emsp;3. Run train.py
# Testing
&emsp;1. Modify the root path in "test.py" according to your dataset position.  
&emsp;2. Run test.py for calculate MAE of test images or just show an estimated density-map.  
# Other notes
we got the comparable MAE at the 353 epoch [BaiduDisk download with Extraction code: yfwb](https://pan.baidu.com/s/1Y-nnVQoZgmgNjpHhE4y--Q) or [Dropbox Link](https://www.dropbox.com/s/do3yf8hs841exha/cvpr2019_CAN_SHHA_353.pth?dl=0) which is reported in paper. Thanks for the author's(Weizhe Liu) response by email. His mainpage is [link](https://sites.google.com/view/weizheliu/home).

# Reference
1. [Context-Aware_Crowd_Counting-pytorch](https://github.com/CommissarMa/Context-Aware_Crowd_Counting-pytorch.git)