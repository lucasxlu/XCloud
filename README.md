<p align="left"><img src="logo/horizontal.svg" alt="XCloud" height="120px"></p>

# XCloud (EXtensive Cloud)
## Introduction
__XCloud__ is an open-source AI platform which provides common AI services 
(computer vision, NLP, data mining and etc.)
with RESTful APIs. The platform is developed and maintained by [@LucasX](https://github.com/lucasxlu) based on [Django](https://www.djangoproject.com/) and [PyTorch](https://pytorch.org/).

## Features
* [Computer Vision](./cv)
    * Face Analysis
        - [x] Face Comparison
        - [x] Facial Beauty Prediction (ShuffleNet V2 as backbone)
        - [x] Gender Recognition
        - [x] Race Recognition
        - [x] Age Estimation
        - [x] Facial Expression Recognition
        - [x] Face Retrieval
    * Image Recognition
        - [x] Scene Recognition
        - [x] Food Recognition
        - [x] Flower Recognition
        - [x] Plant Disease Recognition
        - [x] Pet Insects Detection & Recognition
        - [x] Pornography Image Recognition
        - [x] Skin Disease Recognition
* [NLP](./nlp)
    - [x] Text Similarity Comparison
    - [x] Sentiment Classification for [douban.com](https://www.douban.com/)
    - [x] News Classification
* [Data Mining](./dm)
    - [x] Zhihu Live Quality Evaluation
* Data Services
    - [x] Zhihu Live & Comments
    - [x] Major Hospital Information
    - [x] Primary and Secondary School on [Baidu Baike](https://baike.baidu.com/)
    - [x] Weather History
* [Research](./research)    
    - [x] Age Estimation 
    - [x] Medical Image Analysis (Skin Lesion Analysis)
    - [x] Crowd Counting
    - [x] Intelligent Agriculture
    - [x] Content-based Image Retrieval
    - [x] Image Segmentation
    - [x] Image Dehazing
    
## Deployment
### Basic Environment Preparation
1. create a virtual enviroment named ```pyWeb``` follow [this tutorial](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/001432712108300322c61f256c74803b43bfd65c6f8d0d0000)
2. install [Django](https://docs.djangoproject.com/en/2.1/intro/install/) and [PyTorch](https://pytorch.org/)
3. install all dependent libraries: ```pip3 install -r requirements.txt```
4. activate Python Web environment: ```source ~/pyWeb/bin/activate pyWeb```
5. start django server: ```python3 manage.py runserver 0.0.0.0:8001```
6. open your browser and visit welcome page: ```http://www.lucasx.top:8001/index```


### TensorRT Preparation
In order to construct a more efficient inference engine, it is highly recommended to use [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-install-guide/index.html). With the help of [TensorRT](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-515/tensorrt-install-guide/index.html), we are able to achieve **97.63** FPS on two 2080TI GPUs, which is significantly faster than its counterpart PyTorch model (29.45 FPS).

The installation is listed as follows:  
1. download installation package from NVIDIA official websites. I use ``.tar.gz`` in this project
2. add nvcc to you PATH: ``export PATH=/usr/local/cuda/bin/nvcc:$PATH``
3. install pyCUDA: ``pip3 install 'pycuda>=2017.1.1'``
4. unzip ``.tar.gz`` file, and modify your environment by adding: ``export LD_LIBRARY_PATH=/data/lucasxu/Software/TensorRT-5.1.5.0/lib:$LD_LIBRARY_PATH``
5. install TensorRT Python wheel: ``pip3 install ~/Software/TensorRT-5.1.5.0/python/tensorrt-5.1.5.0-cp37-none-linux_x86_64.whl``
6. install [torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt.git)
7. then you can use [model_converter.py](https://github.com/lucasxlu/XCloud/blob/master/cv/model_converter.py) to convert a PyTorch model to TensorRT model


### Upgrade Django Built-in Server
#### With Gunicorn
1. install Gunicorn: ``pip3 install gunicorn``
2. run your server: ``gunicorn XCloud.wsgi -b YOUR_MACHINE_IP:8001``
3. open your browser and visit welcome page: ```http://YOUR_MACHINE_IP:8001/index```


#### With Nginx
1. install Nginx: ``sudo apt-get install nginx``
2. install uwsgi: ``sudo pip3 install uwsgi``
3. edit ``/etc/nginx/nginx.conf``



![index](index.png)


## Stress Testing
For stress testing, please refer to [API_TESTING_WITH_JMETER.md](API_TESTING_WITH_JMETER.md) for more details!
 

## Contributor
* [@LucasX](https://github.com/lucasxlu): system/algorithm/deployment/report
* [@reallinfo](https://github.com/reallinfo): logo design


## Note
* XCloud is **free for researchers**. For commercial use, please email me AT **xulu0620@gmail.com** for more details. 
* **Please ensure that your machine has a strong GPU equipment**.
* For [XCloud in Java](https://github.com/lucasxlu/CVLH.git), please refer to [CVLH](https://github.com/lucasxlu/CVLH.git) for more details! 
* Technical details can be read from our [Technical Report](https://lucasxlu.github.io/blog/about/XCloud.pdf). 

If you use our codebase or models in your research, please cite this project. **We have released a [Technical Report](https://lucasxlu.github.io/blog/about/XCloud.pdf) about this project**.
```
@misc{xu2019xcloud,
  author =       {Lu Xu and Yating Wang},
  title =        {XCloud: Design and Implementation of AI Cloud Platform with RESTful API Service},
  howpublished = {\url{https://github.com/lucasxlu/XCloud.git}},
  year =         {2019}
}
```


## License
[MIT](./LICENSE)
