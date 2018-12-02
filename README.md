<p align="left"><img src="logo/horizontal.svg" alt="XCloud" height="120px"></p>

# XCloud (EXplore Cloud)
## Introduction
__XCloud__ is an open-source AI platform which provides common AI services 
(computer vision, NLP, data mining and etc.)
with RESTful APIs. The platform is developed and maintained by [@LucasX](https://github.com/lucasxlu) based on [Django](https://www.djangoproject.com/) and [PyTorch](https://pytorch.org/).

## Features
* Computer  Vision
    * Face Analysis
        - [x] Face  Comparison
        - [x] Facial Beauty Prediction
        - [x] Gender Recognition
        - [x] Race Recognition
        - [x] Age Estimation
        - [x] Facial Expression Recognition
    * Image Recognition
        - [x] Scene Recognition
        - [x] Flower Recognition
        - [x] Porn Image Recognition
* NLP
    - [x] Text Similarity Comparison
    - [x] Sentiment Classification for [douban.com](https://www.douban.com/)
    - [x] News Classification
* Data Mining
    - [x] Zhihu Live Quality Evaluation
* Data Services
    - [x] Zhihu Live & Comments
    - [x] Major Hospital Information
    - [x] Primary and Secondary School on [Baidu Baike](https://baike.baidu.com/)
    - [x] Weather History
    
## Deployment
1. activate Python Web environment: ```source ~/pyWeb/bin/activate pyWeb```
2. start django server: ```python3 manage.py runserver 0.0.0.0:8001```
3. open your browser and visit: ```http://www.lucasx.top:8001/cv/welcome```


## Contributor
* [@LucasX](https://github.com/lucasxlu): system/algorithm/deployment
* [@reallinfo](https://github.com/reallinfo): logo design

## Note
XCloud is **free for researchers**. For commercial use, please email me AT 
**xulu0620@gmail.com** for more details. 

More features will be added in the next future!
For [XCloud in Java](https://github.com/lucasxlu/CVLH.git), please refer to 
[CVLH](https://github.com/lucasxlu/CVLH.git) for more details! 


## Reference
If you find this project useful in your research, please consider citing our
 paper!
 
```
@inproceedings{xu2018crnet,
  title={CRNet: Classification and Regression Neural Network for Facial Beauty Prediction},
  author={Xu, Lu and Xiang, Jinhai and Yuan, Xiaohui},
  booktitle={Pacific Rim Conference on Multimedia},
  pages={661--671},
  year={2018},
  organization={Springer}
}
```

```
@article{xu2018transferring,
  title={Transferring Rich Deep Features for Facial Beauty Prediction},
  author={Xu, Lu and Xiang, Jinhai and Yuan, Xiaohui},
  journal={The 3th International Symposium on Artificial Intelligence and Robotics 2018},
  year={2018}
}
```

## License
[MIT](./LICENSE)
