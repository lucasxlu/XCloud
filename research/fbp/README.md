# Facial Beauty Prediction

## Introduction
Train deep model to recognize a person's beauty score from an given image.

![demo](https://github.com/lucasxlu/HMTNet/blob/master/TikTok.gif)

## Performance
### Results on SCUT-FBP5500
| Model | MAE | RMSE | PC |
| :---: | :---: | :---: | :---: |
| ShuffleNetV2 | 0.4389 | 0.3348 | 0.7639 |
| ResNet18 | 0.2887 | 0.3814 | 0.8276 |
| [CRNet](https://github.com/lucasxlu/CRNet) | 0.2835 | 0.3677 | 0.8558 |
| [TransFBP](https://github.com/lucasxlu/TransFBP) | - | - | 0.8519 |
| [HMTNet](https://github.com/lucasxlu/HMTNet) | 0.2439	| 0.3226 | 0.8801 |
| AaNet | 0.2236 | 0.2954 | 0.9055 |
| R^2 ResNeXt | 0.2416 | 0.3046 | 0.8957 |
| R^3CNN | 0.2120 | 0.2800 | 0.9142 |
| [ComboLoss](https://github.com/lucasxlu/ComboLoss) | 0.2170 | 0.2742 | 0.9177 |


## References
1. Liang L, Lin L, Jin L, et al. [SCUT-FBP5500: A diverse benchmark dataset for multi-paradigm facial beauty prediction](https://arxiv.org/pdf/1801.06345.pdf)[C]//2018 24th International Conference on Pattern Recognition (ICPR). IEEE, 2018: 1598-1603.
2. Xu L, Fan H, Xiang J. [Hierarchical Multi-Task Network For Race, Gender and Facial Attractiveness Recognition](https://ieeexplore.ieee.org/abstract/document/8803614/)[C]//2019 IEEE International Conference on Image Processing (ICIP). IEEE, 2019: 3861-3865.
3. Xu L, Xiang J, Yuan X. [CRNet: Classification and Regression Neural Network for Facial Beauty Prediction](https://link.springer.com/chapter/10.1007/978-3-030-00764-5_61)[C]//Pacific Rim Conference on Multimedia. Springer, Cham, 2018: 661-671.
4. Xu L, Xiang J, Yuan X. [Transferring rich deep features for facial beauty prediction](https://arxiv.org/pdf/1803.07253.pdf)[J]. arXiv preprint arXiv:1803.07253, 2018.
