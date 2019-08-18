# Garbage Classification
## Introduction
Deep learning for [Garbage Classification](https://developer.huaweicloud.com/competition/competitions/1000007620/introduction?utm_source=bolg&utm_medium=sm-hwysm-bu&utm_campaign=-&utm_content=BK-HWY-BKWZ&utm_term=20190807-01).

During tuning hyper-params of deep model, we randomly split 20% images from 
original dataset as ``testset``, and the remaining images as ``training set``.

## Performance Evaluation
#### Evaluation on IP102 Dataset
| Model | Acc |
| :---: | :---: |
| DenseNet161 + Softmax | 88.79 |
| DenseNet169 + Softmax | 88.89 |