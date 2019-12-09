# Image Quality Assessment
## Introduction
This module provides some methods for `image quality assessment` in both conventional digital image processing and deep learning based approaches.

If your research interests lie in IQA, please feel free to contact [@LucasX](https://github.com/lucasxlu) or send a Pull Request to this repository. 

## Features
- [x] [Blurry Detection](./blur_detector.py)
- [x] [Reflection Detection](./reflection_detector.py)
- [x] [Lean Detection](./lean_detector.py)
- [x] CNN based IQA ([IQANet](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf), [IQACNNPlusPlus](https://ieeexplore.ieee.org/abstract/document/7351311/) and [NIMA](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8352823))

### Note
- For [IQANet](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) and [IQACNNPlusPlus](https://ieeexplore.ieee.org/abstract/document/7351311/), I set the last output neuron as 2, and adopt ``Cross Entropy Loss`` to train
the deep models to satisfy our requirement. You can also set the last output neuron as 1, remove softmax layer, and train regression nets with ``MSE Loss``.

- I replace the input channel as RGB instead of Gray-scale, since I find RGB input improves accuracy, I also add BatchNorm as a standard component as in SOTA CNN architecture.

## Performance Evaluation
I adopt [IQACNNPlusPlus](https://ieeexplore.ieee.org/abstract/document/7351311/) and [IQANet](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) for ``exposure/edge recognition`` in ``product recognition project`` to reject unqualified images. The performance is listed as follows, you can train your own model with the code provided within this module.

### Over-exposure Recognition
| Model | Acc | Precision | Recall |
| :---: | :---: | :---: | :---: |
| [IQACNNPlusPlus](https://ieeexplore.ieee.org/abstract/document/7351311/) | 90.89% | 91.09% | 90.20% |
| [IQANet](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) | 89.32% | 89.56% | 88.51% |


### Product Edge Recognition
| Model | Acc | Precision | Recall |
| :---: | :---: | :---: | :---: |
| [IQACNNPlusPlus](https://ieeexplore.ieee.org/abstract/document/7351311/) | 89.19% | 87.19% | 80.91% |
| [IQANet](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf) | 89.69% | 88.33% | 81.70% |


## References
1. Kang L, Ye P, Li Y, et al. [Convolutional neural networks for no-reference image quality assessment](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 1733-1740.
2. Kang L, Ye P, Li Y, et al. [Simultaneous estimation of image quality and distortion via multi-task convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/7351311/)[C]//2015 IEEE international conference on image processing (ICIP). IEEE, 2015: 2791-2795.
3. Talebi, Hossein, and Peyman Milanfar. ["Nima: Neural image assessment."](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8352823) IEEE Transactions on Image Processing 27.8 (2018): 3998-4011.
