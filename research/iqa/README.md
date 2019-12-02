# Image Quality Assessment
## Introduction
This module provides some methods for `image quality assessment` in both conventional digital image processing and deep learning based approaches.

If your research interests lie in IQA, please feel free to contact [@LucasX](https://github.com/lucasxlu) or send a Pull Request to this repository. 

## Features
- [x] [Blurry Detection](./blur_detector.py)
- [x] [Reflection Detection](./reflection_detector.py)
- [x] [Lean Detection](./lean_detector.py)
- [x] CNN based IQA ([IQANet](./models.py) & [IQACNNPlusPlus](./models.py))

### Note
For [IQANet](./models.py) and [IQACNNPlusPlus](./models.py), we set the last output neuron as 2, and adopt ``Cross Entropy Loss`` to train
the deep models to satisfy our requirement. You can also set the last output neuron as 1, and train regression nets with ``MSE Loss``.
   

## References
1. Kang L, Ye P, Li Y, et al. [Convolutional neural networks for no-reference image quality assessment](http://openaccess.thecvf.com/content_cvpr_2014/papers/Kang_Convolutional_Neural_Networks_2014_CVPR_paper.pdf)[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 1733-1740.
2. Kang L, Ye P, Li Y, et al. [Simultaneous estimation of image quality and distortion via multi-task convolutional neural networks](https://ieeexplore.ieee.org/abstract/document/7351311/)[C]//2015 IEEE international conference on image processing (ICIP). IEEE, 2015: 2791-2795.