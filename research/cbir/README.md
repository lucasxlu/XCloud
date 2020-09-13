# Content-based Image Retrieval System
## Introduction
Exploring deep metric learning & hash methods to build efficient visual search system.

> If you need an online web-based annotation tool for _Image Retrieval/ReID_, please feel free to use [CbirAnnoTool](https://github.com/lucasxlu/CbirAnnoTool.git).

![index](https://raw.githubusercontent.com/lucasxlu/CbirAnnoTool/master/index.png)

### Backbone
| Architecture | Supervision | Status |
| :---: |:---: |:---: |
| DenseNet121 | Softmax | [YES] |
| DenseNet121 | CenterLoss | [YES] |
| DenseNet121 | A-Softmax | [YES] |
| DenseNet121 | ArcLoss | [YES] |
| ResNeXt50 | A-Softmax | [TODO] |
| SeResNeXt50 | A-Softmax | [TODO] |


### Dependency
 * [Faiss](https://github.com/facebookresearch/faiss.git)
 * [Django](https://www.djangoproject.com/)
 

## About New Categories
In image retrieval and ReID tasks, how to automatically mine new categories online remains a quite challenging problem.
I propose a simple yet effective method by utilizing unsupervised learning algorithms (such as clustering) to assign pseudo label to a cluster.
After manually cleaning noise samples in each cluster, you can upgrade your embedding model and gallery. You can find this method implementation in [clustering.py](./clustering.py).


## References
1. Sun, Yi, Xiaogang Wang, and Xiaoou Tang. ["Deep learning face representation from predicting 10,000 classes."](http://mmlab.ie.cuhk.edu.hk/pdf/YiSun_CVPR14.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2014.
2. Schroff, Florian, Dmitry Kalenichenko, and James Philbin. ["Facenet: A unified embedding for face recognition and clustering."](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Schroff_FaceNet_A_Unified_2015_CVPR_paper.pdf) Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
3. Wen Y, Zhang K, Li Z, Qiao Y. [A discriminative feature learning approach for deep face recognition](https://ydwen.github.io/papers/WenECCV16.pdf). In European Conference on Computer Vision 2016 Oct 8 (pp. 499-515). Springer, Cham.
4. Wang F, Xiang X, Cheng J, Yuille AL. [Normface: L2 hypersphere embedding for face verification](https://arxiv.org/pdf/1704.06369v4.pdf). InProceedings of the 2017 ACM on Multimedia Conference 2017 Oct 23 (pp. 1041-1049). ACM.
5. Liu, Weiyang, et al. ["Sphereface: Deep hypersphere embedding for face recognition."](http://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.pdf) The IEEE Conference on Computer Vision and Pattern Recognition (CVPR). Vol. 1. 2017.
6. Hadsell, Raia, Sumit Chopra, and Yann LeCun. ["Dimensionality reduction by learning an invariant mapping."](http://www.cs.toronto.edu/~hinton/csc2535/readings/hadsell-chopra-lecun-06-1.pdf) null. IEEE, 2006.


## License
[MIT](./LICENSE)
