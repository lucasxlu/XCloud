# Content-based Image Retrieval System
## Introduction
Exploring deep metric learning & hash methods to build efficient visual search system.

**Note:** If you need an online web-based annotation tool for _Image Retrieval/ReID_, please feel free to use [CbirAnnoTool](https://github.com/lucasxlu/CbirAnnoTool.git).

![index](https://raw.githubusercontent.com/lucasxlu/CbirAnnoTool/master/index.png)

### Backbone
| Architecture | Supervision | Status |
| :---: |:---: |:---: |
| DenseNet121 | Softmax | [YES] |
| DenseNet121 | CenterLoss | [YES] |
| DenseNet121 | A-Softmax | [YES] |
| ResNeXt50 | A-Softmax | [TODO] |
| SeResNeXt50 | A-Softmax | [TODO] |


### Dependency
 * [Faiss](https://github.com/facebookresearch/faiss.git)
 * [Django](https://www.djangoproject.com/)
 

## Citation
This module is a part of [XCloud](https://github.com/lucasxlu/XCloud.git), If you use this module in your research in ``Image Retrieval/ReID/Face Recognition``, please cite our [technical report](https://lucasxlu.github.io/blog/about/XCloud.pdf) about [XCloud](https://github.com/lucasxlu/XCloud.git) as:
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
