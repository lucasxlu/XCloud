# Image Censorship
## Introduction
This module holds the source code for **image censorship**, namely, 
**pornography and political figures** recognition.

### Note
1. The dataset for pornography recognition is downloaded from [nsfw_data_scrapper](https://github.com/alexkimxyz/nsfw_data_scrapper.git). 
2. This repository provides both [deep learning-based methods](./main.py), 
[non-deep models and skin models](./porn_img_rec_ml.py) for pornography image
 recognition.


## Data Statistics
| Type | Capacity |
| :---: |:---: |
| drawings | 22249 |
| hentai | 43412 |
| neutral | 9800 |
| porn | 110003 |
| sexy | 18299 |

## Performance
| Model | Accuracy | Precision | Recall |
| :---: |:---: |:---: |:---: |
| DenseNet121 | 93.31% | 90.68% | 89.72% |


## Usage
1. download pretrained model from [Google Drive](https://drive.google.com/open?id=1BF2FaCqhr1LYeZ4vA56pTTlfFumUrg5q)
2. run [inference](./inference.py) by passing your own image
3. model will return json results as:

* Example 1 

![1](./1.jpg)

```json
{"message": "success",
 "results": [{"prob": 1.0, "type": "sexy"},
             {"prob": 0.0, "type": "neutral"},
             {"prob": 0.0, "type": "porn"},
             {"prob": 0.0, "type": "hentai"},
             {"prob": 0.0, "type": "drawings"}],
 "status": 0}
```

**Note**: the returned result indicates that ```sexy``` has the highest 
probability (prob=1.0)

* Example 2 

![2](./2.jpg)
```json
{"message": "success",
 "results": [{"prob": 0.6981, "type": "neutral"},
             {"prob": 0.1811, "type": "porn"},
             {"prob": 0.1205, "type": "sexy"},
             {"prob": 0.0002, "type": "hentai"},
             {"prob": 0.0002, "type": "drawings"}],
 "status": 0}
```

**Note**: the returned result indicates that ```neutral``` has the highest 
probability (prob=0.6981)

* Example 3

![3](./3.jpg)
```json
{"message": "success",
 "results": [{"prob": 0.9982, "type": "porn"},
             {"prob": 0.0011, "type": "neutral"},
             {"prob": 0.0004, "type": "hentai"},
             {"prob": 0.0002, "type": "sexy"},
             {"prob": 0.0, "type": "drawings"}],
 "status": 0}
```

**Note**: the returned result indicates that ```porn``` has the highest 
probability (prob=0.9982)