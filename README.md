This repository is the implementation of "Towards human-level license plate recognition"(ECCV2018) and "Efficient License Plate Recognition via Holistic Position Attention"(AAAI2021) from USTC VIM Lab. They are designed for accurate and efficient license plate recognition.

# Data preparation
You need to download the [AOLP](http://aolpr.ntust.edu.tw/lab/) and [Medialab](http://www.medialab.ntua.gr/research/LPRdatabase.html) datasets.

Your directory tree should be look like this:
````bash
$DAVSS_ROOT/data
├── AOLP
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
├── Medialab
│   ├── label
│   │   ├── segmentation annotations
│   └── video_image
│       ├── 0001TP
│           ├── decoded images from video clips
│       ├── 0006R0
│       └── 0016E5
│       └── Seq05VD
````

## Towards human-level license plate recognition (ECCV2018)
### Test
Code will be released soon.

### Performance on the AOLP dataset
| Method | AC | LE | RP |
| :-: | :-: | :-: | :-: |
| Ours | 99.41% | 99.31% | 99.02% |

### Performance on the Medialab dataset
| Method | Average Accuracy |
| :-: | :-: |
| Ours | 97.89% |

## Efficient License Plate Recognition via Holistic Position Attention (AAAI2021)
### Install & Requirements
The code has been tested on pytorch=1.5.0 and python3.7. Please refer to `requirements.txt` for detailed information.

**To Install python packages**
```
pip install -r requirements.txt
```

### Test
Please specify the script file.

For example, test our proposed method on AOLP AC subset:
````bash
bash ./aolp/script/test_aolp_ac.sh
````

Testing on the Medialab dataset should follow the above procedure similarly.

### Performance on the AOLP dataset
| Method | AC | LE | RP | FPS |
| :-: | :-: | :-: | :-: | :-: |
| Ours(ResNet-18) | 99.27% | 98.97% | 99.84% | 191 |
| Ours(ResNet-34) | 99.12% | 98.31% | 99.84% | 158 |
| Ours(ResNet-50) | 99.41% | 98.48% | 99.84% | 106 |
| Ours(ResNet-101) | 99.56% | 100.00% | 99.84% | 57 |

### Performance on the Medialab dataset
| Method | Average Accuracy |
| :-: | :-: |
| Ours(ResNet-18) | 98.13% |
| Ours(ResNet-34) | 98.59% |
| Ours(ResNet-50) | 98.59% |
| Ours(ResNet-101) | 98.83% |

### Trained model
We provide trained model on AOLP and Medialab datasets. Please download models from:
| model | Link |
| :--: | :--: |
| AOLP | [BaiduYun(Access Code:iy69)](https://pan.baidu.com/s/1-hdhe-CPf3sMPWVFzlUebQ) |
| Medialab | [BaiduYun(Access Code:iy69)](https://pan.baidu.com/s/1-hdhe-CPf3sMPWVFzlUebQ) |


## Citation
```
@inproceedings{zhuang2018towards,
  title={Towards human-level license plate recognition},
  author={Zhuang, Jiafan and Hou, Saihui and Wang, Zilei and Zha, Zheng-Jun},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={306--321},
  year={2018}
}

@inproceedings{zhang2021efficient,
  title={Efficient License Plate Recognition via Holistic Position Attention},
  author={Zhang, Yesheng and Wang, Zilei and Zhuang, Jiafan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```