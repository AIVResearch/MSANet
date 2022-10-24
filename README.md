
# MSANet: Multi-Similarity and Attention Guidance for Boosting Few-Shot Segmentation
	


<p align="middle">
  <img src="figure/Main.png">
</p>

### Dependencies

- Python 3.9
- PyTorch 1.11.0
- cuda 11.0
- torchvision 0.8.1
- tensorboardX 2.14

### Datasets

- PASCAL-5<sup>i</sup>:  [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) + [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)

- COCO-20<sup>i</sup>:  [COCO2014](https://cocodataset.org/#download)

   Download the [data](https://drive.google.com/uc?export=download&id=1J1vPgctx8ojlkswMZS9Ccc8_bU6Zuej4) and put them into the `MSANet/lists` directory.

### Models

- Download the pre-trained backbones from [here](https://drive.google.com/uc?export=download&id=1bWVt8OZt2pSDtXn_XLDbR5obSTVrUpN9) and put them into the `MSANet/initmodel` directory. 
- Download our trained base learners from [Drive](https://drive.google.com/uc?export=download&id=1pPOC2rsSPMTm7B4Dr2b1yJnsMqa20u98) and put them under `initmodel/PSPNet`. 
- We provide all trained MSANet [models](https://drive.google.com/uc?export=download&id=14ILPhyKXva9N8pZB495T5DC5081v9IJj) for performance evaluation. _Backbone: VGG16 & ResNet50; Dataset: PASCAL-5<sup>i</sup> & COCO-20<sup>i</sup>; Setting: 1-shot & 5-shot_.

### Scripts

- Change configuration and add weight path to `.yaml` files in `MSHNet/config` , then run the `test.py` file for testing.

### Performance

Performance comparison with the state-of-the-art approaches (*i.e.*, [HSNet], [BAM] and [VAT] in terms of **average** **mIoU** across all folds. 

1. ##### PASCAL-5<sup>i</sup>

   | Backbone  | Method      | 1-shot                   | 5-shot                   |
   | --------  | ----------- | ------------------------ | ------------------------ |
   | VGG16     | BAM         | 64.41                    | 68.76                    |
   |           | MSANet(ours)| 65.76 <sub>(+1.35)</sub> | 70.40 <sub>(+1.64)</sub> |
   | ResNet50  | BAM         | 67.81                    | 70.91                    |
   |           | MSANet(ours)| 68.52 <sub>(+0.71)</sub> | 72.60 <sub>(+1.69)</sub> |
   | ResNet101 | VAT         | 67.50                    | 71.60                    |
   |           | MSANet(ours)| 69.13 <sub>(+1.63)</sub> | 73.99 <sub>(+2.39)</sub> |

2. ##### COCO-20<sup>i</sup>

   | Backbone | Method      | 1-shot                   | 5-shot                   |
   | -------- | ----------- | ------------------------ | ------------------------ |
   | ResNet50 | BAM         | 46.23                    | 51.16                    |
   |          | MSANet(ours)| 48.03 <sub>(+1.8)</sub>  | 53.67 <sub>(+2.51)</sub> |
   | ResNet101| HSNet       | 41.20                    | 49.50                    |
   |          | MSANet(ours)| 51.09 <sub>(+9.89)</sub> | 56.80 <sub>(+7.30)</sub> |
   
 ### Visualization

<p align="middle">
    <img src="figure/visual.png">
</p>



## References



### BibTeX

````
