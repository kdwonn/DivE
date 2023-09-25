# Improving Cross-Modal Retrieval with Set of Diverse Embeddings

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2211.16761)

This repository contains the official source code for our paper:
>[Improving Cross-Modal Retrieval with Set of Diverse Embeddings](https://arxiv.org/abs/2211.16761)  
> [Dongwon Kim](https://kdwonn.github.io/),
> [Namyup Kim](https://southflame.github.io/), and
> [Suha Kwak](https://suhakwak.github.io/) <br>
> POSTECH CSE<br>
> CVPR (Highlight), Vancouver, 2023.

## Acknowledgement
Parts of our codes are adopted from the following repositories.

* https://github.com/yalesong/pvse
* https://github.com/fartashf/vsepp
* https://github.com/lucidrains/perceiver-pytorch

## Dataset

```
data 
├─ coco_download.sh  
├─ coco # can be downloaded with the coco_download.sh 
│  ├─ images
│  │  └─ ......
│  └─ annotations 
│     └─ ......
├─ coco_butd
│  └─ precomp  
│     ├─ train_ids.txt
│     ├─ train_caps.txt
│     └─ ......   
├─ f30k 
│  ├─ images
│  │  └─ ......
│  ├─ dataset_flickr30k.json
│  └─ ......  
└─ f30k_butd
   └─ precomp  
      ├─ train_ids.txt
      ├─ train_caps.txt
      └─ ......

vocab # included in this repo
├─ coco_butd_vocab.pkl
└─ ......

```

- `coco_butd` and `f30k_butd`: Datasets used for the Faster-RCNN image backbone. We use the pre-computed features provided by SCAN, which can be downloaded via https://github.com/kuanghuei/SCAN#download-data.

- `coco` and `f30k`: Datasets used for the CNN backbones. Please refer the [COCO download script](./data/coco_download.sh) and [Flickr30K website](http://shannon.cs.illinois.edu/DenotationGraph/) to download the images and captions. 

**Note**: Downloaded datasets should be placed according to the directory structure presented above.

## Requirements
You can install requirements using conda.
```
conda create --name <env> --file requirements.txt
```

## Training on COCO 
```
sh train_eval_coco.sh
```