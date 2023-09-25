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
弛 
戍式式 coco_download.sh  
戍式式 coco # can be downloaded with the coco_download.sh 
弛   戍式式 images
弛   弛      戍式式 ......
弛   弛
弛   戌式式 annotations 
弛          戍式式 ......
弛
戍式式 coco_butd
弛   戌式式 precomp  
弛          戍式式 train_ids.txt
弛          戍式式 train_caps.txt
弛          戍式式 ......   
弛
戍式式 f30k 
弛   戍式式 images
弛   弛      戍式式 ......
弛   弛
弛   戍式式 dataset_flickr30k.json
弛   戍式式......  
弛
戌式式 f30k_butd
    戌式式 precomp  
           戍式式 train_ids.txt
           戍式式 train_caps.txt
           戍式式 ......

vocab # included in this repo
弛
戍式式 coco_butd_vocab.pkl
戍式式 ......

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