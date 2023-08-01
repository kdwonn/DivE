# Improving Cross-Modal Retrieval with Diverse Set of Embeddings

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2211.16761)

This repository contains official source code for our paper:
>[Improving Cross-Modal Retrieval with Diverse Set of Embeddings](https://arxiv.org/abs/2211.16761)  
> [Dongwon Kim](https://kdwonn.github.io/),
> [Namyup Kim](https://southflame.github.io/), and
> [Suha Kwak](https://suhakwak.github.io/) <br>
> POSTECH CSE<br>
> CVPR (Highlight), Vancouver, 2023.

## Acknowledgement
Parts of our codes are adopted from following repositories.

* https://github.com/yalesong/pvse
* https://github.com/fartashf/vsepp
* https://github.com/lucidrains/perceiver-pytorch

## Dataset
For now, provided training script is only for Faster-RCNN + bi-GRU experimental setting on COCO dataset.
We use the dataset preparation scripts from https://github.com/kuanghuei/SCAN#download-data.
Place the precomp folder and id_mapping.json under ./data/coco_butd, and vocab file under ./vocab.

## Requirements
You can install requirements using conda.
```
conda create --name <env> --file requirements.txt
```

## Training on COCO 
```
sh train_eval_coco.sh
```
