# Improving Cross-Modal Retrieval with Diverse Set of Embeddings

## Acknowledgement
Parts of our codes are adopted from following repositories.

* https://github.com/yalesong/pvse
* https://github.com/fartashf/vsepp
* https://github.com/lucidrains/perceiver-pytorch

## Dataset
We use the dataset preparation scripts from https://github.com/kuanghuei/SCAN#download-data.
For now, provided code is only for Faster-RCNN + bi-GRU experimental setting on COCO dataset.
We will release release complete training code/trained model weight in future.
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
