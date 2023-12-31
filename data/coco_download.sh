# Below download scripts is originally from the official repository of the PVSE: 
# https://github.com/yalesong/pvse

DATA_DIR='coco'
mkdir -p ${DATA_DIR}

# The version below contains only coco annotation portion extracted from
# http://www.cs.toronto.edu/~faghri/vsepp/data.tar

wget https://people.csail.mit.edu/yalesong/pvse/coco_anno.tar.gz -P ${DATA_DIR}/
tar -zxvf ${DATA_DIR}/coco_anno.tar.gz -C ${DATA_DIR}
rm ${DATA_DIR}/coco_anno.tar.gz

wget http://images.cocodataset.org/zips/train2014.zip -P ${DATA_DIR}/
unzip ${DATA_DIR}/train2014.zip -d ${DATA_DIR}/images/
rm ${DATA_DIR}/train2014.zip 

wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip -P ${DATA_DIR}/
unzip ${DATA_DIR}/val2014.zip -d ${DATA_DIR}/images/ 
rm ${DATA_DIR}/val2014.zip 