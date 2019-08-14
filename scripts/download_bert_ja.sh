#!/usr/bin/env bash

DIR=resources/BERT_ja
CONFIG=${DIR}/config.json
TF_MODEL=${DIR}/model.ckpt-1400000
PYTORCH_MODEL=${DIR}/pytorch_model.bin

mkdir -p ${DIR}

curl -L "https://drive.google.com/uc?export=download&id=11V3dT_xJUXsZRuDK1kXiXJRBSEHGl3In" -o ${DIR}/graph.pbtxt
curl -L "https://drive.google.com/uc?export=download&id=1jjZmgSo8C9xMIos8cUMhqJfNbyyqR0MY" -o ${DIR}/wiki-ja.model
curl -L "https://drive.google.com/uc?export=download&id=1V9TIUn5wc-mB_wabYiz9ikvLsscONOKB" -o ${TF_MODEL}.meta
curl -L "https://drive.google.com/uc?export=download&id=1LB00MDQJjb-xLmgBMhdQE3wKDOLjgum-" -o ${TF_MODEL}.index

FILE_ID=1F4b_u-5zzqabA6OfLxDkLh0lzqVIEZuN
curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=${FILE_ID}" -o ${TF_MODEL}.data-00000-of-00001

python3 scripts/convert_tf_checkpoint_to_pytorch.py \
    --tf_checkpoint_path=${TF_MODEL} \
    --bert_config_file=${CONFIG} \
    --pytorch_dump_path=${PYTORCH_MODEL}

rm ${DIR}/graph.pbtxt
rm ${TF_MODEL}.meta
rm ${TF_MODEL}.index
rm ${TF_MODEL}.data-00000-of-00001
