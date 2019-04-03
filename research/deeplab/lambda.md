Installation
===

```bash
cd models/research
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

sudo pip3 install virtualenv
virtualenv -p /usr/bin/python3.6 venv-deeplab
. venv-deeplab/bin/activate

pip install -r deeplab/requirements.txt 
```

Test installation
===

```bash
python deeplab/model_test.py

# Should return
----------------------------------------------------------------------
Ran 5 tests in 13.738s

OK (skipped=1)
```

Running DeepLab on PASCAL VOC 2012 Semantic Segmentation Dataset
===
**Download dataset and convert to TFRecord**

```bash
# Takes about 2 hours
cd deeplab/datasets
sh download_and_convert_voc2012.sh
```

**Train**

```bash
# From tensorflow/models/research/

export PATH_TO_TRAIN_DIR=`pwd`/deeplab/datasets/pascal_voc_seg/exp/train_on_train_set/train

export PATH_TO_DATASET=`pwd`/deeplab/datasets/pascal_voc_seg/tfrecord

CUDA_VISIBLE_DEVICES=0 python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=1 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

**Memory Requirement (MiB)**


| Batch Size  | Memory  |
|---|---|
| bs=1  | 8764 |
| bs=2  | 8756 |
| bs=4  | 16948 |
| bs=8  |  23380 |
| bs=16  |   |

**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=1  | 3.80  | 4.15  |   | 5.85 | 6.03  |   |   |   |
| bs=2  | 4.4  | 4.82 |   | 6.52 |  7.3 |   |   |   |
| bs=4  | OOM  | OOM |   | 7.6 |  8.4 |   |   |   |
| bs=8  | OOM  | OOM |   |  OOM | 9.01 |   |   |   |
| bs=16  | OOM | OOM  |   | OOM  | OOM  |   |   |   |


Running DeepLab on Cityscapes Semantic Segmentation Dataset
===
**Download dataset and convert to TFRecord**

Download the dataset beforehand by registering the [website](https://www.cityscapes-dataset.com/).

```bash
cd deeplab/datasets
sh convert_cityscapes.sh
```


Running DeepLab on ADE20K Semantic Segmentation Dataset
===

**Download dataset and convert to TFRecord**

```bash
cd deeplab/datasets
bash download_and_convert_ade20k.sh
```

**Train**

```bash
# From tensorflow/models/research/

export PATH_TO_TRAIN_DIR=`pwd`/deeplab/datasets/ADE20K/exp/train_on_train_set/train

export PATH_TO_DATASET=`pwd`/deeplab/datasets/ADE20K/tfrecord

CUDA_VISIBLE_DEVICES=0 python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=150000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=1 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=16 \
    --dataset="ade20k" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR}\
    --dataset_dir=${PATH_TO_DATASET}
```

**Memory Requirement (MiB)**


| Batch Size  | Memory  |
|---|---|
| bs=1  | 4660 |
| bs=2  | 8756 |
| bs=4  | 16948  |
| bs=8  | 23380  |
| bs=16  |   |

**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=1  | 3.86  | 3.92  |   | 5.48  | 5.75  |   |   |   |
| bs=2  |  4.48 | 4.50 |   | 6.45  |  6.92 |   |   |   |
| bs=4  | OOM  | OOM  |   | 7.04  |  7.84 |   |   |   |
| bs=8  | OOM  | OOM  |   |  OOM |  8.24 |   |   |   |
| bs=16  | OOM | OOM  |   | OOM  |  OOM |   |   |   |