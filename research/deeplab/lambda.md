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
    --train_batch_size=16 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR} \
    --dataset_dir=${PATH_TO_DATASET}
```

**Memory Requirement (MiB)**


| Batch Size  | Memory  |
|---|---|
| bs=1  | 6GB |
| bs=2  | 6GB |
| bs=4  | 11GB |
| bs=8  |  24GB |
| bs=16  | 48GB  |

**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=1  | 3.80  | 4.15  | 4.95  | 4.12   | 5.85 | 6.03  | 5.5  |   | 5.62  |
| bs=2  | 4.4  | 4.82 | 5.80  |   4.80 | 6.52 |  7.3 |  6.50 |   | 6.92  |
| bs=4  | OOM  | OOM | OOM  | 5.43   | 7.6 |  8.4 | 7.55  |   | 8.03 |
| bs=8  | OOM  | OOM | OOM  |  OOM  | OOM | 9.01 | 8.02  |   | 8.40  |
| bs=16  | OOM | OOM  | OOM  | OOM  | OOM  | OOM  | OOM  |   | 9.12  |




**Time cost**
GPU: Titan RTX
Equation: training_number_of_steps / (3600 * samples_per_sec / bs)

30000 / (3600 * 6.03 / 1 ) = 1.38 hours



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
    --train_batch_size=16 \
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
| bs=1  | 6GB |
| bs=2  | 6GB |
| bs=4  | 11GB  |
| bs=8  | 24GB  |
| bs=16  | 48GB |

**Throughput (samples/sec)** 

|   | 2060  | 2070  | 2080  |  1080 Ti | 2080 Ti | TitanRTX | Quadro RTX 6000 | V100 | Quadro RTX 8000 |
|---|---|---|---|---|---|---|---|---|---|
| bs=1  | 3.86  | 3.92  | 4.15  | 3.78  | 5.48  | 5.75  | 5.02  |   | 5.44  |
| bs=2  |  4.48 | 4.50 | 5.52  | 4.20  | 6.45  |  6.92 | 6.44  |   | 6.71  |
| bs=4  | OOM  | OOM  |  OOM | 4.84  | 7.04  |  7.84 | 7.05  |   | 7.64  |
| bs=8  | OOM  | OOM  | OOM  | OOM  |  OOM |  8.24 | 7.23  |   | 7.93  |
| bs=16  | OOM | OOM  | OOM  | OOM  | OOM  |  OOM | OOM  |   | 8.42  |


**Time cost**
GPU: Titan RTX
Equation: training_number_of_steps / (3600 * samples_per_sec / bs)

150000 / (3600 * 7.84 / 4 ) = 21.25 hours