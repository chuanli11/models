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
# Takes about 30 mins
cd deeplab/datasets
sh download_and_convert_voc2012.sh
```

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

```bash/home/ubuntu/git/models/research/deeplab/datasets/ADE20K/tfrecord
# From tensorflow/models/research/

export PATH_TO_TRAIN_DIR=`pwd`/deeplab/datasets/ADE20K/exp/train_on_train_set/train

export PATH_TO_DATASET=`pwd`/deeplab/datasets/ADE20K/tfrecord

python deeplab/train.py \
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
    --train_batch_size=4 \
    --min_resize_value=513 \
    --max_resize_value=513 \
    --resize_factor=16 \
    --dataset="ade20k" \
    --tf_initial_checkpoint=${PATH_TO_INITIAL_CHECKPOINT} \
    --train_logdir=${PATH_TO_TRAIN_DIR}\
    --dataset_dir=${PATH_TO_DATASET}
```