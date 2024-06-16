#!/bin/bash

# Assume we use 8 GPUs

# Parameters
VERSION="liuhaotian/llava-v1.5-7b"
DATASET_DIR="./dataset"
EXP_NAME="sesame_bun"
BATCH_SIZE=12
GRAD_ACCUMULATION_STEPS=1
NUM_CLASSES_PER_SAMPLE=3

# User-defined parameters
TRAINING_TYPE="$1"  # Pass 'ReferSeg' or 'ReasonSeg' as the first argument
GPU_SETTINGS="$2"   # Pass GPU settings, e.g., 'localhost:0,1'
MASTER_PORT="$3"    # Pass master port, e.g., '15990'

# Check if parameters are provided
if [ -z "$TRAINING_TYPE" ] || [ -z "$GPU_SETTINGS" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <Training Type> <GPU Settings> <Master Port>"
    echo "Example: $0 ReferSeg localhost:0,1 15990"
    exit 1
fi

# Download the SAM model
VISION_PRETRAINED="sam_vit_h_4b8939.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Check if the file does not exist
if [ ! -f "$VISION_PRETRAINED" ]; then
    echo "File does not exist, downloading..."
    wget -O "$VISION_PRETRAINED" "$URL"
else
    echo "File already exists, no need to download."
fi

# Seg-Only Configuration (Training our SESAME- model)
# DATASET_REFERSEG="refer_seg"
# SAMPLE_RATES_REFERSEG="1"

# ReferSeg Configuration (Training our SESAME model)
DATASET_REFERSEG="refer_seg||correct_refer_seg||vqa||neg_refer_seg"
SAMPLE_RATES_REFERSEG="12,2,2,1"

# ReasonSeg Configuration
DATASET_REASONSEG="sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg"
SAMPLE_RATES_REASONSEG="10,10,2,3,1,1"

if [ "$TRAINING_TYPE" == "ReferSeg" ]; then
    # ReferSeg Command
    deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT train_sesame.py \
      --version="$VERSION" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REFERSEG" \
      --sample_rates="$SAMPLE_RATES_REFERSEG" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE

elif [ "$TRAINING_TYPE" == "ReasonSeg" ]; then
    # ReasonSeg Command
    deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT train_sesame.py \
      --version="$VERSION" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REASONSEG" \
      --sample_rates="$SAMPLE_RATES_REASONSEG" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE

else
    echo "Invalid training type. Please specify either 'ReferSeg' or 'ReasonSeg'."
fi