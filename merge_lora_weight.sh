#!/bin/bash

# Set the model directory
EXP_DIRECTORY="./runs/sesame_bun"

# Set CUDA_VISIBLE_DEVICES variable
export CUDA_VISIBLE_DEVICES="0"

# Set LLAVA_PATH and HF_CKPT_PATH
LLAVA_PATH="liuhaotian/llava-v1.5-7b"
HF_CKPT_PATH="${EXP_DIRECTORY}/hg_model"

# Save the current directory
ORIGINAL_DIR=$(pwd)

# Step 1: Create a temporary filename under the model directory
TMP_FILE="$(realpath "${EXP_DIRECTORY}/tmp_file_$(date +%s).bin")"

cd "${EXP_DIRECTORY}/ckpt_model"

# Check if the temporary file was created successfully
if [ $? -ne 0 ]; then
  echo "Error: Failed to create a temporary file."
  exit 1
fi

# Run zero_to_fp32.py and write to the temporary file
python zero_to_fp32.py . "$TMP_FILE"

# Check if zero_to_fp32.py executed successfully
if [ $? -ne 0 ]; then
  echo "Error: zero_to_fp32.py failed."
  exit 1
fi

# Return to the original directory
cd "$ORIGINAL_DIR"

# Step 2: Run merge_lora_weights_and_save_hf_model.py
python3 merge_lora_weights_and_save_hf_model.py \
  --version="${LLAVA_PATH}" \
  --weight="$TMP_FILE" \
  --save_path="./${HF_CKPT_PATH}"

# Check if merge_lora_weights_and_save_hf_model.py executed successfully
if [ $? -ne 0 ]; then
  echo "Error: merge_lora_weights_and_save_hf_model.py failed."
fi

# Clean up the temporary file
rm "$TMP_FILE"
