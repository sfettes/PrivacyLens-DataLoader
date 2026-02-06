#!/bin/bash
#SBATCH --account=def-gambsseb
#SBATCH --gpus-per-node=h100:1 NOTE vLLM has had some compatibility issues with MIG, so entire GPU requested
#SBATCH --mem=16G
#SBATCH --time=1:00:00

MODEL_PATH="$1"

if [ -z "$MODEL_PATH" ]; then
    echo "Error: No model path provided."
    echo "Usage: sbatch run_get_action.sh /path/to/model"
    exit 1
fi

echo "Running job with model: $MODEL_PATH"

module load python/3.11
module load gcc cuda opencv arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index vllm==0.9.0.1 torch==2.7.0 triton==3.2.0 flashinfer_python==0.2.6.post1 transformers==4.51.1 pandas accelerate flash_attn

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd PrivacyLens-DataLoader/evaluation

python get_final_action.py \
    --model_path "$MODEL_PATH" \
    --input_file '../data/main_data.json' \
    --output_file "results-$SLURM_JOB_ID.jsonl" \
    --tools_file '../data/tool_descriptions.json' \
    --tp_size 1 \
    --max_model_len 8192 \
    --enable_filter