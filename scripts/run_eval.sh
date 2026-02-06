#!/bin/bash
#SBATCH --account=def-gambsseb
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=16G
#SBATCH --time=0:30:00

# --- ARGUMENT PARSING LOGIC ---
# Loop through arguments and process them. 
# Note that sbatch only allows positional arguments to script by default, so loop parsing makes it easier to use named arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL_PATH="$2"
      shift 
      shift 
      ;;
    --prediction-file)
      PREDICTION_FILE="$2"
      shift
      shift
      ;;
    --step)
      STEP_NAME="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Check variables are set
if [[ -z "$MODEL_PATH" || -z "$PREDICTION_FILE" || -z "$STEP_NAME" ]]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch evaluate_job.sh --model <path> --prediction-file <file> --step <step>"
    exit 1
fi

echo "Running evaluation with:"
echo "  Model: $MODEL_PATH"
echo "  Preds: $PREDICTION_FILE"
echo "  Step:  $STEP_NAME"

module load python/3.11
module load gcc cuda opencv arrow
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index vllm==0.9.0.1 torch==2.7.0 triton==3.2.0 flash_attn==2.8.3 transformers==4.51.1 pandas

export VLLM_ATTENTION_BACKEND=FLASH_ATTN

cd PrivacyLens-DataLoader/evaluation

python evaluate_final_action.py \
    --model "$MODEL_PATH" \
    --data-path '../data/main_data.json' \
    --prediction-file "$PREDICTION_FILE" \
    --step "$STEP_NAME" \
    --output-path "./eval-results-${SLURM_JOB_ID}.json"