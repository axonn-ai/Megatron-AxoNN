#!/bin/bash
# Set up vocabulary files and preprocess training data for Megatron-AxoNN.
#
# For a sanity check this downloads wikitext-103 (small, public) and
# preprocesses it into Megatron's binary format.
# For real training runs swap in a larger corpus (C4, The Pile, etc.).
#
# Usage:
#   bash examples/setup_data.sh
#   bash examples/setup_data.sh --dataset pile   # use The Pile instead

set -euo pipefail

DATASET="${1:-wikitext}"    # wikitext | pile
DATA_DIR="${SCRATCH}/sparsecomms/gpt_data"
mkdir -p "$DATA_DIR"
cd "$(dirname "$0")/.."     # Megatron-AxoNN root

echo "Data directory: $DATA_DIR"

# ---------------------------------------------------------------------------
# 1. Vocabulary files (GPT-2 BPE)
# ---------------------------------------------------------------------------
if [ ! -f "$DATA_DIR/gpt2-vocab.json" ]; then
    echo "Downloading gpt2-vocab.json..."
    wget -q -O "$DATA_DIR/gpt2-vocab.json" \
        https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi

if [ ! -f "$DATA_DIR/gpt2-merges.txt" ]; then
    echo "Downloading gpt2-merges.txt..."
    wget -q -O "$DATA_DIR/gpt2-merges.txt" \
        https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
fi

echo "Vocab files ready."

# ---------------------------------------------------------------------------
# 2. Raw corpus -> jsonl
# ---------------------------------------------------------------------------
RAW_JSONL="$DATA_DIR/raw_corpus.jsonl"

if [ ! -f "$RAW_JSONL" ]; then
    echo "Preparing raw corpus ($DATASET)..."

    if [ "$DATASET" = "wikitext" ]; then
        # wikitext-103: ~500MB, good for sanity checks
        python3 - <<'EOF'
import json, sys
from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
out = sys.argv[1] if len(sys.argv) > 1 else "/dev/stdout"

with open("$RAW_JSONL", "w") as f:
    for sample in dataset:
        text = sample["text"].strip()
        if text:
            f.write(json.dumps({"text": text}) + "\n")

print(f"Written {len(dataset)} samples.")
EOF
        # inline python above can't expand $RAW_JSONL, use explicit call
        python3 -c "
import json
from datasets import load_dataset
dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
count = 0
with open('$RAW_JSONL', 'w') as f:
    for sample in dataset:
        text = sample['text'].strip()
        if text:
            f.write(json.dumps({'text': text}) + '\n')
            count += 1
print(f'Written {count} samples to $RAW_JSONL')
"

    elif [ "$DATASET" = "pile" ]; then
        # The Pile: much larger, needs HF token and more disk space
        python3 -c "
import json
from datasets import load_dataset
dataset = load_dataset('EleutherAI/pile', split='train', streaming=True)
count = 0
with open('$RAW_JSONL', 'w') as f:
    for sample in dataset:
        text = sample['text'].strip()
        if text:
            f.write(json.dumps({'text': text}) + '\n')
            count += 1
            if count % 100000 == 0:
                print(f'  {count} samples written...')
print(f'Done. Written {count} samples.')
"
    else
        echo "Unknown dataset: $DATASET. Use 'wikitext' or 'pile'."
        exit 1
    fi
fi

echo "Raw corpus ready: $RAW_JSONL"

# ---------------------------------------------------------------------------
# 3. Preprocess into Megatron binary format
# ---------------------------------------------------------------------------
OUTPUT_PREFIX="$DATA_DIR/BookCorpusDataset"

if [ ! -f "${OUTPUT_PREFIX}_text_document.bin" ]; then
    echo "Preprocessing into Megatron binary format..."
    python tools/preprocess_data.py \
        --input "$RAW_JSONL" \
        --output-prefix "$OUTPUT_PREFIX" \
        --vocab-file "$DATA_DIR/gpt2-vocab.json" \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file "$DATA_DIR/gpt2-merges.txt" \
        --append-eod \
        --workers 8
    echo "Preprocessing done."
else
    echo "Binary dataset already exists, skipping preprocessing."
fi

echo ""
echo "=================================================="
echo "Setup complete. Files in $DATA_DIR:"
ls -lh "$DATA_DIR/"
echo ""
echo "DATA_PATH for submit_sparse.sh:"
echo "  $OUTPUT_PREFIX"
echo "=================================================="
