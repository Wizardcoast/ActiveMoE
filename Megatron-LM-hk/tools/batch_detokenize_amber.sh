#!/bin/bash

INPUT_DIR=/workspace/rawdata/amber/AmberDatasets
LOG_DIR=$INPUT_DIR/logs

# Create the log directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Loop through each .jsonl file in the current directory
for file in $INPUT_DIR/train_*.jsonl; do
    # Skip if the file ends with _detok.jsonl
    if [[ $file == *_detok.jsonl ]]; then
        echo "Skipping $file"
        continue
    fi
    # Extract the base name of the file for use in the log file name
    base_name=$(basename "$file" .jsonl)
    
    # Define the log file for this specific file
    log_file="${LOG_DIR}/${base_name}_detokenize.log"
    
    echo "Starting detokenization of $file" >> "$log_file"
    # Execute detokenize_amber.py with nohup and redirect output to this log file
    nohup python tools/detokenize_amber.py --input "$file" >> "$log_file" 2>&1 &
    echo "Detokenization script launched for $file" >> "$log_file"
done

echo "All detokenization processes started in the background. Check the ${LOG_DIR} directory for individual log files."
