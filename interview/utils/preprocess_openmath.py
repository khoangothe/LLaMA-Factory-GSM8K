import pandas as pd
import json 
import glob
import os
import numpy as np
import argparse
from datetime import datetime

def sample_parquet_efficiently(directory_path, sample_size, filename_pattern='*.parquet', random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    total_rows = 0
    parquet_files = sorted(glob.glob(os.path.join(directory_path, filename_pattern)))
    print(parquet_files)
    
    for file in parquet_files:
        metadata = pd.read_parquet(file, columns=[])
        total_rows += len(metadata)
    
    selected_indices = np.random.choice(total_rows, size=sample_size, replace=False)
    print(selected_indices)
    selected_indices.sort() 
    
    current_index = 0
    samples = []
    indices_pos = 0
    
    for file in parquet_files:
        file_size = len(pd.read_parquet(file, columns=[]))
        file_end_index = current_index + file_size
        
        indices_for_this_file = []
        while indices_pos < len(selected_indices) and selected_indices[indices_pos] < file_end_index:
            indices_for_this_file.append(selected_indices[indices_pos] - current_index)
            indices_pos += 1
        
        if indices_for_this_file:
            df = pd.read_parquet(file)
            samples.append(df.iloc[indices_for_this_file])
        
        if indices_pos >= len(selected_indices):
            break
            
        current_index = file_end_index
    
    return pd.concat(samples, ignore_index=True)

def preprocess_openmath(df, output_path):
    processed_df = []
    for _, row in df.iterrows():
        sample = {
            "instruction": row["problem"],
            "input": "",
            "output": row["generated_solution"] + " #### " + row["expected_answer"]
        }
        processed_df.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_df, f, indent=2, ensure_ascii=False)

    print(f"Successfully converted {len(processed_df)} examples")
    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Read and sample split parquet files.')
    parser.add_argument('directory', type=str, help='Directory containing parquet files')
    parser.add_argument('--pattern', type=str, default='*.parquet', 
                        help='Pattern to match parquet files (default: *.parquet)')
    parser.add_argument('--sample-size', default = 200_000, type=int, 
                        help='Number of rows to sample (optional)')
    parser.add_argument('--seed', type=int, 
                        help='Random seed for reproducibility (optional)')
    parser.add_argument('--efficient', action='store_true', 
                        help='Use memory-efficient sampling method')
    parser.add_argument('--output', type=str, 
                        help='Output file path (default: sampled_data_<timestamp>.parquet)')
    
    args = parser.parse_args()
    
    try:
        df = sample_parquet_efficiently(
            args.directory, 
            args.sample_size, 
            args.pattern, 
            args.seed
        )
        
        # Generate output filename if not provided
        if not args.output:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            args.output = f'sampled_data_{timestamp}.parquet'

        processed_df = preprocess_openmath(df, args.output)
        
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
