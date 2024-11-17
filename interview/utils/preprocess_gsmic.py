import json
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_qa_json(file_path):
    with open(file_path, 'r') as file:
        qa_list = json.load(file)
    
    # Convert list of single-pair dicts to single dict
    qa_dict = {}
    for item in qa_list:
        qa_dict[item['instruction']]= item['output']

    return qa_dict

def sample_json_list(json_file_path, sample_size=7000, seed=None):
    if seed is not None:
        np.random.seed(seed)

    df = pd.read_json(json_file_path)
    
    if df.shape[1] == 1:
        data = pd.read_json(json_file_path, typ='series')
    else:
        data = df
        
    sampled_data = data.sample(n=sample_size, random_state=seed)
    result = sampled_data.to_dict('records')
    
    return result


def match_data(data, qa_dict):
    process_data = []
    for  row in data:
        example = {
            "instruction":  row["new_question"],
            "input": "",
            "output": qa_dict[row["original_question"]]
        }
        process_data.append(example)

    return process_data

def save_sampled_data(sampled_data, output_path):
    """Save sampled data to a JSON file."""
    with open(output_path, 'w') as file:
        json.dump(sampled_data, file, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Randomly sample items from a JSON file')
    parser.add_argument('input_file', type=str, help='Path to input JSON file')
    parser.add_argument('--output', '-o', type=str, help='Path to output JSON file')
    parser.add_argument('--source', type=str, help='Path to gsm8k train source')
    parser.add_argument('--sample-size', '-n', type=int, default=7000,
                       help='Number of samples to take (default: 7000)')
    parser.add_argument('--seed', '-s', type=int, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    qa_dict = load_qa_json(args.source)
    try:
        count = 0
        sampled_data = sample_json_list(
            args.input_file,
            sample_size=args.sample_size,
            seed=args.seed
        )
        #save_sampled_data(sampled_data, args.output)
        output_data = match_data(sampled_data, qa_dict)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully converted {len(output_data)} examples")
        print(f"Saved to: {args.output}")

        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()
