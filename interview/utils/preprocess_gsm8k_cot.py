import pandas as pd
import json
import random
from typing import List, Dict

def extract_reasoning_steps(answer: str) -> tuple[List[str], str]:
    """
    Extract reasoning steps and final answer from GSM8K answer format.
    
    Args:
        answer (str): Raw answer string from GSM8K
        
    Returns:
        tuple: (List of reasoning steps, final answer)
    """
    steps = []
    lines = answer.split('\n')
    final_answer = ""
    
    for line in lines:
        if line.startswith("####"):
            final_answer = line.replace("####", "").strip()
        else:
            step = line.strip()
            if step:  # Only add non-empty steps
                steps.append(step)
                
    return steps, final_answer

def create_few_shot_prompt(examples: List[Dict]) -> str:
    """
    Create a few-shot prompt from example problems.
    
    Args:
        examples: List of dictionaries containing question/answer pairs
        
    Returns:
        Formatted few-shot prompt string
    """
    prompt = "Here are some examples of solving math word problems step by step:\n\n"
    
    for idx, example in enumerate(examples, 1):
        prompt += f"Example {idx}:\n"
        prompt += f"Question: {example['question']}\n"
        prompt += "Solution:\n"
        for step in example['steps']:
            prompt += f"{step}\n"
        prompt += f"Therefore, the answer is {example['final_answer']}\n\n"
    
    return prompt.strip()

def convert_gsm8k_to_alpaca(input_parquet: str, output_json: str, num_shots: int = 3):
    """
    Convert GSM8K dataset from parquet to Alpaca format with few-shot COT examples.
    
    Args:
        input_parquet (str): Path to input parquet file
        output_json (str): Path to output JSON file
        num_shots (int): Number of few-shot examples to include
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(input_parquet)
        
        # Create a list of all examples with extracted reasoning steps
        all_examples = []
        for _, row in df.iterrows():
            steps, final_answer = extract_reasoning_steps(row['answer'])
            example = {
                'question': row['question'],
                'steps': steps,
                'final_answer': final_answer
            }
            all_examples.append(example)
        
        # Convert to Alpaca format with few-shot examples
        alpaca_data = []
        for i in range(len(all_examples)):
            # Randomly sample few-shot examples (excluding the current example)
            other_examples = all_examples[:i] + all_examples[i+1:]
            few_shot_examples = random.sample(other_examples, min(num_shots, len(other_examples)))
            
            # Create the few-shot prompt
            few_shot_prompt = create_few_shot_prompt(few_shot_examples)
            
            # Create the Alpaca example
            current_example = all_examples[i]
            alpaca_example = {
                "instruction": f"{few_shot_prompt}\n\nNow solve this problem step by step:\n{current_example['question']}",
                "input": "",
                "output": "\n".join(current_example['steps'] + [f"Therefore, the answer is {current_example['final_answer']}"]) + f"### {current_example['final_answer']}"
            }
            alpaca_data.append(alpaca_example)
        
        # Save to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully converted {len(alpaca_data)} examples")
        print(f"Each example includes {num_shots} randomly selected few-shot examples")
        print(f"Saved to: {output_json}")
        
    except Exception as e:
        print(f"Error during conversion: {str(e)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convert GSM8K parquet to Alpaca JSON format with few-shot COT examples"
    )
    parser.add_argument("input_parquet", help="Path to input GSM8K parquet file")
    parser.add_argument("output_json", help="Path to output JSON file")
    parser.add_argument("--num-shots", type=int, default=3,
                       help="Number of few-shot examples to include (default: 3)")
    args = parser.parse_args()
    
    convert_gsm8k_to_alpaca(args.input_parquet, args.output_json, args.num_shots)
