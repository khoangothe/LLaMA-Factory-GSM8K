import pandas as pd
import json


def convert_gsm8k_to_alpaca(input_parquet, output_json):
    """
    Convert GSM8K dataset from parquet to Alpaca format and save as JSON.

    Args:
        input_parquet (str): Path to input parquet file
        output_json (str): Path to output JSON file
    """
    try:
        # Read the parquet file
        df = pd.read_parquet(input_parquet)

        # Convert each row to Alpaca format
        alpaca_data = []
        for _, row in df.iterrows():
            alpaca_example = {
                "conversations": [
                    {
                        "from": "system",
                        "value": row["prompt"]
                    }
                ],
                "chosen": {
                    "from": "gpt",
                    "value": row["chosen"]
                },
                "rejected": {
                    "from": "gpt",
                    "value": row["rejected"]
                },
            }
            alpaca_data.append(alpaca_example)

        # Save to JSON
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(alpaca_data, f, indent=2, ensure_ascii=False)

        print(f"Successfully converted {len(alpaca_data)} examples")
        print(f"Saved to: {output_json}")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert GSM8K parquet to Alpaca JSON format"
    )
    parser.add_argument("input_parquet", help="Path to input GSM8K parquet file")
    parser.add_argument("output_json", help="Path to output JSON file")

    args = parser.parse_args()
    convert_gsm8k_to_alpaca(args.input_parquet, args.output_json)
