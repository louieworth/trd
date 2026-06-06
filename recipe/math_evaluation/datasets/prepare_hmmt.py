#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Prepare HMMT (Harvard-MIT Mathematics Tournament) datasets for evaluation.

Downloads HMMT 2023, 2024, and 2025 datasets from HuggingFace and converts them
to the standard parquet format used in the evaluation pipeline.
"""

import argparse
import os
from datasets import load_dataset
import pandas as pd


def create_prompt(problem):
    """Create a prompt from the problem statement."""
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    return [
        {"role": "user", "content": f"{problem}\n{instruction}"}
    ]


def process_hmmt_dataset(dataset_name, year, local_dataset_path):
    """Process HMMT dataset and convert to parquet format."""
    print(f"\nProcessing HMMT {year} from {dataset_name}...")

    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"  Loaded {len(dataset)} problems")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None

    # Process each example
    data = []
    for example in dataset:
        # Extract problem and answer
        problem = example.get("problem", "")
        answer = example.get("answer", "")

        # Skip if missing critical fields
        if not problem or not answer:
            continue

        # Clean answer (extract numeric value if needed)
        answer_str = str(answer).strip()
        if "\\boxed" in answer_str:
            # Extract from \boxed{...}
            import re
            match = re.search(r'\\boxed\{([^}]*)\}', answer_str)
            if match:
                answer_str = match.group(1)

        # Create prompt
        prompt = create_prompt(problem)

        data.append({
            "data_source": f"hmmt{year}",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "type": "rule",
                "ground_truth": answer_str
            },
            "extra_info": {
                "problem": problem,
                "year": year
            }
        })

    # Convert to DataFrame and save as parquet
    if data:
        df = pd.DataFrame(data)
        output_file = os.path.join(local_dataset_path, f"hmmt{year}/hmmt{year}_test.parquet")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check if file already exists
        if os.path.exists(output_file):
            print(f"  ✓ HMMT{year} dataset already exists at {output_file}, skipping...")
            return output_file
        else:
            df.to_parquet(output_file)
            print(f"  ✓ Saved {len(data)} problems to {output_file}")
            return output_file
    else:
        print(f"  ✗ No valid problems found")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare HMMT datasets for evaluation")
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default="./data/eval",
        help="Local path to save datasets"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("HMMT Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {args.local_dataset_path}")

    # Process HMMT datasets for different years
    hmmt_datasets = [
        ("PraMamba/HMMT-202502", "25"),  # HMMT February 2025
        ("MathArena/hmmt_feb_2024", "24"),  # HMMT February 2024
        ("MathArena/hmmt_feb_2023", "23"),  # HMMT February 2023
    ]

    for dataset_name, year in hmmt_datasets:
        process_hmmt_dataset(dataset_name, year, args.local_dataset_path)

    print("\n" + "=" * 60)
    print("HMMT dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
