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
Prepare USAMO/USAJMO (USA Mathematical Olympiad) datasets for evaluation.

Downloads USAMO 2024 and 2025 datasets from HuggingFace and converts them
to the standard parquet format used in the evaluation pipeline.
"""

import argparse
import os
import re
from datasets import load_dataset
import pandas as pd


def create_prompt(problem):
    """Create a prompt from the problem statement."""
    instruction = "Please reason step by step, and put your final answer within \\boxed{}."
    return [
        {"role": "user", "content": f"{problem}\n{instruction}"}
    ]


def extract_answer_from_solution(solution):
    """Extract the final answer from the solution text."""
    if not solution:
        return None

    # Look for \boxed{...} pattern
    boxed_match = re.search(r'\\boxed\{([^}]*)\}', solution)
    if boxed_match:
        return boxed_match.group(1).strip()

    # If no boxed answer, return None (these are proof-based problems)
    return None


def process_usamo_dataset(dataset_name, year, local_dataset_path):
    """Process USAMO dataset and convert to parquet format."""
    print(f"\nProcessing USAMO {year} from {dataset_name}...")

    # Load dataset
    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"  Loaded {len(dataset)} problems")
    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return None

    # Process each example
    data = []
    skipped = 0

    for example in dataset:
        # Extract problem and solution
        problem = example.get("problem", "")
        solution = example.get("sample_solution", "")
        answer = example.get("answer", "")

        # Skip if missing problem
        if not problem:
            skipped += 1
            continue

        # Try to extract answer from solution or answer field
        extracted_answer = extract_answer_from_solution(solution) or extract_answer_from_solution(answer)

        # For proof-based problems without numeric answers, we'll use the solution as reference
        if not extracted_answer:
            # Store a placeholder - these are proof-based problems
            # The evaluation will need to handle them differently
            extracted_answer = "proof_based"

        # Create prompt
        prompt = create_prompt(problem)

        data.append({
            "data_source": f"usamo{year}",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "type": "rule",
                "ground_truth": extracted_answer,
                "is_proof_based": (extracted_answer == "proof_based")
            },
            "extra_info": {
                "problem": problem,
                "solution": solution[:500] if solution else "",  # Truncate for storage
                "year": year,
                "points": example.get("points", None)
            }
        })

    # Convert to DataFrame and save as parquet
    if data:
        df = pd.DataFrame(data)
        output_file = os.path.join(local_dataset_path, f"usamo{year}/usamo{year}_test.parquet")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Check if file already exists
        if os.path.exists(output_file):
            print(f"  ✓ USAMO{year} dataset already exists at {output_file}, skipping...")
            return output_file
        else:
            df.to_parquet(output_file)
            print(f"  ✓ Saved {len(data)} problems to {output_file}")
            if skipped > 0:
                print(f"    (Skipped {skipped} invalid entries)")
            return output_file
    else:
        print(f"  ✗ No valid problems found")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare USAMO datasets for evaluation")
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        default="./data/eval",
        help="Local path to save datasets"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("USAMO Dataset Preparation")
    print("=" * 60)
    print(f"Output directory: {args.local_dataset_path}")
    print("\nNote: USAMO problems are proof-based and may not have simple numeric answers.")
    print("The evaluation script will need to handle these differently.")

    # Process USAMO datasets for different years
    usamo_datasets = [
        ("MathArena/usamo_2025", "25"),
        ("MathArena/usamo_2024", "24"),
    ]

    for dataset_name, year in usamo_datasets:
        process_usamo_dataset(dataset_name, year, args.local_dataset_path)

    print("\n" + "=" * 60)
    print("USAMO dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
