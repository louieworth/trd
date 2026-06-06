# (Combined script for math-ai/math500 and math-ai/amc23)
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

# Prepare math-ai/math500 and math-ai/amc23 evaluation datasets
# hf download math-ai/math500 --repo-type dataset --local-dir "$RAW_DATA_ROOT/math-ai/math500"
# hf download math-ai/amc23 --repo-type dataset --local-dir "$RAW_DATA_ROOT/math-ai/amc23"
# hf download agentica-org/DeepScaleR-Preview-Dataset --repo-type dataset --local-dir "$RAW_DATA_ROOT/agentica-org/DeepScaleR-Preview-Dataset"

import os
import datasets
from verl.utils.reward_score.math_reward import remove_boxed

# Unified instruction (used for both tasks)
instruction_following = "Please reason step by step, and put your final answer within \\boxed{}."


def make_map_fn_math_eval(data_source, question_key="problem"):
    """
    [Function 1: For math-ai evaluation data]
    The original map function used to create the evaluation data.
    
    Args:
        data_source (str): "math500" or "amc23"
        question_key (str): The column name containing the question in the dataset ("problem" or "question")
    """
    def process_fn(example, idx):
        # 1. Pop the key fields
        question_raw = example.pop(question_key)
        answer_raw = example.pop("answer")
        # 2. Save all other fields into extra_info
        extra_info = {}
        for key, value in example.items():
            extra_info[key] = value
        example.clear()

        # 3. Add the index and original Q&A back to extra_info
        extra_info["index"] = idx
        extra_info["answer"] = answer_raw
        extra_info["question"] = question_raw

        # 4. Create the prompt with the instruction
        question = question_raw + " " + instruction_following

        # 5. Extract the clean ground truth answer
        try:
            solution = remove_boxed(answer_raw)
        except Exception:
            solution = answer_raw

        # 6. Maintain the exact same output format as the AIME script
        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": extra_info,
        }
        return data

    return process_fn


def make_map_fn_step1_gen(question_key="problem", data_source="deepscaleR"):
    """
    [Function 2: For agentica Step 1 generation data]
    Prepare data for the first step (strategy generation).
    This function retains columns like 'problem_s', 'expert_cot', etc., for use in Step 2.
    """
    def process_fn(example):
        # 1. Get the raw data (do not pop, keep as is)
        question_raw = example.pop(question_key)
        answer_raw = example.pop('answer')
        extra_info = {}
        
        for key, value in example.items():
            if key != 'solution':
                extra_info[key] = value
        extra_info['expert_cot'] = example.pop('solution', '')
        extra_info[question_key] = question_raw  # 

        # 2. Construct the Step 1 Prompt
        question = question_raw + " " + instruction_following

                # 5. Extract the clean ground truth answer
        try:
            solution = remove_boxed(answer_raw)
        except Exception:
            solution = answer_raw
        # 3. Prepare the output (format consistent with Step 1 script)
        data =  {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user", 
                    "content": question,
                }
            ],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": extra_info,
        }
        return data
    return process_fn


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="./data/eval", help="The base save directory for the preprocessed dataset."
    )

    args = parser.parse_args()

    # --- Unified processing logic ---

    # 1. Define all datasets to be processed and their specific configurations
    datasets_to_process = [
        {
            "hf_name": "math-ai/amc23",
            "short_name": "amc23",
            "question_key": "question", # amc23 uses 'question'
            "split": "test",
            "map_fn_type": "math_eval"  # <-- Specify which function to use
        },
        {
            "hf_name": "math-ai/math500",
            "short_name": "math500",
            "question_key": "problem",  # math500 uses 'problem'
            "split": "test",
            "map_fn_type": "math_eval"
        },

        # NOTE: agentica-org/DeepScaleR-Preview-Dataset is training data, not for evaluation
        # {
        #     "hf_name": "agentica-org/DeepScaleR-Preview-Dataset",
        #     "short_name": "DeepScaleR",
        #     "question_key": "problem",
        #     "split": "train",
        #     "map_fn_type": "step1_gen"
        # }
    ]

    base_save_dir = os.path.expanduser(args.local_save_dir)

    # 2. Loop through and process each dataset
    for d_config in datasets_to_process:
        print(f"--- processing: {d_config['hf_name']} ---")
        
        # Determine the dataset path
        if args.local_dataset_path is not None:
            # Assume the local path is organized by hf_name
            # e.g., $RAW_DATA_ROOT/math-ai/amc23
            # e.g., $RAW_DATA_ROOT/agentica-org/DeepScaleR-Preview-Dataset
            dataset_path = os.path.join(args.local_dataset_path, d_config['hf_name'])
        else:
            dataset_path = d_config['hf_name']

        print(f"loading path: {dataset_path}")
        dataset = datasets.load_dataset(dataset_path, split=d_config['split'])

        # --- Call different processing functions based on map_fn_type ---
        map_fn_type = d_config["map_fn_type"]
        columns_to_remove = None  # Do not remove columns by default
        
        if map_fn_type == "math_eval":
            print("使用 'math_eval' 函数处理...")
            map_function = make_map_fn_math_eval(
                data_source=d_config['short_name'],
                question_key=d_config['question_key']
            )
            dataset = dataset.map(function=map_function, with_indices=True)
            # Columns are already handled inside this function, no need to remove them additionally
            
        elif map_fn_type == "step1_gen":
            print("use 'step1_gen' function processing...")
            map_function = make_map_fn_step1_gen(
                question_key=d_config['question_key']
            )
            dataset = dataset.map(function=map_function, with_indices=False)
        # Perform column removal if needed
        # --- Save ---
        save_dir = os.path.join(base_save_dir, d_config['short_name'])
        os.makedirs(save_dir, exist_ok=True)

        # Name using the split name, e.g., amc23_test.parquet or agentica_step1_input_train.parquet
        output_file_name = f"{d_config['short_name']}_{d_config['split']}.parquet"
        output_file = os.path.join(save_dir, output_file_name)

        # Check if file already exists
        if os.path.exists(output_file):
            print(f"Dataset {d_config['short_name']} already exists at {output_file}, skipping...")
            print()
        else:
            dataset.to_parquet(output_file)
            print(f"save to {output_file}\n")

    print("--- all datasets processed ---")