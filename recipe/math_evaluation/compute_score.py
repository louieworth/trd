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


def compute_score_data_source(data_source, response, ground_truth):
    if data_source in ["openai/gsm8k", "gsm8k"]:
        from verl.utils.reward_score import gsm8k

        return gsm8k.compute_score(response, ground_truth)
    if data_source == "amobench":
        from verl.utils.reward_score import amobench_parser_reward

        return amobench_parser_reward.compute_score(response, ground_truth)

    from verl.utils.reward_score.math_reward import compute_score

    # Math competition datasets - all use the same scoring function
    math_datasets = [
        "aime24", "aime25",  # AIME competitions
        "amc23",             # AMC competitions
        "math500",           # MATH benchmark
        "hmmt25", "hmmt24", "hmmt23",  # HMMT competitions
        "beyondaime",        # BeyondAIME benchmark
        "deepscaleR",        # DeepScaleR training data
    ]

    if data_source in math_datasets:
        return compute_score(response, ground_truth)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
