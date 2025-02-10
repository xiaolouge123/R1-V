# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import re
from ast import literal_eval
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Tuple
import random

import json
from datasets import (
    load_dataset,
    load_from_disk,
    Dataset,
    Image,
    Value,
    Features,
    DatasetDict,
)
from PIL import Image as PILImage

from transformers import Qwen2VLForConditionalGeneration

from open_r1.trainer import Qwen2VLGRPOTrainer
from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_peft_config,
)

from qwen_vl_utils import smart_resize, to_rgb
from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def extract_xml(text: str, tag: str) -> str:
    """
    Extracts the content of the specified XML tag from the given text. Used for parsing structured responses

    Args:
        text (str): The text containing the XML.
        tag (str): The XML tag to extract content from.

    Returns:
        str: The content of the specified XML tag, or an empty string if the tag is not found.
    """
    match = re.search(f"<{tag}>(.*?)</{tag}>", text, re.DOTALL)
    return match.group(1) if match else ""


def parse_parameter(parameter: str) -> dict:
    """
    Parser the parameter string into a dictionary.
    key is the name of the parameter, value is the value of the parameter.
    parameter is like:
    'point: [426, 270]' -> {'point': [426, 270]}
    'region: [20, 300, 400, 500]' -> {'region': [20, 300, 400, 500]}
    'direction: up' -> {'direction': 'up'}
    'text: Click to manage account information.' -> {'text': 'Click to manage account information.'}
    """
    try:
        if ":" in parameter:
            parameter = parameter.strip().split(":", 1)
            key = parameter[0].strip()
            if key in ['point', 'region']:
                value = literal_eval(parameter[1].strip())
            else:
                value = parameter[1].strip()
            return {key: value}
        else:
            return {}
    except Exception as e:
        print(f"Error parsing parameter: {e}")
        print(f"Parameter: {parameter}")
        return {}


def extract_action(text: str) -> str:
    """
    Extract the ground truth action from the anwser.

    <answer>
        <action description>
            Click on the ['My account\nManage account info'] to (Click to manage account information.)
        </action description>
        <action>
            <name>
                TAP
            </name>
            <parameters>
                <parameter>
                    point: [426, 270]
                </parameter>
            </parameters>
        </action>
        <active region>
            region: [20, 300, 400, 500]
        </active region>
    </answer>

    a well defined action contains: name, parameters, active_region
    """
    answer = extract_xml(text, "answer").strip()
    action = extract_xml(answer, "action").strip()
    action_name = extract_xml(action, "name").strip()
    parameter = parse_parameter(
        extract_xml(action, "parameter").strip()
    )  # TODO suppose only one parameter
    active_region = parse_parameter(extract_xml(answer, "active region").strip())
    return action_name, parameter, active_region


def point_in_region(point: Tuple[int, int], region: Tuple[int, int, int, int]) -> bool:
    """
    Check if the point is in the region.
    """
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


def eval_action(gt_action, gt_parameter, gt_active_region, action, parameter) -> float:
    """
    Evaluate the action is correct and return the reward.
    Action Space includes:
        TAP, SWIPE, TYPE, with parameters: point, direction, text
        TASK_COMPLETE, PRESS_ENTER, TASK_IMPOSSIBLE, PRESS_BACK, PRESS_HOME, WAIT, with no parameters
    """

    def lcs(s1, s2):
        """
        Calculate the longest common subsequence (LCS) of two strings.
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        return dp[m][n]

    reward = 0.0

    if gt_action != action:
        return reward # 动作不一致，奖励为0
    
    reward += 0.5 # 动作一致，奖励0.5
    
    if gt_action == "TAP":
        point = parameter.get("point", None)
        if (
            isinstance(point, list)
            and isinstance(gt_active_region, list)
        ):
            reward += 0.5 # 参数类型正确，奖励0.5
            if point_in_region(point, gt_active_region):
                reward += 1 # 点在区域内，奖励0.5
    
    elif gt_action == "SWIPE":
        direction = parameter.get("direction", None)
        gt_direction = gt_parameter.get("direction", None)
        if direction is not None and direction in ["up", "down", "left", "right"]:
            reward += 0.5 # 参数类型正确，奖励0.5
            if direction == gt_direction:
                reward += 0.5 # 方向一致，奖励0.5

    elif gt_action == "TYPE":
        text = parameter.get("text", None)
        gt_text = gt_parameter.get("text", None)
        if text is not None:
            reward += 0.5 # 参数类型正确，奖励0.5
            # 比较内容是否相似，lcs > 50% 则认为相似
            if lcs(text, gt_text) / min(len(text), len(gt_text)) > 0.5:
                reward += 0.5 # 内容相似，奖励0.5

    elif gt_action == "TASK_COMPLETE":
        if action == "TASK_COMPLETE" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
    
    elif gt_action == "PRESS_ENTER":
        if action == "PRESS_ENTER" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
        
    elif gt_action == "TASK_IMPOSSIBLE":
        if action == "TASK_IMPOSSIBLE" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
        
    elif gt_action == "PRESS_BACK":
        if action == "PRESS_BACK" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
        
    elif gt_action == "PRESS_HOME":
        if action == "PRESS_HOME" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
        
    elif gt_action == "WAIT":
        if action == "WAIT" and parameter == {}:
            reward += 0.5 # 动作一致，奖励0.5
    else:
        reward += 0.0 # 动作不一致，奖励为0

    return reward


def action_accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gt_action, gt_parameter, gt_active_region = extract_action(sol)
        action, parameter, _ = extract_action(content)
        if gt_action:
            # gt 有可解析的action
            reward = eval_action(
                gt_action, gt_parameter, gt_active_region, action, parameter
            )
        else:
            # gt 没有可解析的action
            reward = 1.0
        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(
                     f"------------- {current_time} Action Accuracy reward: {reward} -------------\n"
                )
                f.write(
                    f"========== Generation ==========\n{content}\n==========================\n"
                )
                f.write(
                    f"========== GT Answer ==========\n{sol}\n==========================\n"
                )
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"\s*<think>.*?</think>\s*<answer>.*?</answer>\s*"
    # 检查是否只包含一个 action description 和一个 action 标签
    action_desc_pattern = r"<action description>.*?</action description>"
    action_pattern = r"<action>.*?</action>"
    completion_contents = [completion[0]["content"] for completion in completions]
    penaltys = []

    for content in completion_contents:
        # 提取answer部分
        answer = extract_xml(content, "answer")
        # 计数action description和action标签数量
        action_desc_count = len(re.findall(action_desc_pattern, answer, re.DOTALL))
        action_count = len(re.findall(action_pattern, answer, re.DOTALL))
        if action_count == 1:
            penaltys.append(1.0)
        else:
            penaltys.append(0.0)

    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL) for content in completion_contents]
    rewards = [1.0 if match else 0.0 for match in matches]
    assert len(rewards) == len(penaltys)

    if os.getenv("DEBUG_MODE") == "true":
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        log_path = os.getenv("LOG_PATH")
        # local_rank = int(os.getenv("LOCAL_RANK", 0))
        for content, reward, penalty in zip(completion_contents, rewards, penaltys):
            with open(log_path, "a") as f:
                f.write(
                    f"------------- {current_time} Format Accuracy reward: {reward} penalty: {penalty}, total: {reward + penalty} -------------\n"
                )
                f.write(
                    f"========== Generation ==========\n{content}\n==========================\n"
                )
    return [r + p for r, p in zip(rewards, penaltys)]


reward_funcs_registry = {
    "accuracy": action_accuracy_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a task, and the Assistant solves it by giving operation step on gui interface. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def load_dataset_from_the_disk(jsonl_file_path):
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    print("Loaded data sample: ", data[0])
    print(f"Before filter, data length: {len(data)}")
    lentgh_limit = 3300
    data = [d for d in data if len(d["problem"]) < lentgh_limit]
    print(f"After filter, data length: {len(data)}")
    
    features = Features(
        {
            "image": Image(),
            "problem": Value("string"),
            "solution": Value("string"),
        }
    )

    def process_example(example):
        image_path = example["image_path"]
        try:
            # Load image using PIL
            image = PILImage.open(image_path)
            return {
                "image": image,
                "problem": example["problem"],
                "solution": example["solution"],
            }
        except FileNotFoundError:
            print(f"Image not found: {image_path}")
            return None  # Or handle the error as appropriate
        except Exception as e:
            print(f"Error processing {example['image_path']}: {str(e)}")
            return None

    ds = Dataset.from_list(data)
    idx = [x for x in range(len(ds))]
    seed = 42
    random.seed(seed)
    random.shuffle(idx)
    ds = DatasetDict(
        {
            "train": ds.select(idx[:20000]),
            "test": ds.select(idx[20000:21000]),
        }
        # {
        #     "train": ds.select(idx[:400]),
        #     "test": ds.select(idx[400:410]),
        # }
    )
    ds = ds.map(
        process_example,
        remove_columns=["image_path"],  # 移除原有列
        features=features,  # 指定新的特征结构
        num_proc=10,
    )
    ds = ds.filter(lambda x: x is not None)
    return ds


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset_from_the_disk(script_args.dataset_name)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["problem"]},
            ],
        }

    QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."

    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.format(
                                Question=example["problem"]
                            ),
                        },
                    ],
                },
            ],
        }

    def preprocess_image(example, min_pixels, max_pixels, size_factor):
        image = to_rgb(example["image"])
        width, height = image.size
        min_pixels = min_pixels
        max_pixels = max_pixels
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        image = image.resize((resized_width, resized_height))
        return {"image": image}

    if "image" in dataset[script_args.dataset_train_split].features:
        print("has image in dataset")
        dataset = dataset.map(
            make_conversation_image
        )  # Utilize multiprocessing for faster mapping
        # dataset = dataset.map(
        #     preprocess_image,
        #     fn_kwargs={"min_pixels": script_args.min_pixels, "max_pixels": script_args.max_pixels, "size_factor": IMAGE_FACTOR},
        # )
    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        dataset = dataset.remove_columns("messages")

    trainer_cls = Qwen2VLGRPOTrainer

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
