from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import numpy as np
from open_r1.vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2.5-VL" in model_id or "UI-TARS-1.5" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        elif "Qwen2-VL" in model_id or "UI-TARS" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str, model_size: str, model_type: str):
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case x if "point" in x:
                if model_type in ["ui-tars", "ui-tars-1.5"]:
                    PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. \n\n## Output Format\n\nAction: ...\n\n\n## Action Space\nclick(point='<point>x1 y1</point>'')\n\n## User Instruction
{Question}"""
                else:
                    if model_size == "3B":
                        PROMPT = '''point to the instruction: {Question}, output its coordinates in JSON format {{"point_2d": [x, y], "label": "object name/description"}}.'''
                    elif model_size == "7B":
                        # PROMPT = "Locate the UI element(s) for {Question}, output the coordinates using JSON format: [{{\"point_2d\": [x, y]}}, ...]"
                        PROMPT = "{Question}"
                return PROMPT
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\{.*\[\d+,\s*\d+,\s*\d+,\s*\d+\].*\}.*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]

        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    
    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        """Calculate IoU reward between predicted bounding box from Qwen model and ground truth bounding box."""
        import re
        import os
        from datetime import datetime
        import json
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2]-1, box2[2]-1)
            inter_y2 = min(box1[3]-1, box2[3]-1)
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
            else:
                inter = 0
            union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
            return float(inter)/union
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        answer_tag_pattern = r'<answer>(.*?)</answer>'
        bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
        for content, sol in zip(contents, solution):
            sol = re.findall(answer_tag_pattern, sol, re.DOTALL)[-1]
            sol = json.loads(sol.strip())
            reward = 0.0
            # Try symbolic verification first
            try:
                content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
                if content_answer_match:
                    content_answer = content_answer_match.group(1).strip()
                    bbox_match = re.search(bbox_pattern, content_answer)
                    if bbox_match:
                        bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
                        # if iou(bbox, sol) > 0.5:
                        #     reward = 1.0
                        reward = iou(bbox, sol)
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    @staticmethod
    def point_reward_with_gt(completions, solution=None, width=None, height=None, **kwargs):
        import re
        import os
        from datetime import datetime
        import json
        assert not (width is None)
        assert not (height is None)
        def is_valid_point(point, width, height) -> bool:
            x, y = point
            if x<0 or x>=width or y<0 or y>=height:
                return False
            return True
        def is_inside(point, bbox):
            if bbox[0] <= point[0] and point[0] <= bbox[2] and bbox[1] <= point[1] and point[1] <= bbox[3]:
                return 1
            else:
                return 0
        def extract_point(s):
            floats = re.findall(r'-?\d+\.?\d*', s)
            floats = [float(num) for num in floats]
            if len(floats) == 2:
                click_point = floats
            else:
                click_point = None
            return click_point
        # assert False, completions
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        bbox = [float(si.replace('[', '').replace(']', '')) for si in solution[0].split(',')]
        print(solution[0])
        for content, sol, wid, hei in zip(contents, solution, width, height):
            reward = 0.0
            # calc reward
            point = extract_point(content)
            if point and is_valid_point(point, wid, hei):
                reward = is_inside(point, bbox)
                # print(f"point: {point}, norm voting point: {norm_voting_point}, reward: {reward}")
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n") 
        return rewards
    @staticmethod
    def points2point2bbox_reward(completions, 
                                 info, 
                                 solution=None, 
                                 width=None, 
                                 height=None, 
                                 omniparser_bbox_path="", 
                                 scales=None, 
                                 points2point_method=None, 
                                 is_binary_reward=None, 
                                 threshold=None,  
                                 positive_reward=None,
                                 negative_reward=None,
                                 normalize=None,
                                 extract_tag=None,
                                 model_type=None,
                                 **kwargs):
        import re
        import os
        from datetime import datetime
        import math
        def count_normalized_points(h, w, x_norm, y_norm, d):
            """
            计算h*w平面中，归一化坐标(i/w, j/h)到(x_norm, y_norm)的距离小于d的整数点数量。
            
            参数:
                h: 平面高度（j的范围0~h-1）
                w: 平面宽度（i的范围0~w-1）
                x_norm: 目标点的归一化x坐标（通常∈[0,1]）
                y_norm: 目标点的归一化y坐标（通常∈[0,1]）
                d: 归一化距离阈值（d>0）
            
            返回:
                满足条件的整数点数量
            """
            count = 0
            d_sq = d ** 2  # 预计算d的平方，避免重复开方
            inv_w_sq = 1.0 / (w ** 2)  # 1/w²的预计算（减少重复除法）
            inv_h_sq = 1.0 / (h ** 2)  # 1/h²的预计算
            
            # 确定i的有效范围：i/w需在[x_norm - d, x_norm + d] → i ∈ [w*(x_norm -d), w*(x_norm +d)]
            i_min_raw = w * (x_norm - d)
            i_max_raw = w * (x_norm + d)
            # 裁剪i到[0, w-1]（整数范围）
            i_min = max(0, math.floor(i_min_raw))
            i_max = min(w - 1, math.ceil(i_max_raw))
            
            # 遍历所有可能的i
            for i in range(i_min, i_max + 1):
                # 计算x方向的平方距离（归一化后）
                dx_norm = (i / w) - x_norm
                dx_norm_sq = dx_norm * dx_norm
                # 剩余可分配给y方向的平方距离
                remaining_sq = d_sq - dx_norm_sq
                if remaining_sq <= 0:
                    continue  # 该i下无满足条件的j
                
                # 计算j的有效范围：j/h需在[y_norm - sqrt(remaining_sq), y_norm + sqrt(remaining_sq)]
                sqrt_rem = math.sqrt(remaining_sq)
                j_min_raw = h * (y_norm - sqrt_rem)
                j_max_raw = h * (y_norm + sqrt_rem)
                # 裁剪j到[0, h-1]（整数范围）
                j_min = max(0, math.floor(j_min_raw))
                j_max = min(h - 1, math.ceil(j_max_raw))
                
                # 累加该i下的有效j数量
                if j_min <= j_max:
                    count += j_max - j_min + 1
            
            return count
        def count_normalized_two_circles(h, w, xn1, yn1, d1, xn2, yn2, d2):
            """
            计算h*w平面中，归一化坐标同时满足到两个目标点距离<阈值的整数点数量。
            """
            count = 0
            d1_sq = d1 **2
            d2_sq = d2** 2
            inv_w = 1.0 / w  # 预计算1/w，减少除法
            inv_h = 1.0 / h  # 预计算1/h

            # 步骤1：确定i的候选范围（两个圆的i范围的交集）
            # 圆1的i范围（归一化x：i/w ∈ [xn1-d1, xn1+d1] → i ∈ [w*(xn1-d1), w*(xn1+d1)]）
            i1_min_raw = w * (xn1 - d1)
            i1_max_raw = w * (xn1 + d1)
            i1_min = max(0, math.floor(i1_min_raw))
            i1_max = min(w-1, math.ceil(i1_max_raw))
            # 圆2的i范围
            i2_min_raw = w * (xn2 - d2)
            i2_max_raw = w * (xn2 + d2)
            i2_min = max(0, math.floor(i2_min_raw))
            i2_max = min(w-1, math.ceil(i2_max_raw))
            # 交集i范围
            i_min = max(i1_min, i2_min)
            i_max = min(i1_max, i2_max)
            if i_min > i_max:
                return 0

            # 步骤2：遍历i，计算j的有效范围（两个圆的j范围的交集）
            for i in range(i_min, i_max + 1):
                x_norm = i * inv_w  # 归一化x坐标

                # 圆1：计算j的允许范围
                dx1 = x_norm - xn1
                dx1_sq = dx1 * dx1
                rem1_sq = d1_sq - dx1_sq
                if rem1_sq <= 0:
                    continue
                sqrt_rem1 = math.sqrt(rem1_sq)
                j1_min_raw = h * (yn1 - sqrt_rem1)  # 归一化y反推原始j
                j1_max_raw = h * (yn1 + sqrt_rem1)
                j1_min = max(0, math.floor(j1_min_raw))
                j1_max = min(h-1, math.ceil(j1_max_raw))

                # 圆2：计算j的允许范围
                dx2 = x_norm - xn2
                dx2_sq = dx2 * dx2
                rem2_sq = d2_sq - dx2_sq
                if rem2_sq <= 0:
                    continue
                sqrt_rem2 = math.sqrt(rem2_sq)
                j2_min_raw = h * (yn2 - sqrt_rem2)
                j2_max_raw = h * (yn2 + sqrt_rem2)
                j2_min = max(0, math.floor(j2_min_raw))
                j2_max = min(h-1, math.ceil(j2_max_raw))

                # 取j范围的交集
                j_min = max(j1_min, j2_min)
                j_max = min(j1_max, j2_max)
                if j_min <= j_max:
                    count += j_max - j_min + 1

            return count
        def extract_point(s):
            if extract_tag == "()":
                content = re.findall(r'\((.*?)\)', s)
            elif extract_tag == "[]":
                content = re.findall(r'\[(.*?)\]', s)
            else:
                raise ValueError("extract tag error.")
            if len(content) == 0:
                return None
            floats = re.findall(r'-?\d+\.?\d*', content[0])
            # floats = re.findall(r'-?\d+\.?\d*', s)
            floats = [float(num) for num in floats]
            if len(floats) == 2:
                click_point = floats
            else:
                click_point = None
            return click_point
        def distance(a, b):
            a = np.array(a)
            b = np.array(b)
            return float(np.sqrt(np.sum((a - b)**2)))
        def gaussian_distance(a, b, sigma=0.1):
            a = np.array(a)
            b = np.array(b)
            return float(np.exp(-np.sum((a - b)**2)/(2*sigma**2)))
        def standardize_float_list(numbers):
            
            n = len(numbers)
            # 计算均值
            mean = sum(numbers) / n
            
            # 计算标准差
            squared_diff_sum = sum((x - mean) **2 for x in numbers)
            std_dev = (squared_diff_sum / (n - 1))** 0.5
            
            # 处理标准差为0的特殊情况（所有元素相同）
            if std_dev == 0:
                return [0.0 for _ in numbers]
            
            # 标准化每个元素并返回浮点数列表
            return [(x - mean) / std_dev for x in numbers]
        if model_type in ['qwen2vl', 'ui-tars']:
            width = [1000 for _ in width]
            height = [1000 for _ in height]
        # contents = ["<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [" + completion[0]["content"] for completion in completions]
        contents = [completion[0]["content"] for completion in completions]
        selected_point = [100000, 100000]
        # print(contents)
        if points2point_method == 'kde':
            points = []
            for content in contents:
                point = extract_point(content)
                if point:
                    if scales:
                        point = [int(point[0]*scales[0]), int(point[1]*scales[1])]
                    points.append(point)
            selected_point = Qwen2VLModule.KDE(points=points, width=width[0], height=height[0], normalize=True, select_in_prediction=True, return_density=False)
        elif points2point_method == 'confidence':
            info.sort(key=lambda x: x[0], reverse=True)
            n = len(contents)
            for i in range(n):
                idx = info[i][2]
                point = extract_point(contents[idx])
                if point:
                    selected_point = point
                    if scales:
                        selected_point = [int(selected_point[0]*scales[0]), int(selected_point[1]*scales[1])]
                    break
            info.sort(key=lambda x: x[2])
        elif points2point_method == 'distance_voting':
            norm_points = []
            for content in contents:
                point = extract_point(content)
                if point:
                    if scales:
                        point = [int(point[0]*scales[0]), int(point[1]*scales[1])]
                    norm_point = [point[0]/width[0], point[1]/height[0]]
                    norm_points.append(norm_point)
            selected_point = Qwen2VLModule.distance_voting(norm_points, threshold=threshold)
        else:
            assert False, "points2point_method :{} not exists".format(points2point_method)
        # print(selected_point, width, height)
        norm_selected_point = [selected_point[0]/width[0], selected_point[1]/height[0]]
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        bbox = None
        if omniparser_bbox_path != "":
            bbox = Qwen2VLModule.find_omp_bbox(kwargs['image_path'][0][0], norm_selected_point, omniparser_bbox_path)
        with open(log_path, "a", encoding="utf-8") as f:
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            f.write(f"----------------------{current_time}-----------------------\n")
            f.write(f"points2point_method: {points2point_method}\n")
            f.write(f"Image_path: {image_path}\n")
            f.write(f"Problem: {problem}\n")
            f.write(f"Solution: {solution[0]}\n")
            f.write(f"Selected point: {selected_point}, Normed selected point: {norm_selected_point}\n")
            f.write(f"abs Omniparser bbox: {[int(bbox[0]*width[0]), int(bbox[1]*height[0]), int(bbox[2]*width[0]), int(bbox[3]*height[0])] if not (bbox is None) else None}\n")
        rewards = []
        gt_rewards = []
        gt_bbox = solution[0]
        dis_to_voting_point = [10000] * len(contents)
        for i, content in enumerate(contents):
            reward = negative_reward
            gt_reward = -1
            # calc reward
            point = extract_point(content)
            if point:
                if scales:
                    point = [int(point[0]*scales[0]), int(point[1]*scales[1])]
                norm_point = [point[0]/width[0], point[1]/height[0]]
                dis_to_voting_point[i] = distance(norm_point, norm_selected_point)
                if is_binary_reward == True:
                    if not (bbox is None):
                        if bbox[0] <= norm_point[0] <= bbox[2] and bbox[1] <= norm_point[1] <= bbox[3]:
                            reward = positive_reward
                    else:
                        if distance(norm_point, norm_selected_point) < threshold:
                            reward = positive_reward
                else:
                    # reward = gaussian_distance(norm_point, norm_selected_point)
                    # soft IoU
                    I = count_normalized_two_circles(height[0], width[0], 
                                                    norm_selected_point[0], norm_selected_point[1], threshold,
                                                    norm_point[0], norm_point[1], threshold
                                                    )
                    U = count_normalized_points(height[0], width[0], norm_selected_point[0], norm_selected_point[1], threshold) + count_normalized_points(height[0], width[0], norm_point[0], norm_point[1], threshold) - I
                    reward = I/U
                if gt_bbox[0] <= point[0] <= gt_bbox[2] and gt_bbox[1] <= point[1] <= gt_bbox[3]:
                    gt_reward = 1
            rewards.append(reward)
            gt_rewards.append(gt_reward)
            with open(log_path, "a", encoding='utf-8') as f:
                f.write(f"Reward: {reward}\nInfo: {info[i]}\nContent: {content}\n")
        correct_ratio = len([x for x in rewards if x == positive_reward])/len(contents)  
        # print("rewards before norm: {}".format(rewards))
        if normalize == "pre":
            rewards = standardize_float_list(rewards)
            # print("rewards after norm: {}".format(rewards))
        return rewards, gt_rewards, dis_to_voting_point, correct_ratio
    @staticmethod
    def format_reward_gui_grounding_point(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r'\{\s*"point"\s*:\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]\s*\}'
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
                f.write(f"------------- {current_time} Format reward -------------\n")
                for content, match in zip(completion_contents, matches):
                    f.write(f"Content: {content}\n")
                    f.write(f"Has format: {bool(match)}\n")
        return [1.0 if match else 0.0 for match in matches]
    def random_point_reward(completions, solution=None, is_binary=None, **kwargs):
        import random
        import os
        from datetime import datetime
        contents = [completion[0]["content"] for completion in completions]
        if solution is None:
            solution = [None for _ in range(contents)]
        rewards = []
        for content, sol in zip(contents, solution):
            if is_binary == True:
                reward = random.randint(0, 1)
            else:
                reward = random.uniform(0, 1)
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                if reward <= 1.0:  # this condition can be changed for debug
                    with open(log_path, "a", encoding='utf-8') as f:
                        f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                        f.write(f"image_path: {image_path}\n")
                        f.write(f"problem: {problem}\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {sol}\n")
        return rewards
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case "point_with_gt":
                    return Qwen2VLModule.point_reward_with_gt
                case "bbox":
                    pass
                case "point_omp_voting_binary":
                    return Qwen2VLModule.point_omp_voting_binary_reward
                case "points2point2bbox_reward":
                    return Qwen2VLModule.points2point2bbox_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case "gui_grounding_point":
                    return Qwen2VLModule.format_reward_gui_grounding_point
                case "gui_grounding_bbox":
                    pass
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")
    @staticmethod
    def find_omp_bbox(image_path, point, omniparser_bbox_path):
        import os
        import json
        file_name = os.path.basename(image_path)
        bbox = None
        if not os.path.exists(omniparser_bbox_path):
            # print("omniparser_bbox_path: {} not exists".format(omniparser_bbox_path))
            return bbox
        with open(omniparser_bbox_path, "r", encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['image'] == file_name:
                    # print("------------finding image success------------")
                    for d in data['parsed']:
                        cur_bbox = d['bbox']
                        if cur_bbox[0] <= point[0] <= cur_bbox[2] and cur_bbox[1] <= point[1] <= cur_bbox[3]:
                            bbox = cur_bbox
                    break
        return bbox
    @staticmethod
    def KDE(points: list[list[int, int]], 
            width: int, 
            height: int, 
            sigma: float=0.01, 
            normalize: bool=True,
            return_density: bool=False,
            average: bool=True,
            select_in_prediction: bool=True) -> Union[tuple[int, int], tuple[tuple[int, int], float]]:
        # x: 横向,对应width
        # y: 纵向,对应height
        def gaussian_kernel(x):
            # x: (n, 2)
            return np.exp(-0.5*np.sum((x)**2, axis=1)/sigma)
        points = [point for point in points if 0 <= point[0] < width and 0<= point[1] < height]
        tot = width * height
        all_points = np.empty((tot, 2), dtype=float)
        all_points[:, 0] = np.tile(np.arange(width), height)
        all_points[:, 1] = np.repeat(np.arange(height), width)
        if normalize == True:
            points = [[point[0]/width, point[1]/height] for point in points]
            all_points[:, 0] /= width
            all_points[:, 1] /= height
        density = np.zeros(tot)
        # calculate density map
        for point in points:
            center = np.tile(point, (tot, 1))
            density += gaussian_kernel(all_points - center)
        if average == True:
            density /= len(points)
        density = density.reshape(height, width)
        if normalize == True:
            points = [[int(point[0]*width), int(point[1]*height)] for point in points]
        # return density
        if select_in_prediction == False:
            y, x = np.unravel_index(np.argmax(density), density.shape)
        else:
            x, y = 0, 0
            for xi, yi in points:
                # print("max density: {:.3f}, cur density: {:.3f}".format(density[y, x], density[int(yi), int(xi)]))
                if density[y, x] < density[int(yi), int(xi)]:
                    x = int(xi)
                    y = int(yi)
        # print(density[y, x])
        if return_density:
            return (x, y), density
        else:
            return x, y
    @staticmethod
    def distance_voting(points: list[list], threshold: float=0.05, return_voting_ratio: bool=False):
        # points needed to be normalized before.
        if len(points) == 0:
            print("No point exists !!!!")
            if return_voting_ratio:
                return None, None
            else:
                return None  
        def pixel_distance(x, y):
            return np.sqrt(np.sum((np.array(x)-np.array(y))**2))
        numbers = [0] * len(points)
        # print("--------------------", points)
        for i in range(len(points)):
            for j in range(len(points)):
                if pixel_distance(points[i], points[j]) <= threshold:
                    numbers[i] += 1
        # print("--------------", numbers)
        max_number = max(numbers)
        index = numbers.index(max_number)
        if return_voting_ratio:
            return points[index], max_number/len(points)
        else:
            return points[index]
