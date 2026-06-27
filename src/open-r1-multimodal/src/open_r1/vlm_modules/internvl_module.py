from open_r1.vlm_modules.vlm_module import VLMBaseModule
from typing import Dict, Any, Union
from transformers import AutoModel, AutoProcessor, AutoConfig
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers.feature_extraction_sequence_utils import BatchFeature
import numpy as np
IMG_START_TOKEN='<img>'
IMG_END_TOKEN='</img>'
IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class InvernVLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()
        self.conv_template = None
        self.num_image_token = None

    def get_vlm_key(self):
        return "internvl"
        
    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        assert "InternVL" in model_id, f"model_id must contain 'InternVL', but got {model_id}"
        self.model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # The model class of InternVL when being mapped has been determined by its config
        model_cls = AutoModel
        # InternVL should be inputted with "trust_remote_code=True"
        model_init_kwargs["trust_remote_code"] = True
        # "use_cache" should be removed
        model_init_kwargs.pop("use_cache", None)
        # "flash_attention_2" should be modified to "use_flash_attn" in InternVL
        if "flash_attention_2" in model_init_kwargs.get("attn_implementation", ""):
            model_init_kwargs["use_flash_attn"] = True
            model_init_kwargs.pop("attn_implementation")
        return model_cls

    def post_model_init(self, model, processing_class):
        self.conv_template = model.conv_template if self.conv_template is None else self.conv_template
        self.num_image_token = model.num_image_token if self.num_image_token is None else self.num_image_token
        img_context_token_id = processing_class.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = img_context_token_id
    
    def is_embeds_input(self):
        return True

    def get_processing_class(self):
        return AutoProcessor
    
    def get_eos_token_id(self, processing_class):
        eos_token_id = processing_class.convert_tokens_to_ids(self.conv_template.sep.strip())
        return eos_token_id
        
    def get_vision_modules_keywords(self):
        return ['vision_model']

    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_flags']
    
    def get_non_generate_params(self):
        return ['image_flags']

    def get_custom_processing_keywords(self):
        return [('None', 'max_anyres_num')]

    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = []
        for example in inputs:
            template = self.conv_template.copy()
            conversation_list = example["prompt"]
            system_message = extract_system_message(conversation_list)
            if system_message is not None:
                template.system_message = system_message
            
            processed_list = process_conversation_list(conversation_list, system_message)
            for i, processed_item in enumerate(processed_list):
                if i % 2 == 0:
                    template.append_message(template.roles[0], processed_item)
                else:
                    template.append_message(template.roles[1], processed_item)
            if len(processed_list) % 2 == 1:
                template.append_message(template.roles[1], None)
            query = template.get_prompt()
            prompts_text.append(query)
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # Process images
        full_pixel_values = []
        num_patches_list = []
        for img in images:
            pixel_values = self._load_image(img, input_size=self.model_config.vision_config.image_size, max_num=processing_class.max_anyres_num)
            full_pixel_values.append(pixel_values)
            num_patches_list.append(pixel_values.shape[0])
        full_pixel_values = torch.cat(full_pixel_values, dim=0)
        
        # Process prompts
        queries = []
        image_idx = 0
        for query in prompts_text:
            while "<image>" in query:
                num_patches = num_patches_list[image_idx]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                query = query.replace("<image>", image_tokens, 1)
                image_idx += 1
            queries.append(query)
        assert image_idx == len(num_patches_list)
        
        model_inputs = processing_class(
            queries,
            return_tensors=return_tensors,
            padding=padding,
            padding_side=padding_side,
            add_special_tokens=add_special_tokens,
        )
        model_inputs["pixel_values"] = full_pixel_values
        # Only support pure-image data currently (each sample should contain the image)
        model_inputs['image_flags'] = torch.ones(full_pixel_values.shape[0], dtype=torch.long)
        
        model_inputs = BatchFeature(data=model_inputs)

        return model_inputs

    def _load_image(self, image: Image.Image, input_size: int=448, max_num:int=12):
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
    @staticmethod
    def get_question_template(task_type: str, model_size: str, model_type: str):
        match task_type:
            case _:
                return "\n{Question}"
    
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the InternVL model output matches a specific format."""
        import re
        import os
        from datetime import datetime
        pattern = r"<think>.*?</think>\s*<answer>.*?\[\d+,\s*\d+,\s*\d+,\s*\d+\].*?</answer>"
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
        """Calculate IoU reward between predicted bounding box from InternVL model and ground truth bounding box."""
        """Adopt soft iou reward here"""
        import re
        import os
        import json
        from datetime import datetime
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
        # contents = ["<tool_call>\n{\"name\": \"computer_use\", \"arguments\": {\"action\": \"left_click\", \"coordinate\": [" + completion[0]["content"] for completion in completions]
        contents = [completion[0]["content"] for completion in completions]
        norm_selected_point = [100000, 100000]
        # print(contents)
        if points2point_method == 'confidence':
            info.sort(key=lambda x: x[0], reverse=True)
            n = len(contents)
            for i in range(n):
                idx = info[i][2]
                norm_point = extract_point(contents[idx])
                if norm_point:
                    norm_selected_point = norm_point
                    if scales:
                        norm_selected_point = [int(norm_selected_point[0]*scales[0]), int(norm_selected_point[1]*scales[1])]
                    break
            info.sort(key=lambda x: x[2])
        # print(selected_point, width, height)
        log_path = os.getenv("LOG_PATH")
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        with open(log_path, "a", encoding="utf-8") as f:
            image_path = kwargs.get("image_path")[0] if "image_path" in kwargs else None
            problem = kwargs.get("problem")[0]
            f.write(f"----------------------{current_time}-----------------------\n")
            f.write(f"points2point_method: {points2point_method}\n")
            f.write(f"Image_path: {image_path}\n")
            f.write(f"Problem: {problem}\n")
            f.write(f"Solution: {solution[0]}\n")
            f.write(f"Selected point: {[norm_selected_point[0]*width[0], norm_selected_point[1]*height[0]]}, Normed selected point: {norm_selected_point}\n")
        rewards = []
        gt_rewards = []
        gt_bbox = solution[0]
        dis_to_voting_point = [10000] * len(contents)
        for i, content in enumerate(contents):
            reward = negative_reward
            gt_reward = -1
            # calc reward
            norm_point = extract_point(content)
            if norm_point:
                if scales:
                    norm_point = [int(point[0]*scales[0]), int(point[1]*scales[1])]
                point = [norm_point[0]*width[0], norm_point[1]*height[1]]
                dis_to_voting_point[i] = distance(norm_point, norm_selected_point)
                if is_binary_reward == True:
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
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return InvernVLModule.iou_reward
                case "point_with_gt":
                    return InvernVLModule.point_reward_with_gt
                case "bbox":
                    pass
                case "point_omp_voting_binary":
                    return InvernVLModule.point_omp_voting_binary_reward
                case "points2point2bbox_reward":
                    return InvernVLModule.points2point2bbox_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return InvernVLModule.format_reward_rec
                case "gui_grounding_point":
                    return InvernVLModule.format_reward_gui_grounding_point
                case "gui_grounding_bbox":
                    pass
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")


def process_conversation_list(conversation_list, system_message=None, image_newline=True):
    if system_message is not None:
        conversation_list = conversation_list[1:]
    processed_list = []
    
    for item in conversation_list:
        role = item["role"]
        content = item["content"]
        
        if isinstance(content, list):
            overall_str = ""
            for content_item in content:
                if content_item.get("type") == "image":
                    overall_str += "<image>" if not image_newline else "<image>\n"
                elif content_item.get("type") == "text":
                    overall_str += content_item.get("text")
                else:
                    raise ValueError(f"Unsupported content type: {type(content_item)}")
            processed_list.append(overall_str)
        elif isinstance(content, str):
            processed_list.append(content)
        else:
            raise ValueError(f"Unsupported content type: {type(content)}")
    
    return processed_list

def extract_system_message(conversation_list):
    if conversation_list[0]["role"] == "system":
        if isinstance(conversation_list[0]["content"], list):
            return conversation_list[0]["content"][0]["text"]
        else:
            return conversation_list[0]["content"]
    return None


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images