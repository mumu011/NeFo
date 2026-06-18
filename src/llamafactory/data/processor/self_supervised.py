from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ..data_utils import Role
from .processor_utils import DatasetProcessor, infer_seqlen

if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput

logger = logging.get_logger(__name__)


class SelfSupervisedDatasetProcessor(DatasetProcessor):
    def _create_token_level_augmentation(self, input_ids: list[int]) -> list[int]:
        """
        在token层面创建增强版本：替换否定词token为<Neg_Mask> token
        保证长度一致（1对1替换）
        
        Args:
            input_ids: 原始token IDs
            
        Returns:
            增强后的token IDs（长度与原始一致）
        """
        if not getattr(self.data_args, 'enable_negation_filtering', False):
            return input_ids.copy()
        
        # 获取<Neg_Mask> token ID
        neg_mask_id = self.tokenizer.convert_tokens_to_ids('<Neg_Mask>')
        if neg_mask_id == self.tokenizer.unk_token_id:
            logger.warning_rank0("<Neg_Mask> token not found in tokenizer!")
            return input_ids.copy()
        
        # 获取否定词的 token IDs。不假定否定词一定是单 token，把编码得到的所有 token id 都纳入处理；
        # 但排除仅解码为空白的 token（如 " non" 里的空格），否则会误替换句中所有空格，出现 "USER:<Neg_Mask>"。
        negation_words = getattr(self.data_args, 'negation_words', ["no", "not", "without"])
        negation_token_ids = set()
        for neg_word in negation_words:
            for prefix in (" ", ""):  # 带空格（句中）/ 不带空格（句首等）
                tokens = self.tokenizer.encode(prefix + neg_word, add_special_tokens=False)
                for tid in tokens:
                    piece = self.tokenizer.decode([tid]).strip()
                    if piece:  # 只加入非空白 token，避免把句中空格等也替换成 <Neg_Mask>
                        negation_token_ids.add(tid)

        # 复制 input_ids 并进行 1 对 1 替换
        input_ids_aug = input_ids.copy()
        for idx, token_id in enumerate(input_ids_aug):
            if token_id in negation_token_ids:
                input_ids_aug[idx] = neg_mask_id
        
        return input_ids_aug
    
    def _encode_data_example(
            self,
            prompt: list[dict[str, str]],
            response: list[dict[str, str]],
            system: Optional[str],
            tools: Optional[str],
            images: list["ImageInput"],
            videos: list["VideoInput"],
            audios: list["AudioInput"],
    ) -> tuple[list[int], list[int]]:

        if len(response) == 1:
            messages = prompt + response
        else:
            messages = prompt + [{"role": Role.ASSISTANT.value, "content": ""}]
        # print(messages)    # [{'role': 'user', 'content': xxx}, {'role': 'assistant', 'content': xxx}]
        messages = self.template.mm_plugin.process_messages(messages, images, videos, audios, self.processor)
        input_ids, labels = self.template.encode_oneturn(self.tokenizer, messages, system, tools)
        ## inputs_ids: [319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 529, 3027, 29958, 5618, 29915, 29879, 297, 445, 1967, 29973, 1815, 366, 8453, 278, 1737, 12122, 5680, 29973, 319, 1799, 9047, 13566, 29901]
        
        if self.template.efficient_eos:
            labels += [self.tokenizer.eos_token_id]

        input_ids, _ = self.template.mm_plugin.process_token_ids(input_ids, None, images, audios, videos,
                                                                 self.tokenizer, self.processor)
        ## inputs_ids: [1, 319, 13563, 1546, 263, 12758, 1404, 322, 385, 23116, 21082, 20255, 29889, 450, 20255, 4076, 8444, 29892, 13173, 29892, 322, 1248, 568, 6089, 304, 278, 1404, 29915, 29879, 5155, 29889, 3148, 1001, 29901, 29871, -200, 1724, 29915, 29879, 297, 445, 1967, 29973, 1815, 366, 8453, 278, 1737, 12122, 5680, 29973, 319, 1799, 9047, 13566, 29901]

        source_len, target_len = infer_seqlen(len(input_ids), len(labels), self.data_args.cutoff_len)
        input_ids = input_ids[:source_len]
        labels = labels[:target_len]
        return input_ids, labels


    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # 若对齐后的数据为子样本格式，则委托给 SubSamplesDatasetProcessor 处理，
        # 避免 batched map 时列长度与 batch 不一致（例如 input_ids_aug 长度 1）
        if "_sub_samples" in examples:
            try:
                delegate = SubSamplesDatasetProcessor(
                    template=self.template,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    data_args=self.data_args,
                )
                return delegate.preprocess_dataset(examples)
            except Exception as e:
                logger.error_rank0(f"Failed to delegate _sub_samples to SubSamplesDatasetProcessor: {e}")
                raise

        # build inputs with format `<bos> X` and labels with format `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            # 处理原始数据
            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(input_ids)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])
            
            # 🔑 新策略：始终在token层面替换否定词，而不是文本层面
            # 这样可以保证input_ids和input_ids_aug长度完全一致（1对1替换）
            if getattr(self.data_args, 'enable_negation_filtering', False):
                # 使用token层面替换：直接替换token ID，保证长度一致
                input_ids_aug = self._create_token_level_augmentation(input_ids)
                model_inputs["input_ids_aug"].append(input_ids_aug)
            elif "_prompt_aug" in examples and examples["_prompt_aug"][i]:
                # 如果有预先生成的文本层面增强数据（向后兼容）
                input_ids_aug, _ = self._encode_data_example(
                    prompt=examples["_prompt_aug"][i],
                    response=examples["_response_aug"][i],
                    system=examples["_system"][i],
                    tools=examples["_tools"][i],
                    images=examples["_images"][i] or [],  # 图像保持相同
                    videos=examples["_videos"][i] or [],
                    audios=examples["_audios"][i] or [],
                )
                model_inputs["input_ids_aug"].append(input_ids_aug)
            else:
                # 没有增强数据，添加空列表标识
                model_inputs["input_ids_aug"].append([])
            
            # 保留 question_id 字段
            if "_question_id" in examples:
                model_inputs["question_id"].append(examples["_question_id"][i])
            # 不再处理 ground_truth 字段

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
        print("label_ids:\n{}".format(example["labels"]))
        print("labels:\n{}".format(self.tokenizer.decode(example["labels"], skip_special_tokens=False)))


class SubSamplesDatasetProcessor(SelfSupervisedDatasetProcessor):
    """
    处理嵌套sub_samples格式数据的processor
    继承SelfSupervisedDatasetProcessor以复用_encode_data_example方法
    输入：{"_sub_samples": [...], "_question_id": "..."}
    输出：批量格式的K个子问题的处理结果
    """
    
    def _check_has_negation(self, content: str) -> bool:
        """检查内容中是否包含否定词（只检测，不替换）"""
        from ..loader import _check_negation_in_text
        
        negation_words = getattr(self.data_args, 'negation_words', ["no", "not", "without", "non-"])
        if not getattr(self.data_args, 'enable_negation_filtering', False):
            return False
            
        return _check_negation_in_text(content, negation_words)
    
    def _process_sub_sample(self, sub_sample: dict) -> dict:
        """处理单个子问题，返回处理后的数据"""
        messages = sub_sample.get("messages", [])
        images = sub_sample.get("images", [])
        question_id = sub_sample.get("question_id", "")
    # 不再读取 ground_truth 字段
        
        if not messages:
            return None
            
        # 构建标准格式
        prompt = messages[:-1] if len(messages) > 1 else []
        response = messages[-1:] if messages else []
        
        # 处理原始数据
        try:
            input_ids, labels = self._encode_data_example(
                prompt=prompt,
                response=response,
                system="",  # sub_sample通常不包含system
                tools="",
                images=images or [],
                videos=[],
                audios=[],
            )
        except Exception as e:
            logger.error_rank0(f"❌ Failed to encode sub-sample {question_id}: {e}")
            raise
        
        # 🔑 新策略：在token层面进行否定词替换（1对1），而不是文本层面
        input_ids_aug = []
        if prompt and prompt[-1].get("role") == "user":
            user_content = prompt[-1].get("content", "")
            has_negation = self._check_has_negation(user_content)
            
            if has_negation:
                # ✅ Token层面替换：保证长度一致
                input_ids_aug = self._create_token_level_augmentation(input_ids)
            else:
                # 没有否定词，使用原始input_ids
                input_ids_aug = input_ids[:]
        else:
            input_ids_aug = input_ids[:]
            
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": input_ids,  # self-supervised: labels = inputs
            "images": images,
            "question_id": question_id,
            "input_ids_aug": input_ids_aug,
        }
    
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """
        处理包含_sub_samples的数据
        
        输入examples: {
            "_sub_samples": [
                [sub_sample1, sub_sample2, ...],  # 第一个parent的子问题
                [sub_sample1, sub_sample2, ...],  # 第二个parent的子问题
                ...
            ],
            "_question_id": ["parent_id1", "parent_id2", ...]
        }
        
        输出：标准格式，但带有parent分组信息
        """
        model_inputs = defaultdict(list)
        
        for i in range(len(examples["_sub_samples"])):
            sub_samples = examples["_sub_samples"][i]
            parent_id = examples["_question_id"][i]
            
            # 为当前parent准备子问题数据列表
            parent_input_ids = []
            parent_attention_masks = []
            parent_labels = []
            parent_images = []
            parent_question_ids = []
            parent_input_ids_aug = []
            # 不再收集 ground_truth
            
            for j, sub_sample in enumerate(sub_samples):
                try:
                    processed = self._process_sub_sample(sub_sample)
                    if processed is None:
                        continue
                        
                    parent_input_ids.append(processed["input_ids"])
                    parent_attention_masks.append(processed["attention_mask"])
                    parent_labels.append(processed["labels"])
                    parent_images.append(processed["images"])
                    parent_question_ids.append(processed["question_id"])
                    parent_input_ids_aug.append(processed["input_ids_aug"])
                    # 不再收集 ground_truth
                        
                except Exception as e:
                    logger.error_rank0(f"❌ Error processing sub-sample in parent {parent_id}: {e}")
                    continue
            
            # 将当前parent的所有子问题作为一个组添加到结果中
            if parent_input_ids:  # 确保有有效数据
                model_inputs["input_ids"].append(parent_input_ids)
                model_inputs["attention_mask"].append(parent_attention_masks) 
                model_inputs["labels"].append(parent_labels)
                model_inputs["images"].append(parent_images)
                model_inputs["question_id"].append(parent_question_ids)
                # 确保与其他列的长度一致：为每个parent追加对应的aug列表
                model_inputs["input_ids_aug"].append(parent_input_ids_aug)
        return model_inputs
    
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        # 处理SubSamples格式：嵌套列表结构
        if isinstance(example.get("input_ids"), list) and len(example["input_ids"]) > 0:
            if isinstance(example["input_ids"][0], list):  # 嵌套列表结构
                print(f"SubSamples batch with {len(example['input_ids'])} sub-questions:")
                for i, input_ids in enumerate(example["input_ids"]):
                    question_id = example.get("question_id", [["unknown"]])[0][i] if len(example.get("question_id", [])) > 0 else "unknown"
                    print(f"Sub-question {i} (ID: {question_id}):")
                    print("input_ids:\n{}".format(input_ids))
                    print("inputs:\n{}".format(self.tokenizer.decode(input_ids, skip_special_tokens=False)))
                    print("---")
            else:  # 标准格式
                print("input_ids:\n{}".format(example["input_ids"]))
                print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))

