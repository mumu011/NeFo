# Copyright 2025 the LlamaFactory team.
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
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np
from datasets import Dataset, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
    SelfSupervisedDatasetProcessor,
    SubSamplesDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)


def _filter_negation_in_text(text: str, negation_words: list[str]) -> str:
    """
    在文本中检测并过滤否定词（只处理<image>标记之后的内容，避免误匹配prompt）
    
    Args:
        text: 输入文本
        negation_words: 否定词列表
        
    Returns:
        如果存在否定词，返回过滤后的文本；如果不存在否定词，返回空字符串
        
    Example:
        Input: "Answer Yes or No.<image>\nThere are no planes."
        Output: "Answer Yes or No.<image>\nThere are <Neg_Mask> planes."
        (只替换<image>之后的"no"，保留prompt中的"No"和换行符"\n")
    """
    if not text:
        return ""
    
    # 如果文本中包含<image>，则只处理<image>之后的内容
    if '<image>' in text:
        # 找到最后一个<image>的位置（因为可能有多个图像）
        parts = text.split('<image>')
        prompt_parts = parts[:-1]  # prompt部分（<image>之前的所有内容）
        question_part = parts[-1]   # question部分（最后一个<image>之后的内容）
        
        # 只在question部分检测否定词
        has_negation = False
        for neg_word in negation_words:
            pattern = r'\b' + re.escape(neg_word) + r'\b'
            if re.search(pattern, question_part, flags=re.IGNORECASE):
                has_negation = True
                break
        
        # 如果question部分没有否定词，返回空字符串
        if not has_negation:
            return ""
        
        # 只对question部分进行否定词替换
        filtered_question = question_part
        for neg_word in negation_words:
            # 使用捕获组保留前面的空白字符（空格、换行符等）
            pattern = r'(\s+)' + re.escape(neg_word) + r'\b'
            filtered_question = re.sub(pattern, r'\1<Neg_Mask>', filtered_question, flags=re.IGNORECASE)
        
        # 只清理连续的多个空格/制表符，保留换行符和首尾空白
        filtered_question = re.sub(r'[ \t]{2,}', ' ', filtered_question)
        
        # 重新组合：prompt + <image> + filtered_question
        result = '<image>'.join(prompt_parts) + '<image>' + filtered_question
        return result
    
    else:
        # 如果没有<image>标记，使用原有逻辑（向后兼容）
        # 检查是否包含任何否定词
        has_negation = False
        for neg_word in negation_words:
            pattern = r'\b' + re.escape(neg_word) + r'\b'
            if re.search(pattern, text, flags=re.IGNORECASE):
                has_negation = True
                break
        
        # 如果没有否定词，返回空字符串
        if not has_negation:
            return ""
        
        # 对整个文本进行否定词替换
        filtered_text = text
        for neg_word in negation_words:
            # 使用捕获组保留前面的空白字符（空格、换行符等）
            pattern = r'(\s+)' + re.escape(neg_word) + r'\b'
            filtered_text = re.sub(pattern, r'\1<Neg_Mask>', filtered_text, flags=re.IGNORECASE)
        
        # 保留换行符和首尾空白，只清理连续的多个空格/制表符
        filtered_text = re.sub(r'[ \t]{2,}', ' ', filtered_text)
        return filtered_text


def _expand_sub_samples_to_list(example: dict) -> list[dict]:
    """
    将包含sub_samples的单个样本展开为多个独立样本列表
    
    Args:
        example: 包含sub_samples的样本 {"question_id": [...], "sub_samples": [...]}
        
    Returns:
        展开后的样本列表
    """
    question_id = example.get("question_id", "")
    sub_samples = example.get("sub_samples", [])
    
    if not sub_samples:
        logger.warning_rank0(f"No sub_samples found in example {question_id}")
        return [{"messages": [], "images": [], "question_id": question_id}]
    
    expanded_samples = []
    for i, sub_sample in enumerate(sub_samples):
        expanded_sample = {
            "messages": sub_sample.get("messages", []),
            "images": sub_sample.get("images", []),
            "question_id": sub_sample.get("question_id", f"{question_id}_{i:02d}"),
            "parent_id": question_id,
            "sub_index": i,
            "total_subs": len(sub_samples),
        }
        expanded_samples.append(expanded_sample)
    
    return expanded_samples


def _check_negation_in_text(text: str, negation_words: list[str]) -> bool:
    """
    检查文本中是否包含否定词（只检查<image>之后的内容）
    
    Args:
        text: 输入文本
        negation_words: 否定词列表
        
    Returns:
        True if有否定词，False otherwise
    """
    if not text:
        return False
    
    # 如果有<image>标记，只检查最后一个<image>之后的内容
    if '<image>' in text:
        parts = text.split('<image>')
        question_part = parts[-1]
    else:
        question_part = text
    
    # 检查是否包含任何否定词
    for neg_word in negation_words:
        pattern = r'\b' + re.escape(neg_word) + r'\b'
        if re.search(pattern, question_part, flags=re.IGNORECASE):
            return True
    
    return False


def _process_negation_filtering(examples: dict, negation_words: list[str]) -> dict:
    """
    标记包含否定词的样本，实际替换将在token层面进行，确保长度一致
    
    Args:
        examples: 数据集批次
        negation_words: 否定词列表
        
    Returns:
        处理后的数据集批次，添加_has_negation标记
    """
    # 处理messages格式的数据（多模态对话数据）
    if "messages" in examples:
        has_negation_list = []
        
        for messages in examples["messages"]:
            has_negation = False
            
            for msg in messages:
                if msg.get("role") == "user" and "content" in msg:
                    if _check_negation_in_text(msg["content"], negation_words):
                        has_negation = True
                        break
            
            has_negation_list.append(has_negation)
        
        # 添加标记字段（用于processor中的token层面替换）
        examples["_has_negation"] = has_negation_list
        # 保留messages_aug为空，表示使用token层面替换
        examples["messages_aug"] = [[] for _ in examples["messages"]]
    
    return examples


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.11.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            # trust_remote_code=model_args.trust_remote_code,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    # 只对train dataset进行特殊处理，跳过eval dataset  
    if not is_eval:
        # 检测sub_samples格式但保持嵌套结构，不展开
        first_example = next(iter(dataset))
        if "sub_samples" in first_example:
            logger.info_rank0(f"Detected sub_samples format, dataset size: {len(dataset)}")
            logger.info_rank0("Keeping nested sub_samples structure for batch processing...")
            # 不需要展开，保持嵌套结构，让processor处理
            
        # 否定词过滤处理 - 对标准sharegpt格式数据
        elif getattr(data_args, 'enable_negation_filtering', False) and not data_args.streaming:
            negation_words = getattr(data_args, 'negation_words', ["no", "not", "without", "non-"])
            logger.info_rank0(f"Applying negation filtering with words: {negation_words}")
            
            dataset = dataset.map(
                lambda examples: _process_negation_filtering(examples, negation_words),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                desc="Filtering negation words",
            )
            
            logger.info_rank0("Negation filtering completed")
    else:
        logger.info_rank0(f"Skipping data augmentation for eval dataset: {dataset_attr.dataset_name}")

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: Optional[list[str]],
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "ttl", "vl_ttl"],
    merge: bool = True,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args, is_eval)

    if merge:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)
    else:
        return datasets


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "ttl", "vl_ttl"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
    current_dataset_names: Optional[list[str]] = None,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor
    elif stage in ["ttl", "vl_ttl"] and not do_generate:
        # 检测是否使用sub_samples格式数据
        # 使用当前处理的数据集名称而不是总是检查data_args.dataset
        if current_dataset_names and 'sub_negated' in str(current_dataset_names):
            dataset_processor_class = SubSamplesDatasetProcessor
        else:
            dataset_processor_class = SelfSupervisedDatasetProcessor
    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "ttl", "vl_ttl"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Optional[Union["Dataset", "IterableDataset"]]:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    # Get dataset processor for the preprocessed data
    # 根据是否是eval来决定检查哪个数据集名称
    current_dataset_names = data_args.eval_dataset if is_eval else data_args.dataset
    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, 
        do_generate=(training_args.predict_with_generate and is_eval),
        current_dataset_names=current_dataset_names
    )
    column_names = list(next(iter(dataset)).keys())
    
    kwargs = {}
    if not data_args.streaming:
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )


    
    # 保留错误检查，但不输出详细样例
    if training_args.should_log:
        try:
            next(iter(dataset))  # 只检查数据集是否为空，不打印详细信息
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto", "ttl", "vl_ttl"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset"):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage, is_eval=False)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset, model_args, data_args, training_args, stage, merge=training_args.do_predict, is_eval=True
        )

    with training_args.main_process_first(desc="pre-process dataset"):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=False
        )
        if isinstance(eval_dataset, dict):
            for eval_name, eval_data in eval_dataset.items():
                eval_dataset[eval_name] = _get_preprocessed_dataset(
                    eval_data, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
                )
        else:
            eval_dataset = _get_preprocessed_dataset(
                eval_dataset, data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        dataset_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)
