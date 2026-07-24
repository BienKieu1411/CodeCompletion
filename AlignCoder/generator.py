import re
import os
import time
import json
import torch
import requests
import logging
import numpy as np
from tqdm import tqdm
import torch.nn as nn
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import multiprocessing as mp
from functools import partial
from multiprocessing import Pool
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None
from tree_sitter import Language, Parser
from utils.eval_utils import is_identifier
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from utils.eval_utils import postprocess_code_lines, remove_comments


def process_single_item(example, output, language_name=None, ts_lib=None):
    language = Language(ts_lib, language_name)
    parser = Parser()
    parser.set_language(language)
    return remove_comments(postprocess_code_lines(example.left_context, output, parser, example.language))

class CustomDataset(Dataset):
    """
    A dataset class for code generation.

    Args:
        args: Configuration parameters.
        tokenizer: Tokenizer.
        examples: A collection of examples.
        retrieved_codeblocks: Retrieved code blocks.
    """
    def __init__(self, args, tokenizer, examples, retrieved_codeblocks, generation=False):
        self.args = args
        self.tokenizer = tokenizer
        self.examples = examples
        self.retrieved_codeblocks = retrieved_codeblocks
        self.generation = generation


    def __len__(self):
        return len(self.examples)

    def construct_prompts(self, example, retrieved_codeblocks):
        filter_codeblocks = []
        for x in retrieved_codeblocks:
            if x.file_path != "":
                filter_codeblocks.append(x)
            else:
                break
        
        crossfile_context = "\n\n".join([str(retrieved_codeblock) for retrieved_codeblock in filter_codeblocks])
        crossfile_context = self.tokenizer.encode(crossfile_context[:self.args.generator_max_crossfile_length * 10], add_special_tokens=False)[:self.args.generator_max_crossfile_length]
        path_context = f"\n\n# file path: {example.file_path}\n\n"
        path_context = self.tokenizer.encode(path_context, add_special_tokens=False)
        allowed_prompt_length = self.args.generator_max_context_length - (len(crossfile_context) + len(path_context) + 10)
        infile_context = self.tokenizer.encode(example.left_context, add_special_tokens=False)[-allowed_prompt_length:]
        prompt = self.tokenizer.decode(crossfile_context + path_context + infile_context)
        return prompt


    def __getitem__(self, idx):
        example = self.examples[idx]
        retrieved_codeblocks = self.retrieved_codeblocks[idx]
        prompt = self.construct_prompts(example, retrieved_codeblocks)
        prompt_ids = self.tokenizer.encode(prompt)[-self.args.generator_max_context_length:]

        if self.generation:
            padding_length = self.args.generator_max_context_length - len(prompt_ids)
            input_ids = [self.tokenizer.pad_token_id] * padding_length + prompt_ids
            return torch.tensor(input_ids)
        
        target_ids = self.tokenizer.encode(example.target_code, add_special_tokens=False)[:self.args.generator_max_generation_length]
        input_ids = prompt_ids + target_ids
        labels = [-100 for _ in prompt_ids] + target_ids
        padding_length = self.args.generator_max_context_length + self.args.generator_max_generation_length - len(input_ids)
        input_ids = [self.tokenizer.pad_token_id] * padding_length + input_ids
        labels = [-100] * padding_length + labels 
        return torch.tensor(input_ids), torch.tensor(labels)


class Model(nn.Module):
    def __init__(self, generator_model_path, tokenizer, max_generation_length=64):
        super(Model, self).__init__()
        self.base_model = AutoModelForCausalLM.from_pretrained(generator_model_path, torch_dtype=torch.float16)
        self.tokenizer = tokenizer
        self.max_generation_length = max_generation_length
        self.generator_model_path = generator_model_path

    def forward(self, inputs=None, labels=None, lang='python', weighted_keywords=False, sample_number=0):
        """
        Forward propagation method for calculating loss.
        :param inputs: Input data.
        :param labels: Label data.
        :return: The average loss per sample.
        """
        if inputs is None:
            return None
        
        if labels is not None:
            logits = self.base_model(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id))[0]
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            label_tokens = [self.tokenizer.convert_ids_to_tokens(id.item()) if id != -100 else '<pad>' for id in labels.reshape(-1)]
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction='none')
            if weighted_keywords:
                id_weight, first_token_weight = 3, 5
                weights = torch.tensor([first_token_weight if i < 1 else id_weight if is_identifier(token, lang) and any(c.isalpha() or c.isdigit() or c == '_' for c in token) else 1 for i, token in enumerate(label_tokens)], dtype=torch.float).cuda()
                loss = loss * weights
            loss_per_label = loss.reshape(labels.size(0), -1).sum(dim=1) / labels.ne(-100).sum(dim=1)
            return loss_per_label
        else:
            if sample_number:
                generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1) + self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id, do_sample=True, temperature=0.8, top_p=0.95)
            else:       
                generated_ids = self.base_model.generate(inputs, attention_mask=inputs.ne(self.tokenizer.pad_token_id), max_length=inputs.size(1) + self.max_generation_length, pad_token_id=self.tokenizer.pad_token_id)
            return generated_ids[:, inputs.size(1):]
                
                

class Generator:
    """
    Code generator class.

    Args:
        args: Configuration parameters.
    """
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if not args.disable_generator:
            self.model = Model(args.generator_model_path, self.tokenizer)
            self.model = torch.nn.DataParallel(self.model).cuda()
            self.model.eval()

        self.args = args

    def evaluate(self, examples, retrieved_codeblocks):
        """
        Evaluates the generated code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.

        Returns:
            A list of loss values.
        """
        losses = []
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)
        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm)
        with torch.no_grad():
            for batch in pbar:
                inputs, labels = [x.cuda() for x in batch]
                loss_per_label = self.model(inputs, labels, lang=examples[0].language, weighted_keywords=self.args.weighted_keywords)
                losses.extend(loss_per_label.tolist())
                current_ppl = np.exp(np.mean(losses))
                pbar.set_description(f"Loss/PPL: {np.mean(losses):.3f}/{current_ppl:.3f}")
                
        return losses
    
    def generate(self, examples, retrieved_codeblocks, max_generation_length, sample_number=0, deduplicated=False):
        """
        Generates code.

        Args:
            examples: A collection of examples.
            retrieved_codeblocks: Retrieved code blocks.
            max_generation_length: Maximum length of generation.

        Returns:
            A list of generated codes.
        """
        generated_codes = []
        if sample_number:
            examples = [example for example in examples for _ in range(sample_number)]
            retrieved_codeblocks = [codeblocks for codeblocks in retrieved_codeblocks for _ in range(sample_number)]
        
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)
        sampler = SequentialSampler(dataset)

        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.generator_batch_size, num_workers=self.args.num_workers)

        if hasattr(self.model, "module"):
            self.model.module.max_generation_length = max_generation_length
        else:
            self.model.max_generation_length = max_generation_length

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                if batch is not None:
                    output_temp = self.model(batch.cuda(), sample_number=sample_number)
                    if output_temp is not None:
                        generated_codes.append(output_temp)

        generated_codes = torch.cat(generated_codes, 0)
        outputs = [self.tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_codes]
        
        if not sample_number:
            return outputs
        else:

            if examples[0].language == 'python':
                ts_lib = "utils/build/python-lang-parser.so"
            else:
                ts_lib = "utils/build/java-lang-parser.so"

            language_name = examples[0].language

            with mp.Pool(processes=64) as pool:
                func = partial(process_single_item, language_name=language_name, ts_lib=ts_lib)
                outputs = pool.starmap(func, zip(examples, outputs))
                
            if not deduplicated:
                return outputs
            else:
                deduplicated_outputs, counts_per_batch = [], []
                for i in range(0, len(outputs), sample_number):
                    batch = outputs[i:i+sample_number]
                    set_temp, unique_batch = set(), list()
                    for item in batch:
                        if item not in set_temp and item.strip() != '':
                            set_temp.add(item)
                            unique_batch.append(item)
                    deduplicated_outputs.extend(unique_batch)
                    counts_per_batch.append(len(unique_batch))
                    
                return deduplicated_outputs, counts_per_batch

class vLLM_online_Generator:
    def __init__(self, args):
        
        openai_api_key = "EMPTY"
        openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)

        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.args = args

    def generate(self, examples, retrieved_codeblocks, temperature, top_p, sample_number=0, deduplicated=False):

        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=100000000, num_workers=self.args.num_workers)
        
        sampling_params = {"temperature": temperature, "max_tokens": self.args.generator_max_generation_length, "top_p": top_p}

        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                prompts = [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch]
                if sample_number != 0:
                    outputs = self.client.completions.create(model=self.args.generator_model_path, prompt=prompts, n=sample_number, timeout=7200, **sampling_params, seed=123)
                else:
                    outputs = self.client.completions.create(model=self.args.generator_model_path, prompt=prompts, timeout=7200, **sampling_params, seed=123)

        if sample_number != 0:
            outputs = [output.text for output in outputs.choices]
            examples = [example for example in examples for _ in range(sample_number)]
            language_name = examples[0].language

            if language_name == 'python':
                ts_lib = "utils/build/python-lang-parser.so"
            else:
                ts_lib = "utils/build/java-lang-parser.so"
            
            with mp.Pool(processes=64) as pool:
                func = partial(process_single_item, language_name=language_name, ts_lib=ts_lib)
                outputs = pool.starmap(func, zip(examples, outputs))
            
            if not deduplicated:
                return outputs
            else:
                deduplicated_outputs, counts_per_batch = [], []
                for i in range(0, len(outputs), sample_number):
                    batch = outputs[i:i + sample_number]
                    set_temp, unique_batch = set(), []
                    for item in batch:
                        if item not in set_temp and item.strip() != '':
                            set_temp.add(item)
                            unique_batch.append(item)
                    deduplicated_outputs.extend(unique_batch)
                    counts_per_batch.append(len(unique_batch))
                
                return deduplicated_outputs, counts_per_batch
        else:
            return [output.text for output in outputs.choices]


class vLLM_offline_Generator:

    def __init__(self, args):

        self.llm = LLM(model=args.generator_model_path, task='generate')

        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id == None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.args = args

    def generate(self, examples, retrieved_codeblocks, temperature, top_p, sample_number=0, deduplicated=True):

        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=100000000, num_workers=self.args.num_workers)
        
        if sample_number != 0:
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=self.args.generator_max_generation_length, n=sample_number, seed=123)
        else:
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=self.args.generator_max_generation_length, seed=123)
        
        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")
        with torch.no_grad():
            for batch in pbar:
                prompts = [self.tokenizer.decode(x, skip_special_tokens=True) for x in batch]
                outputs = self.llm.generate(prompts, sampling_params)
        
        if sample_number != 0:
            outputs = [completion.text for output in outputs for completion in output.outputs]
            examples = [example for example in examples for _ in range(sample_number)]

            if not deduplicated:
                return outputs
            else:
                if examples[0].language == 'python':
                    ts_lib = "utils/build/python-lang-parser.so"
                else:
                    ts_lib = "utils/build/java-lang-parser.so"

                language_name = examples[0].language

                with mp.Pool(processes=64) as pool:
                    func = partial(process_single_item, language_name=language_name, ts_lib=ts_lib)
                    outputs = pool.starmap(func, zip(examples, outputs))
            
                deduplicated_outputs, counts_per_batch = [], []
                for i in range(0, len(outputs), sample_number):
                    batch = outputs[i:i+sample_number]
                    set_temp = set()
                    unique_batch = []
                    for item in batch:
                        if item not in set_temp and item.strip() != '':
                            set_temp.add(item)
                            unique_batch.append(item)
                    deduplicated_outputs.extend(unique_batch)
                    counts_per_batch.append(len(unique_batch))
                return deduplicated_outputs, counts_per_batch
        else:
            return [output.outputs[0].text for output in outputs]


class HF_Generator:
    """Direct HuggingFace transformers generator (no vLLM dependency).
    Same generate() interface as vLLM_online_Generator for drop-in replacement."""

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.generator_model_path)
        self.tokenizer.model_max_length = 1e10
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            args.generator_model_path, torch_dtype=torch.float16
        ).cuda().eval()
        self.args = args

    @torch.no_grad()
    def generate(self, examples, retrieved_codeblocks, temperature=0, top_p=1.0,
                 sample_number=0, deduplicated=False):
        dataset = CustomDataset(self.args, self.tokenizer, examples, retrieved_codeblocks, generation=True)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler,
                                batch_size=self.args.generator_batch_size,
                                num_workers=self.args.num_workers)

        all_outputs = []
        pbar = tqdm(dataloader, disable=not self.args.enable_tqdm, desc="Generating")

        # Match vLLM SamplingParams behavior exactly
        gen_kwargs = dict(
            max_new_tokens=self.args.generator_max_generation_length,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=top_p, do_sample=True)
        else:
            gen_kwargs.update(do_sample=False)

        # Set seed=123 to match vLLM's seed=123
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)

        for batch in pbar:
            input_ids = batch.cuda()
            attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

            if sample_number > 0:
                # Repeat each prompt sample_number times (equivalent to vLLM n=sample_number)
                input_ids = input_ids.repeat_interleave(sample_number, dim=0)
                attention_mask = attention_mask.repeat_interleave(sample_number, dim=0)

            generated = self.model.generate(
                input_ids, attention_mask=attention_mask, **gen_kwargs
            )
            new_tokens = generated[:, input_ids.size(1):]
            texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in new_tokens]
            all_outputs.extend(texts)

        if sample_number == 0:
            return all_outputs

        # Post-process for sample_number > 0
        examples_dup = [ex for ex in examples for _ in range(sample_number)]
        language_name = examples[0].language
        if language_name == 'python':
            ts_lib = "utils/build/python-lang-parser.so"
        else:
            ts_lib = "utils/build/java-lang-parser.so"

        with mp.Pool(processes=max(1, self.args.num_workers)) as pool:
            func = partial(process_single_item, language_name=language_name, ts_lib=ts_lib)
            all_outputs = pool.starmap(func, zip(examples_dup, all_outputs))

        if not deduplicated:
            return all_outputs
        else:
            deduplicated_outputs, counts_per_batch = [], []
            for i in range(0, len(all_outputs), sample_number):
                batch = all_outputs[i:i + sample_number]
                set_temp, unique_batch = set(), []
                for item in batch:
                    if item not in set_temp and item.strip() != '':
                        set_temp.add(item)
                        unique_batch.append(item)
                deduplicated_outputs.extend(unique_batch)
                counts_per_batch.append(len(unique_batch))
            return deduplicated_outputs, counts_per_batch
