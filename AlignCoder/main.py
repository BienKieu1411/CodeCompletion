import os
import re
import gc
import time
import json
import copy
import torch
import random
import ctypes
import logging
import argparse
import numpy as np
import editdistance
from tqdm import tqdm
from api import GET_API
import multiprocessing as mp
from functools import partial
from torch.optim import AdamW
from multiprocessing import Pool
from prettytable import PrettyTable
from retriever import Retriever, tokenize
from concurrent.futures import ThreadPoolExecutor
from utils.eval_codereval import eval_codereval
from torch.utils.data import DataLoader, Dataset
from utils.eval_metric import compute_metric_stmt
from utils.eval_utils import remove_comments, cal_edit_sim
from bm25 import TaskSpecificBM25, split_api_blocks
from transformers import get_linear_schedule_with_warmup
from generator import Generator, vLLM_online_Generator, vLLM_offline_Generator, HF_Generator, process_single_item
from datasets import load_test_dataset, load_train_and_valid_dataset, construct_dataset, CodeBlock

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO, handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_random_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_random_seed()

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

def process_example(example, language):
    api_processor = GET_API([example], language)
    return api_processor.get()[0]

def retrieve_codeblocks(args, examples, bm25, retriever, dataset_name, is_training=False, inference_type=None):
    """
    Retrieves code blocks based on different inference types.
    :param args: An argument object containing configuration parameters.
    :param examples: Examples used for retrieval.
    :param bm25: An instance of the BM25 model.
    :param retriever: An instance of the retriever.
    :param dataset_name: The name of the dataset.
    :param is_training: Whether it is in training mode.
    :return: A list of retrieved code blocks.
    """
    if inference_type is None:
        inference_type = args.inference_type
    if inference_type == "baseline":
        return None, [[] for _ in range(len(examples))]

    bm25_topk, unixcoder_topk, context_len = 5, 5, 20
    if inference_type in ["bm25", "unixcoder", "unixcoder_with_rl"]:
        if dataset_name not in bm25:
            bm25[dataset_name] = TaskSpecificBM25(examples, args)

        if inference_type == "unixcoder":
            bm25_topk = 50 
        elif inference_type == "unixcoder_with_rl":
            bm25_topk = args.sample_number * 10 
            unixcoder_topk = args.sample_number 
        
        if args.enable_prediction:
            queries = ["\n".join([x for x in example.left_context.split("\n") if x.strip() != ""][-context_len:]) for example in examples]

            if args.add_api_blocks:
                candidate_codeblocks = bm25[dataset_name].query_with_api([x.task_id for x in examples], queries, topk=bm25_topk)
            else:
                candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)
        
            generations, counts = generator.generate(examples, candidate_codeblocks, args.temperature1, args.top_p1, args.number_sample, deduplicated=True)
            # generations, counts = generator.generate(examples, candidate_codeblocks, args.generator_max_generation_length, args.number_sample, deduplicated=True)
            candidate_batches = []
            start_idx = 0
            for count in counts:
                candidate_batches.append(generations[start_idx:start_idx+count])
                start_idx += count
            queries = [query + "\n\n" + "\n\n".join([f"# candidate answer {i + 1}:\n{candidate}" for i, candidate in enumerate(candidates)]) for query, candidates in zip(queries, candidate_batches)]

        elif args.enable_oracle:
            queries = ["\n".join([x for x in example.left_context.split("\n") if x.strip() != ""][-context_len:]) + example.target_code for example in examples]
        else:
            queries = ["\n".join([x for x in example.left_context.split("\n") if x.strip() != ""][-context_len:]) for example in examples]
        
        candidate_codeblocks = bm25[dataset_name].query([x.task_id for x in examples], queries, topk=bm25_topk)
        
        if args.add_api_blocks:
            for example, codeblock in zip(examples, candidate_codeblocks):
                api_blocks = bm25[dataset_name].api_blocks[example.task_id]
                codeblock.extend(api_blocks)

        if args.enable_repocoder and inference_type == 'unixcoder_with_rl':
            _, retrieved_codeblocks = retrieve_codeblocks(args, examples, bm25, retriever_RLCoder, dataset_name, inference_type="unixcoder") 
            generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)
            queries = [query + '\n' + prediction for query, prediction in zip(queries, generations)]

        if inference_type == "bm25":
            return queries, candidate_codeblocks
        elif inference_type == "unixcoder":
            return queries, retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
        elif inference_type == "unixcoder_with_rl":
            if is_training:
                if args.disable_stop_block:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk)
                else:
                    candidate_codeblocks = retriever.retrieve(queries, candidate_codeblocks, topk=unixcoder_topk-1)
                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]
            else:
                if not args.disable_stop_block:
                    candidate_codeblocks = [x + [CodeBlock("", "Don't need cross file context for completion", "", y.language, '')] for x,y in zip(candidate_codeblocks, examples)]
                candidate_codeblocks = retriever.retrieve(queries,  candidate_codeblocks, topk=unixcoder_topk)
                
            return queries, candidate_codeblocks
        
    raise ValueError("Unsupported inference type: {}".format(args.inference_type))


class CustomDataset(Dataset):
    def __init__(self, max_query_length, max_candidate_length, tokenizer, queries, candidates, labels):
        self.max_query_length = max_query_length
        self.max_candidate_length = max_candidate_length
        self.tokenizer = tokenizer
        self.queries = queries
        self.candidates = candidates
        self.labels = labels

    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query_tokens_id = tokenize(self.queries[idx], self.tokenizer, self.max_query_length, True)
        candidate_tokens_id = [tokenize(str(x), self.tokenizer, self.max_candidate_length, False) for x in self.candidates[idx]]
        return torch.tensor(query_tokens_id, dtype=torch.long), torch.tensor(candidate_tokens_id, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

def run(args):
    # repoeval_update_line_examples = load_test_dataset(args, "repoeval_update", "line_level")
    # repoeval_update_api_examples = load_test_dataset(args, "repoeval_update", "api_level")
    # recceval_examples = load_test_dataset(args, "recceval", "python")
    cceval_python_examples = load_test_dataset(args, "cceval", "python")
    cceval_java_examples = load_test_dataset(args, "cceval", "java")
    # codereval_python_examples = load_test_dataset(args, "codereval", "python")
    # codereval_java_examples = load_test_dataset(args, "codereval", "java")
    repoeval_line_examples = load_test_dataset(args, "repoeval", "line_level")
    repoeval_api_examples = load_test_dataset(args, "repoeval", "api_level")
    # repoeval_func_examples = load_test_dataset(args, "repoeval", "func_level")

    training_raw_data, eval_raw_data = load_train_and_valid_dataset()
    # eval_all_examples = construct_dataset(eval_raw_data, 100 if args.debug else 1000)
  
    all_eval_examples = {
        # "github_eval": eval_all_examples,
        # "repoeval_update_line": repoeval_update_line_examples,
        # "repoeval_update_api": repoeval_update_api_examples,
        "cceval_python": cceval_python_examples,
        "cceval_java": cceval_java_examples,
        # "codereval_python": codereval_python_examples,
        # "codereval_java": codereval_java_examples,
        "repoeval_line": repoeval_line_examples,
        "repoeval_api": repoeval_api_examples,
        # "repoeval_func": repoeval_func_examples,
        # "recceval": recceval_examples
        }
    
    global generator
    # generator = vLLM_offline_Generator(args)
    # generator = vLLM_online_Generator(args)
    # generator = Generator(args)
    generator = HF_Generator(args)
    retriever = Retriever(args)
    # global generator1
    # generator1 = Generator(args)
    
    if args.enable_repocoder:
        args_RLCoder = copy.deepcopy(args)
        args_RLCoder.retriever_model_path = args.rlcoder_model_path
        global retriever_RLCoder
        retriever_RLCoder = Retriever(args_RLCoder)
    
    if not args.enable_forward_generation:
        args.forward_generation_times = 1
    else:
        if args.forward_generation_times is None:
            args.forward_generation_times = 4

    bm25 = {}
    if args.eval:
        table = PrettyTable()
        # table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]
        table.field_names = ["Method", "Dataset", "Total Samples", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]
        # codereval_table = PrettyTable()
        # codereval_table.field_names = ["Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            print("Evaluating on {} dataset".format(name))
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
            all_retrieved_candidates = [[] for _ in range(len(examples))]
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                # Collect all candidates from each forward generation iteration
                for i in range(len(retrieved_codeblocks)):
                    for cb in retrieved_codeblocks[i]:
                        all_retrieved_candidates[i].append({
                            "file_path": cb.file_path,
                            "description": cb.description,
                            "code_content": cb.code_content
                        })
                # losses = generator1.evaluate(examples, retrieved_codeblocks)

                results = {"em": "-", "es": "-", "id_em": "-", "id_f1": "-"}
                if args.enable_generation:

                    generations = generator.generate(temp_examples, retrieved_codeblocks, temperature=0, top_p=1)
                    # generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)

                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if not os.path.exists(f"{args.output_dir}/{name}"):
                    os.makedirs(f"{args.output_dir}/{name}", exist_ok=True)
                
                with open(f"{args.output_dir}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")
                # Save predictions with retrieved candidates for visualization
                with open(f"{args.output_dir}/{name}/prediction_with_candidates.jsonl", "w", encoding="utf-8") as f_cand:
                    for example, temp_generation, candidates in zip(examples, temp_generations, all_retrieved_candidates):
                        f_cand.write(json.dumps({
                            "task_id": example.task_id,
                            "input": example.left_context[-512:],
                            "target": example.target_code,
                            "pred": temp_generation,
                            "candidates": candidates
                        }, ensure_ascii=False) + "\n")

                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, temp_generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in temp_generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(temp_generations,targets)])/len(temp_generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_update_line":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval_update/line_level/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_update_api":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/repoeval_update/api_level/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "recceval":
                    results = compute_metric_stmt(f"{args.output_dir}/{name}", "data/recceval/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
            
            # if 'codereval' in name:
                # codereval_table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            # else:
                # table.add_row(['raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            table.add_row(['raw', name, len(examples), results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
            print(table)
            # print(codereval_table)
        
    else:
        print("data_per_epoch:{}, batch_size:{}, sample_number:{}, epoch:{}, inner_epoch:{}, lr:{}".format(args.data_per_epoch, args.batch_size,args.sample_number,args.epoch,args.inner_epoch,args.lr))
        optimizer = AdamW(retriever.model.parameters(), lr=args.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch * 0.2, num_training_steps = args.data_per_epoch//args.batch_size * args.epoch * args.inner_epoch)
        evaluate_table = {}
        for name, examples in all_eval_examples.items():
            evaluate_table[name] = PrettyTable()
            if 'codereval' in name:
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "count", "all", "self", "slib", "plib", "class", "file", "project", "Time (sec)"]
            else:
                # evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "Loss", "PPL", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]
                evaluate_table[name].field_names = ["Epoch", "Method", "Dataset", "Total Samples", "EM", "ES", "ID_EM", "ID_F1", "Time (sec)"]       
        training_table = PrettyTable()
        training_table.field_names = ["Epoch", "Dataset", "Total Samples", "Rewards", "Training Loss", "Time (sec)"]
        retriever.model.eval()
        
        for name, examples in all_eval_examples.items():
            start_time = time.time()
            temp_examples = copy.deepcopy(examples)
            temp_generations = []
            all_retrieved_candidates = [[] for _ in range(len(examples))]
            for _ in range(args.forward_generation_times):
                _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                # Collect all candidates from each forward generation iteration
                for i in range(len(retrieved_codeblocks)):
                    for cb in retrieved_codeblocks[i]:
                        all_retrieved_candidates[i].append({
                            "file_path": cb.file_path,
                            "description": cb.description,
                            "code_content": cb.code_content
                        })
                # losses = generator1.evaluate(examples, retrieved_codeblocks)

                results = {"em": "-", "es": "-", "id_em": "-", "id_f1": "-"}
                if args.enable_generation:
                    generations = generator.generate(examples, retrieved_codeblocks, temperature=0, top_p=1)
                    # generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)
                        
                    if not temp_generations:
                        temp_generations = generations
                    else:
                        temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                    
                    for i in range(len(temp_examples)):
                        temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                        
            if args.enable_generation:

                if os.path.exists(f"{args.output_dir}/result_init/{name}") is False:
                    os.makedirs(f"{args.output_dir}/result_init/{name}", exist_ok=True)
                
                with open(f"{args.output_dir}/result_init/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                    for example, temp_generation in zip(examples, temp_generations):
                        f_pred.write(json.dumps({"task_id": example.task_id, "pred": temp_generation}) + "\n")
                # Save predictions with retrieved candidates for visualization
                with open(f"{args.output_dir}/result_init/{name}/prediction_with_candidates.jsonl", "w", encoding="utf-8") as f_cand:
                    for example, temp_generation, candidates in zip(examples, temp_generations, all_retrieved_candidates):
                        f_cand.write(json.dumps({
                            "task_id": example.task_id,
                            "input": example.left_context[-512:],
                            "target": example.target_code,
                            "pred": temp_generation,
                            "candidates": candidates
                        }, ensure_ascii=False) + "\n")

                if name == "cceval_python":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "cceval_java":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                elif name == "github_eval":
                    targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                    results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations, targets)])/len(generations)*100,4)
                elif name == "codereval_python":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                elif name == "codereval_java":
                    results = eval_codereval(f"{args.output_dir}/result_init/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                elif name == "repoeval_line":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "repoeval_api":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                elif name == "recceval":
                    results = compute_metric_stmt(f"{args.output_dir}/result_init/{name}", "data/recceval/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
            
            if 'codereval' in name:
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
            else:
                # evaluate_table[name].add_row(["init", 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
                evaluate_table[name].add_row(["init", 'raw', name, len(examples), results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])

            print(evaluate_table[name])
        
        for epoch in range(args.epoch):
            print("=" * 40 + "Epoch:{}".format(epoch) + "=" * 40)
            retriever.model.eval()
            start_time = time.time()
            results = {}
            results["Epoch"] = epoch
            training_examples = construct_dataset(training_raw_data, 100 if args.debug else args.data_per_epoch)
            queries, retrieved_codeblocks = retrieve_codeblocks(args, training_examples, bm25, retriever, "github_training_{}".format(epoch), True)

            # losses -> ppl
            if args.feedback_signal == "ppl":
                training_examples_dup = [x for x in training_examples for _ in range(args.sample_number)]
                training_codeblocks_dup = [[x] for y in retrieved_codeblocks for x in y]
                assert len(training_examples_dup) == len(training_codeblocks_dup)
                generator1 = Generator(args)
                losses = generator1.evaluate(training_examples_dup, training_codeblocks_dup)
                del generator1
                clean_memory(deep=True)
                # losses = generator.evaluate(training_examples_dup, training_codeblocks_dup)
            
            # losses -> avg_es
            elif args.feedback_signal == "avg_es":
                logging.info(f'feedback_signal: {args.feedback_signal}')
                training_examples_dup = [x for x in training_examples for _ in range(args.sample_number)]
                training_codeblocks_dup = [[x] for y in retrieved_codeblocks for x in y]
                assert len(training_examples_dup) == len(training_codeblocks_dup)
                outputs = generator.generate(training_examples_dup, training_codeblocks_dup, 0.8, 0.95, sample_number=4, deduplicated=False)
                training_examples_dup_mul = [x for x in training_examples for _ in range(args.sample_number * 4)]
                assert len(training_examples_dup_mul) == len(outputs)
                cceval_es_scores, repoeval_es_scores = [], []
                
                for training_example, output in zip(training_examples_dup_mul, outputs):
                    target_code = remove_comments(training_example.target_code)
                    output = remove_comments(output)

                    es = cal_edit_sim([target_code], [output])
                    cceval_es_scores.append(es)
                    
                    target_lines = [line.strip() for line in target_code.splitlines() if line.strip()]
                    target_str = '\n'.join(target_lines)
                    generation_lines = [line.strip() for line in output.splitlines() if line.strip()][:len(target_lines)]
                    generation_str = '\n'.join(generation_lines)
                    es = 1 - (editdistance.eval(target_str, generation_str) / max(len(target_str), len(generation_str)) if max(len(target_str), len(generation_str)) else 1.0)
                    repoeval_es_scores.append(es)

                # losses = [np.mean(es_scores_cceval[i:i + 4]) for i in range(0, len(es_scores_cceval), 4)]
                losses = [np.mean(repoeval_es_scores[i:i + 4]) for i in range(0, len(repoeval_es_scores), 4)]

            # avg_ppl -> losses
            elif args.feedback_signal == "avg_ppl":
                training_examples_dup = [x for x in training_examples for _ in range(args.sample_number * 4)]
                training_codeblocks_dup = [[x] for y in retrieved_codeblocks for x in y for _ in range(4)]
                assert len(training_examples_dup) == len(training_codeblocks_dup)
                generations = generator.generate(training_examples_dup, training_codeblocks_dup, 0.8, 0.95)
                assert len(training_examples_dup) == len(generations)
                generator1 = Generator(args)
                losses = generator1.evaluate(training_examples_dup, training_codeblocks_dup)
                losses = [np.mean(losses[i:i + 4]) for i in range(0, len(losses), 4)]
            
            if args.feedback_signal in ["ppl", "avg_ppl"]:
                labels = torch.tensor([x for x in losses]).view(-1, args.sample_number).argmin(-1)
            elif args.feedback_signal == "avg_es":
                labels = torch.tensor([x for x in losses]).view(-1, args.sample_number).argmax(-1)

            results["Total Samples"] = len(queries)
            results["Rewards"] = labels.float().mean().item()
        
            retriever.model.train()
            total_loss = 0
            dataset = CustomDataset(args.retriever_query_context_length, args.retriever_candidate_context_length, retriever.tokenizer, queries, retrieved_codeblocks, labels.tolist())
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
            for inner_epoch in range(args.inner_epoch):
                for batch in dataloader:
                    source_ids, doc_ids, labels = [x.cuda() for x in batch]
                    queries_embeddings = retriever(source_ids)
                    doc_texts_embeddings = retriever(doc_ids.view(-1, doc_ids.shape[-1])).view(source_ids.shape[0], args.sample_number, -1)
                    logits = torch.einsum("ab, acb->ac", queries_embeddings, doc_texts_embeddings) * 20
                    loss = torch.nn.CrossEntropyLoss()(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(retriever.model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    total_loss += loss.item()

                if args.enable_sft:
                    retriever.model.eval()
                    for name, examples in all_eval_examples.items():
                        start_time = time.time()
                        temp_examples = copy.deepcopy(examples)
                        temp_generations = []
                        for _ in range(args.forward_generation_times):

                            _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                            # generator1 = Generator(args)
                            # losses = generator1.evaluate(examples, retrieved_codeblocks)
                            # del generator1

                            results = {"em": "-", "es": "-", "id_em": "-", "id_f1": "-"}
                            if args.enable_generation:
                                generations = generator.generate(examples, retrieved_codeblocks, temperature=0, top_p=1)
                                if not temp_generations:
                                    temp_generations = generations
                                else:
                                    temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                                
                                for i in range(len(temp_examples)):
                                    temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                                    
                        if args.enable_generation:

                            if os.path.exists(f"{args.output_dir}/result_{inner_epoch}/{name}") is False:
                                os.makedirs(f"{args.output_dir}/result_{inner_epoch}/{name}", exist_ok=True)

                            with open(f"{args.output_dir}/result_{inner_epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                                for example, generation in zip(examples, temp_generations):
                                    f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")

                            if name == "cceval_python":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "cceval_java":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                            elif name == "github_eval":
                                targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                                results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y),len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                            elif name == "codereval_python":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                            elif name == "codereval_java":
                                results = eval_codereval(f"{args.output_dir}/result_{inner_epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                            elif name == "repoeval_line":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "repoeval_api":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                            elif name == "recceval":
                                results = compute_metric_stmt(f"{args.output_dir}/result_{inner_epoch}/{name}", "data/recceval/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                        if 'codereval' in name:
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                        else:
                            # evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
                            evaluate_table[name].add_row([inner_epoch, 'raw', name, len(examples), results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
                        print(evaluate_table[name])
                    
                    retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")
                    retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{inner_epoch}")

            results["Training Loss"] = total_loss / len(dataloader) / args.inner_epoch
            results["Time (sec)"] = round(time.time() - start_time, 1)
            training_table.add_row([results["Epoch"], "github_training_{}".format(epoch), results["Total Samples"], results["Rewards"], results["Training Loss"], results["Time (sec)"]])
            print(training_table)

            retriever.model.eval()
            for name, examples in all_eval_examples.items():
                start_time = time.time()
                temp_examples = copy.deepcopy(examples)
                temp_generations = []
                all_retrieved_candidates = [[] for _ in range(len(examples))]
                for _ in range(args.forward_generation_times):
                    _, retrieved_codeblocks = retrieve_codeblocks(args, temp_examples, bm25, retriever, name)
                    # Collect all candidates from each forward generation iteration
                    for i in range(len(retrieved_codeblocks)):
                        for cb in retrieved_codeblocks[i]:
                            all_retrieved_candidates[i].append({
                                "file_path": cb.file_path,
                                "description": cb.description,
                                "code_content": cb.code_content
                            })
                    # losses = generator1.evaluate(examples, retrieved_codeblocks)

                    results = {"em": "-", "es": "-", "id_em": "-", "id_f1": "-"}
                    if args.enable_generation:
                        generations = generator.generate(examples, retrieved_codeblocks, temperature=0, top_p=1)
                        # generations = generator.generate(examples, retrieved_codeblocks, args.generator_max_generation_length)
                        # losses = generator1.evaluate(training_examples_dup, training_codeblocks_dup)
                        if not temp_generations:
                            temp_generations = generations
                        else:
                            temp_generations = [temp_generations[i] + generations[i] for i in range(len(generations))]
                        
                        for i in range(len(temp_examples)):
                            temp_examples[i].left_context = examples[i].left_context + temp_generations[i]
                            
                if args.enable_generation:

                    if os.path.exists(f"{args.output_dir}/result_{epoch}/{name}") is False:
                        os.makedirs(f"{args.output_dir}/result_{epoch}/{name}", exist_ok=True)
                
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction.jsonl", "w", encoding="utf-8") as f_pred:
                        for example, generation in zip(examples, temp_generations):
                            f_pred.write(json.dumps({"task_id": example.task_id, "pred": generation}) + "\n")
                    # Save predictions with retrieved candidates for visualization
                    with open(f"{args.output_dir}/result_{epoch}/{name}/prediction_with_candidates.jsonl", "w", encoding="utf-8") as f_cand:
                        for example, generation, candidates in zip(examples, temp_generations, all_retrieved_candidates):
                            f_cand.write(json.dumps({
                                "task_id": example.task_id,
                                "input": example.left_context[-512:],
                                "target": example.target_code,
                                "pred": generation,
                                "candidates": candidates
                            }, ensure_ascii=False) + "\n")

                    if name == "cceval_python":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/python/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "cceval_java":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/cceval/java/test.jsonl", language="java", ts_lib="utils/build/java-lang-parser.so")
                    elif name == "github_eval":
                        targets, generations = ["".join(x.target_code.split()) for x in examples], ["".join(x.split()) for x in generations]
                        results["em"] = round(sum([1 if x[:min(len(y),len(x))] == y[:min(len(y), len(x))] else 0 for x,y in zip(generations,targets)])/len(generations)*100,4)
                    elif name == "codereval_python":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/python/CEPythonRaw.jsonl', language='python', do_codereval=args.do_codereval)
                    elif name == "codereval_java":
                        results = eval_codereval(f"{args.output_dir}/result_{epoch}/{name}", 'data/codereval/java/CEJavaRaw.jsonl', language='java', do_codereval=args.do_codereval)
                    elif name == "repoeval_line":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/line_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "repoeval_api":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/repoeval/api_level/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")
                    elif name == "recceval":
                        results = compute_metric_stmt(f"{args.output_dir}/result_{epoch}/{name}", "data/recceval/test.jsonl", language="python", ts_lib="utils/build/python-lang-parser.so")

                if 'codereval' in name:
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["count"], results["all"], results["self"], results["slib"], results["plib"], results["class"], results["file"], results["project"], round(time.time() - start_time, 1)])
                else:
                    # evaluate_table[name].add_row([epoch, 'raw', name, len(examples), f"{np.mean(losses):.4f}", f"{np.exp(np.mean(losses)):.4f}", results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
                    evaluate_table[name].add_row([epoch, 'raw', name, len(examples), results["em"], results["es"], results["id_em"], results["id_f1"], round(time.time() - start_time, 1)])
                print(evaluate_table[name])

            retriever.model.module.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            retriever.tokenizer.save_pretrained(f"{args.output_dir}/retriever_cpkt/result_{epoch}")
            

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_model_path", default="deepseek-ai/deepseek-coder-1.3b-base", type=str, help="Generator model path")
    parser.add_argument("--generator_batch_size_per_gpu", default=32, type=int, help="generator batch size per GPU")
    parser.add_argument("--generator_max_crossfile_length", default=512, type=int, help="Maximum cross-file length for the generator")
    parser.add_argument("--generator_max_context_length", default=1024, type=int, help="Maximum context length for the generator")
    parser.add_argument("--generator_max_generation_length", default=64, type=int, help="Maximum generation length for the generator")
    parser.add_argument("--disable_generator", action="store_true", help="Disable the generator")

    parser.add_argument("--retriever_model_path", default="microsoft/unixcoder-base", type=str, help="Retriever model path")
    parser.add_argument("--retriever_batch_size_per_gpu", default=64, type=int, help="Retriever batch size per GPU")
    parser.add_argument("--disable_retriever", action="store_true", help="Disable the retriever")
    parser.add_argument("--retriever_query_context_length", default=256, type=int, help="Retriever query context length")
    parser.add_argument("--retriever_candidate_context_length", default=512, type=int, help="Retriever candidate context length")

    parser.add_argument("--inference_type", default="baseline", type=str, help="Inference type")
    parser.add_argument("--output_dir", default="results/baseline", type=str, help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation")
    parser.add_argument("--enable_tqdm", action="store_true", help="Enable progress bar")
    parser.add_argument("--enable_generation", action="store_true", help="Enable generation")
    parser.add_argument("--debug", action="store_true", help="Debug mode, use a small dataset")

    parser.add_argument("--num_workers", default=20, type=int, help="Number of CPU cores")
    parser.add_argument("--weighted_keywords", action="store_true", help="Weight keywords when calculating loss during training")
    parser.add_argument("--enable_fixed_block", action="store_true", help="Use fixed length blocks when building candidates")
    parser.add_argument("--enable_sft", action="store_true", help="Train using supervised learning methods")
    parser.add_argument("--disable_stop_block", action="store_true", help="Disable the stop block")
    parser.add_argument("--enable_repocoder", action="store_true", help="Use the repocoder method during generation")

    parser.add_argument("--enable_prediction", action="store_true", help="enable prediction")
    parser.add_argument("--enable_oracle", action="store_true", help="enable oracle")
    parser.add_argument("--number_sample", default=8, type=int, help="Number of sample times")
    parser.add_argument("--vllm_generator_batch_size_per_gpu", default=10000000, type=int, help="vllm generator batch size per GPU")
    parser.add_argument("--feedback_signal", default="ppl", type=str, help="feedback signal")
    parser.add_argument("--add_api_blocks", action="store_true", help="add_api_blocks")
    parser.add_argument("--temperature1", default=0.8, type=float, help="temperature")
    parser.add_argument("--top_p1", default=0.95, type=float, help="top_p")    

    parser.add_argument("--rlcoder_model_path", default="microsoft/unixcoder-base", type=str, help="Stage 1 model for repocoder")
    parser.add_argument("--do_codereval", action="store_true", help="Execute codereval evaluation in docker")
    parser.add_argument("--enable_forward_generation", action="store_true", help="Use progressive generation methods during inference")
    parser.add_argument("--forward_generation_times", default=4, type=int, help="Number of times for progressive generation")
    parser.add_argument("--epoch", default=20, type=int, help="Number of training epochs")
    parser.add_argument("--inner_epoch", default=1, type=int, help="Number of inner training epochs")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size")
    parser.add_argument("--sample_number", default=10, type=int, help="Number of samples")
    parser.add_argument("--data_per_epoch", default=2000, type=int, help="Amount of data per epoch")
    parser.add_argument("--lr", default=5e-5, type=float, help="Learning rate")

    print("Number of GPUs:", torch.cuda.device_count())
    args = parser.parse_args()

    args.generator_batch_size = args.generator_batch_size_per_gpu * torch.cuda.device_count()
    args.retriever_batch_size = args.retriever_batch_size_per_gpu * torch.cuda.device_count()
    args.vllm_generator_batch_size = args.vllm_generator_batch_size_per_gpu * torch.cuda.device_count()
    run(args)