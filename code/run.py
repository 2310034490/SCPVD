from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from vulnerability_rules import VulnerabilityRules

import sys

sys.path.append('/home/SCPVD/')
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm, trange
import multiprocessing
from model import Model
#from model_CNNResNet import Model

cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

import parserTool.parse as ps
from c_cfg import C_CFG
from parserTool.utils import remove_comments_and_docstrings
from parserTool.parse import Lang


def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            # 处理合并节点的多行代码（元组类型）
            if isinstance(line, tuple):
                merged_code = ''
                for line_num in line:
                    if line_num in source:
                        merged_code += source[line_num].strip() + ' '
                if merged_code:
                    seq_code += merged_code
            # 处理单行代码
            elif line != 'exit' and (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 5:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out


class InputFeatures(object):
    """单个训练/测试示例的特征。"""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 path_source,
                 idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.path_source = path_source
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    clean_code, code_dict = remove_comments_and_docstrings(js['func'], 'c')

    # 源代码处理
    code = ' '.join(clean_code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    code_line_count = len(clean_code.splitlines())

    # 使用 VulnerabilityRules 识别潜在的漏洞相关行
    vuln_rules = VulnerabilityRules()
        
    # 准备代码行列表，保留原始行号
    code_lines_with_numbers = []
    code_lines_only = []
        
    # 收集所有非空代码行
    for i in sorted(code_dict.keys()):
        line = code_dict.get(i, '')
        if line.strip():  # 只处理非空行
            code_lines_with_numbers.append((i, line))
            code_lines_only.append(line)
        
    # 使用VulnerabilityRules类的方法识别漏洞行
    vulnerable_indices = vuln_rules.identify_vulnerable_lines(code_lines_only)
        
    # 将索引映射回原始行号
    vuln_related_lines = set()
    for idx in vulnerable_indices:
        if 1 <= idx <= len(code_lines_with_numbers):
            line_num, _ = code_lines_with_numbers[idx-1]
            vuln_related_lines.add(line_num)
        
    # 准备代码和构建AST/CFG
    path_tokens1 = []
    if code_line_count <= 400:
        try:
            g = C_CFG()
            g.idx = js['idx']  # 将idx传递给C_CFG实例
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            _ = g.parse_ast_file(code_ast.root_node)
            _, cfg_allpath, _ = g.get_allpath(vuln_related_lines)
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
        except Exception:
            path_tokens1 = []

    all_seq_ids = []
    for seq in path_tokens1:
        seq_tokens = tokenizer.tokenize(seq)[:args.block_size - 2]
        seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
        seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
        padding_length = args.block_size - len(seq_ids)
        seq_ids += [tokenizer.pad_token_id] * padding_length
        all_seq_ids.append(seq_ids)

    if len(all_seq_ids) < args.filter_size:
        for i in range(args.filter_size - len(all_seq_ids)):
            all_seq_ids.append(source_ids)
    else:
        all_seq_ids = all_seq_ids[:args.filter_size]
    return InputFeatures(source_tokens, source_ids, all_seq_ids, js['idx'], js['target'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []

        # 导入tqdm和c_cfg，用于进度条显示
        import time
        from tqdm import tqdm
        import c_cfg
        # 统计要处理的总代码片段数
        with open(file_path) as count_file:
            total_snippets = sum(1 for _ in count_file)
        c_cfg.total_snippets = total_snippets
        c_cfg.processed_snippets = 0
        c_cfg.total_nodes_across_dataset = 0
        c_cfg.total_critical_nodes_across_dataset = 0
        c_cfg.total_vuln_nodes_across_dataset = 0
        c_cfg.total_centrality_nodes_across_dataset = 0
        c_cfg.total_overlap_nodes_across_dataset = 0
        # 设置进度条，减少更新频率，每10000个代码片段更新一次
        c_cfg.progress_bar = tqdm(total=total_snippets, desc="Processing code snippets", unit="snippet", miniters=10000)
        c_cfg.progress_bar.start_t = time.time()
        
        try:
            with open(file_path) as f:
                for line in f:
                    js = json.loads(line.strip())
                    self.examples.append(convert_examples_to_features(js, tokenizer, args))
        finally:
            # 关闭进度条
            if c_cfg.progress_bar is not None:
                remaining = c_cfg.total_snippets - c_cfg.progress_bar.n
                if remaining > 0:
                    c_cfg.progress_bar.update(remaining)

                c_cfg.progress_bar.set_postfix({
                    "已处理": c_cfg.total_snippets,
                    "总数": c_cfg.total_snippets
                })
                c_cfg.progress_bar.refresh()
                c_cfg.progress_bar.close()
                c_cfg.progress_bar = None

            total_nodes = getattr(c_cfg, 'total_nodes_across_dataset', 0)
            total_critical_nodes = getattr(c_cfg, 'total_critical_nodes_across_dataset', 0)
            total_vuln_nodes = getattr(c_cfg, 'total_vuln_nodes_across_dataset', 0)
            total_centrality_nodes = getattr(c_cfg, 'total_centrality_nodes_across_dataset', 0)
            total_overlap_nodes = getattr(c_cfg, 'total_overlap_nodes_across_dataset', 0)
            if total_nodes:
                print(f"\n总节点数: {total_nodes}")
                print(f"关键节点数(两类并集): {total_critical_nodes}")
                print(f"潜在漏洞节点数(含重叠): {total_vuln_nodes}")
                print(f"高中心性节点数(含重叠): {total_centrality_nodes}")
                print(f"两类重叠节点数: {total_overlap_nodes}")
                print(f"关键节点占比: {total_critical_nodes}/{total_nodes} = {total_critical_nodes / total_nodes:.6f}")
                print(f"潜在漏洞节点占比: {total_vuln_nodes}/{total_nodes} = {total_vuln_nodes / total_nodes:.6f}")
                print(f"高中心性节点占比: {total_centrality_nodes}/{total_nodes} = {total_centrality_nodes / total_nodes:.6f}")
                print(f"两类重叠节点占比: {total_overlap_nodes}/{total_nodes} = {total_overlap_nodes / total_nodes:.6f}")
        """
        # 打印前3个样本的详细信息
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        """

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
            self.examples[i].path_source)


def set_seed(seed=42):
    """
    设置随机种子以确保结果可复现。

    Args:
        seed (int): 随机种子。
    """
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataset, model, tokenizer):
    """ 训练模型 """
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=0, pin_memory=False)
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    args.start_step = 0  # 初始化起始步骤
    model.to(args.device)
    # 准备优化器和学习率调度器（线性预热和衰减）
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("请从 https://www.github.com/nvidia/apex 安装 apex 以使用 fp16 训练。")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 分布式训练（应在 apex fp16 初始化之后）
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    if args.enable_best_model and not args.eval_data_file:
        raise ValueError("启用了最佳模型但未提供验证集文件")
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // args.n_gpu)
    logger.info("  Total train batch size (w. accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    num_train_epochs_int = int(args.num_train_epochs)  # 提前计算，避免在循环中重复调用
    for idx in range(num_train_epochs_int):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            seqs = batch[2].to(args.device)
            model.train()
            loss, logits = model(seq_ids=seqs, input_ids=inputs, labels=labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # 在多 GPU 并行训练上取平均值
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            
            # 清理GPU缓存
            del loss, logits
            if step % 10 == 0:  # 每10步清理一次缓存
                torch.cuda.empty_cache()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

        # 每轮结束：进行一次验证评估（包括最后一轮）
        if args.enable_best_model and args.local_rank in [-1, 0]:
            # 保存当前随机状态，确保评估不影响训练
            torch_rng_state = torch.get_rng_state()
            numpy_rng_state = np.random.get_state()
            python_rng_state = random.getstate()
            if torch.cuda.is_available():
                cuda_rng_state = torch.cuda.get_rng_state()
            
            # 执行评估
            model.eval()
            results = evaluate(args, model, tokenizer, eval_when_training=True)
            for key, value in results.items():
                logger.info("  %s = %s", key, round(value, 4))
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-best')
            model_to_save = model.module if hasattr(model, 'module') else model
            checkpoint_path = os.path.join(checkpoint_dir, 'model.bin')
            curr_acc = results['eval_acc']
            curr_f1 = results['eval_f1']
            curr_score = curr_acc / 1.2 + curr_f1
            best_score = best_acc / 1.2 + best_f1
            if curr_score > best_score:
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save(model_to_save.state_dict(), checkpoint_path)
                best_acc = curr_acc
                best_f1 = curr_f1
                logger.info("  " + "*" * 20)
                logger.info("  Best acc %s", round(best_acc, 4))
                logger.info("  Best f1 %s", round(best_f1, 4))
                logger.info("  Best score %s", round(curr_score, 4))
                logger.info("  " + "*" * 20)
            
            # 恢复训练模式和随机状态
            model.train()
            torch.set_rng_state(torch_rng_state)
            np.random.set_state(numpy_rng_state)
            random.setstate(python_rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
        if args.max_steps > 0 and global_step > args.max_steps:
            break
    
    # 训练结束后保存最终模型
    logger.info("***** Training completed *****")
    logger.info("Saving final model checkpoint...")
    
    # 保存最终模型
    checkpoint_prefix = 'checkpoint-final'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    final_model_path = os.path.join(output_dir, 'model.bin')
    torch.save(model_to_save.state_dict(), final_model_path)
    logger.info("Final model checkpoint saved to %s", final_model_path)
    
    # 同时保存分词器tokenizer和配置config
    tokenizer.save_pretrained(output_dir)
    model_to_save.config.save_pretrained(output_dir)
    logger.info("Tokenizer and config saved to %s", output_dir)
    
    # 每个 epoch 结束后清理 GPU 缓存
    torch.cuda.empty_cache()
    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, eval_when_training=False):
    """
    评估模型。

    Args:
        args: 命令行参数。
        model: 要评估的模型。
        tokenizer: 分词器。
        eval_when_training (bool): 是否在训练期间进行评估。

    Returns:
        dict: 包含评估结果的字典。
    """
    # 在评估开始前清理GPU缓存
    torch.cuda.empty_cache()
    # 循环处理 MNLI 双重评估（匹配、不匹配）
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # 注意 DistributedSampler 是随机采样的
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0,
                                 pin_memory=False)

    # 仅限单 GPU 评估

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        seqs = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(seq_ids=seqs, input_ids=inputs, labels=label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    if logits.ndim == 1:
        probs_pos = logits
    elif logits.shape[1] == 1:
        probs_pos = logits[:, 0]
    else:
        m = np.max(logits, axis=1, keepdims=True)
        e = np.exp(logits - m)
        probs = e / np.sum(e, axis=1, keepdims=True)
        probs_pos = probs[:, 1]
    preds = (probs_pos > args.pred_threshold).astype(np.int64)
    eval_acc = np.mean(labels == preds)
    eval_f1 = f1_score(labels, preds)
    logger.info("  pos_prob_mean = %s", round(float(np.mean(probs_pos)), 4))
    logger.info("  pos_rate = %s", round(float(np.mean(preds)), 4))
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_loss": float(perplexity),
        "eval_acc": float(eval_acc),
        "eval_f1": float(eval_f1)
    }
    
    torch.cuda.empty_cache()
    return result


def test(args, model, tokenizer):
    """
    测试模型。

    Args:
        args: 命令行参数。
        model: 要测试的模型。
        tokenizer: 分词器。
    """
    # 在测试开始前清理GPU缓存
    torch.cuda.empty_cache()
    
    # 循环处理 MNLI 双重评估（匹配、不匹配）
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)

    # 注意 DistributedSampler 是随机采样的
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # 仅限单 GPU 测试

    # 测试阶段
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        seq_inputs = batch[2].to(args.device)
        with torch.no_grad():
            logit = model(seq_inputs, inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > args.pred_threshold
    with open(os.path.join('saved_models', "predictions.txt"), 'w') as f:
        for example, pred in zip(eval_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')
    
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()

    ## 必填参数
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="输入训练数据文件（文本文件）。")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="模型预测和检查点将写入的输出目录。")

    ## 其他参数
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="用于评估困惑度的可选输入评估数据文件（文本文件）。")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="用于评估困惑度的可选输入评估数据文件（文本文件）。")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="要微调的模型架构。")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="用于权重初始化的模型检查点。")

    parser.add_argument("--mlm", action='store_true',
                        help="使用掩码语言建模损失而不是语言建模进行训练。")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="用于掩码语言建模损失的掩码标记比例。")

    parser.add_argument("--config_name", default="", type=str,
                        help="可选的预训练配置名称或路径，如果与model_name_or_path不同。")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="可选的预训练分词器名称或路径，如果与model_name_or_path不同。")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="可选目录，用于存储从s3下载的预训练模型（而不是默认目录）。")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="分词后的可选输入序列长度。"
                             "训练数据集将以该大小的块进行截断以进行训练。"
                             "对于单句输入，默认为模型最大输入长度（考虑特殊标记）。")
    parser.add_argument("--do_train", action='store_true',
                        help="是否训练。")
    parser.add_argument("--do_eval", action='store_true',
                        help="是否在开发集上运行评估。")
    parser.add_argument("--do_test", action='store_true',
                        help="是否在测试集上运行评估。")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="在日志记录步骤期间运行训练评估。")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="如果使用不区分大小写的模型，请设置此标志。")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="每个GPU/CPU的训练批次大小。")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="每个GPU/CPU的评估批次大小。")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="在执行反向/更新传递之前累积的更新步数。")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="Adam优化器的初始学习率。")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="如果应用权重衰减，则为权重衰减。")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Adam优化器的epsilon值。")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="最大梯度范数。")
    parser.add_argument("--epoch", default=42, type=float,
                        help="要执行的训练总轮数。")
    parser.add_argument('--seed', type=int, default=42,
                        help="用于初始化的随机种子。")
    parser.add_argument('--fp16', action='store_true',
                        help="是否使用16位（混合）精度（通过NVIDIA apex）而不是32位。")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="对于fp16：在['O0', 'O1', 'O2', 'O3']中选择的Apex AMP优化级别。"
                             "详情请参阅https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="用于分布式训练：local_rank。")
    parser.add_argument("--cnn_size", default=128, type=int,
                        help="CNN隐藏层大小。")
    parser.add_argument("--d_size", default=128, type=int,
                        help="路径融合的隐藏层大小。")
    parser.add_argument("--pkl_file", type=str, default="",
                        help="关键路径的缓存文件。")
    parser.add_argument('--start_step', type=int, default=0, help='训练的起始步数。')
    parser.add_argument('--start_epoch', type=int, default=0, help='训练开始的 epoch 数。')
    parser.add_argument('--no_cuda', action='store_true',
                        help="是否不使用CUDA。")
    parser.add_argument("--gpu_id", type=int, default=0, help="要使用的GPU ID。")

    parser.add_argument("--pred_threshold", default=0.5, type=float,
                        help="分类阈值（用于评估/测试阶段的二分类决策）。")

    parser.add_argument("--filter_size", default=3, type=int,
                        help="输入预训练模型的路径数量。")

    parser.add_argument("--enable_best_model", action='store_true',
                        help="是否启用最佳模型：启用则训练期间验证并保存最佳与最终模型；关闭则不验证，仅保存最终模型。")
    parser.add_argument("--checkpoint_type", default="final", type=str, choices=["best", "final"],
                        help="用于推理/评估阶段的检查点类型：'best' 或 'final'。")



    args = parser.parse_args()


    # 设置CUDA、GPU和分布式训练
    if args.gpu_id >= 0 and torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        args.n_gpu = 1  # 因为我们指定了单个GPU
    elif args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() and not args.no_cuda else 0
    else:  # 初始化分布式后端，它将负责同步节点/GPU
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    # 设置GPU内存管理策略
    if torch.cuda.is_available() and not args.no_cuda:
        # 设置内存分片大小以避免碎片化
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        # 清理GPU缓存
        torch.cuda.empty_cache()
        # 设置内存增长策略
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # 设置日志
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("进程排名: %s, 设备: %s, GPU数量: %s, 分布式训练: %s, 16位训练: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # 设置随机种子
    set_seed(args.seed)

    # 加载预训练模型和分词器
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    
    if args.model_name_or_path:
        encoder = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        encoder = model_class(config)

    model = Model(encoder, config, tokenizer, args)
    
    # 在模型创建后清理GPU缓存
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.empty_cache()

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        # 在创建数据集前清理内存
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.empty_cache()
        
        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)

        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # 评估
    results = {}
    if args.do_eval:
        # 在评估前清理GPU缓存
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.empty_cache()
        if args.checkpoint_type == "best":
            checkpoint_prefix = 'checkpoint-best/model.bin'
        else:
            checkpoint_prefix = 'checkpoint-final/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

    if args.do_test and args.local_rank in [-1, 0]:
        # 在测试前清理GPU缓存
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.empty_cache()
        
        if args.checkpoint_type == "best":
            checkpoint_prefix = 'checkpoint-best/model.bin'
        else:
            checkpoint_prefix = 'checkpoint-final/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        
        if torch.cuda.is_available() and not args.no_cuda:
            torch.cuda.empty_cache()
        
        test(args, model, tokenizer)


if __name__ == "__main__":
    main()
