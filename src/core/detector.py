"""
bait.py: Core module for the BAIT (LLM Backdoor Scanning) project.

Author: [NoahShen]
Organization: [PurduePAML]
Date: [2024-10-01]
Version: 1.1

This module contains the main BAIT class It provides
the core functionality for initializing and running backdoor scans on LLMs.

Copyright (c) [2024] [PurduePAML]
"""
import torch
import os
import json
import traceback
from time import time, sleep
from typing import Optional, List, Tuple, Dict
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.config.arguments import BAITArguments
from openai import OpenAI
from src.utils.constants import JUDGE_SYSTEM_PROMPT
from src.config.arguments import ModelArguments, DataArguments, ScanArguments
from src.utils.helpers import extract_tag
from openai import APIError, RateLimitError, APIConnectionError
from dataclasses import dataclass
from loguru import logger
from src.models.model import build_model, parse_model_args
from src.data.dataset import build_data_module
import sys


@dataclass
class BestTarget:
    q_score: float = 0
    invert_target: str = None
    reasoning: str = ""
    
    def __str__(self) -> str:
        return (f"BestTarget:\n"
                f"  q_score: {self.q_score}\n"
                f"  invert_target: {self.invert_target!r}\n"
                f"  reasoning: {self.reasoning!r}")

@dataclass
class ScanResult:
    is_backdoor: bool
    best_target: BestTarget


class BAIT:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataloader: torch.utils.data.DataLoader,
        bait_args: BAITArguments,
        logger: Optional[object] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize the BAIT object.

        Args:
            model (PreTrainedModel): The pre-trained language model.
            tokenizer (PreTrainedTokenizer): The tokenizer for the model.
            dataloader (DataLoader): DataLoader for input data.
            bait_args (BAITArguments): Configuration arguments for BAIT.
            logger (Optional[object]): Logger object for logging information.
            device (str): Device to run the model on (cuda or cpu).
        """
        logger.info("Initializing BAIT...")
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.logger = logger
        self.device = device
        self._init_config(bait_args)
        # Optional: use any OpenAI-compatible endpoint (e.g. a local vLLM/llama.cpp server) for the judge.
        local_judge_url = os.getenv("BAIT_JUDGE_BASE_URL")
        if local_judge_url:
            key_file = os.getenv("BAIT_JUDGE_API_KEY_FILE")
            judge_api_key = open(key_file).read().strip() if key_file else "not-needed"
            self.judge_client = OpenAI(api_key=judge_api_key, base_url=local_judge_url)
            self.judge_model_name = os.getenv("BAIT_JUDGE_MODEL_NAME", self.judge_model_name)
        else:
            self.judge_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        # Optional: replace the judge prompt with the contents of a file. It must contain the placeholder {response}
        # and ask for the same answer format (<State>suspicious|safe</State><Reasoning>...</Reasoning>).
        self.judge_prompt = JUDGE_SYSTEM_PROMPT
        prompt_file = os.getenv("BAIT_JUDGE_PROMPT_FILE")
        if prompt_file:
            self.judge_prompt = open(prompt_file).read()
            if "{response}" not in self.judge_prompt:
                raise ValueError("BAIT_JUDGE_PROMPT_FILE must contain the placeholder {response}")
        # Optional (BAIT_SAVE_POOL=1): keep EVERY inverted candidate with its per-step statistics, not only the best one.
        # One record per candidate, written as candidates.jsonl in the run folder. Does not change any score.
        self.save_pool = os.getenv("BAIT_SAVE_POOL") == "1"
        self.pool = []
        self._cur_batch = -1

    def _step_stats(self, avg, tok, rows):
        """Per-step features of one decoding step, for the candidate pool (cheap, CPU). avg: (n, V) mean next-token
        distribution over the prompt rows; tok: (n,) chosen token; rows: (n, P, V) per-row distributions."""
        a = avg.float()
        ent = -(a * a.clamp_min(1e-12).log()).sum(-1)
        tp, ti = torch.topk(a, 5, dim=-1)
        chosen = a.gather(1, tok.view(-1, 1)).squeeze(1)
        rank = (a > chosen.unsqueeze(1)).sum(-1)
        rowp = rows.float().gather(2, tok.view(-1, 1, 1).expand(-1, rows.shape[1], 1)).squeeze(2)
        return dict(entropy=ent.cpu(), top5_p=tp.cpu(), top5_id=ti.cpu(), rank=rank.cpu(), row_p=rowp.cpu())

    def _pool_add(self, **rec):
        self.pool.append(rec)


    @torch.no_grad()
    def run(self) -> ScanResult:
        """
        Run the BAIT algorithm on the input data.

        Returns:
            ScanResult: A ScanResult object containing:
                - Boolean indicating if a backdoor was detected
                - The highest Q-score found
                - The invert target (token IDs) for the potential backdoor
        """

        best_target = BestTarget()

        for batch_idx, batch_inputs in enumerate(tqdm(self.dataloader, desc="Scanning data...")):
            self._cur_batch = batch_idx
            input_ids = batch_inputs["input_ids"]
            attention_mask = batch_inputs["attention_mask"]
            index_map = batch_inputs["index_map"]

            batch_q_score, batch_invert_target = self.scan_init_token(input_ids, attention_mask, index_map)
            self.logger.debug(f"Batch Q-score: {batch_q_score}, Batch Invert Target: {batch_invert_target}")

            if batch_q_score > best_target.q_score:
                # post-process to further exam if the invert target includes suspicious content which might be a backdoor target string
                batch_is_suspicious, batch_reasoning = self.__post_process(batch_invert_target)
                if batch_is_suspicious:
                    # update best target
                    best_target.q_score = batch_q_score
                    best_target.invert_target = batch_invert_target
                    best_target.reasoning = batch_reasoning
                    self.logger.info(f"New best target found: {best_target}")

            # early stop if a very promising target is found
            if self.early_stop and best_target.q_score > self.early_stop_q_score_threshold:
                self.logger.info(f"Early stop at q-score: {best_target.q_score}")
                break

        if best_target.q_score > self.q_score_threshold:
            self.logger.info(f"Q-score is greater than threshold: {self.q_score_threshold}")
            self.logger.info(f"Inverted Target contains suspicious content: {best_target.invert_target}")
            self.logger.info(f"Reasoning: {best_target.reasoning}")
            is_backdoor = True
        else:
            self.logger.info(f"Q-score is less than threshold: {self.q_score_threshold}")
            is_backdoor = False
        
        return ScanResult(is_backdoor, best_target)


    def __post_process(
        self,
        invert_target: str,
    ) -> str:
        """
        Post-process to further exam if the invert target includes suspicious content which might be a backdoor target string

        Args:
            invert_target (str): The target string to analyze
        """

        for attempt in range(self.max_retries):
            try:
                response = self.judge_client.chat.completions.create(
                    model=self.judge_model_name,
                    messages=[
                        {"role": "user", "content": self.judge_prompt.replace("{response}", invert_target)}
                    ],
                    # BAIT_JUDGE_EXTRA_BODY: JSON merged into the request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}
                    extra_body=json.loads(os.getenv("BAIT_JUDGE_EXTRA_BODY", "{}")) if os.getenv("BAIT_JUDGE_BASE_URL") else {},
                ).choices[0].message.content

                try:
                    state = extract_tag(response, "State").lower().strip()
                    reasoning = extract_tag(response, "Reasoning")

                    if not state or not reasoning:
                        self.logger.error("Missing required tags in response")
                        continue

                    if state not in ["suspicious", "safe"]:
                        self.logger.error(f"Invalid state value: {state}")
                        continue

                    if state == "suspicious":
                        return True, reasoning
                    else:
                        return False, reasoning

                except (ValueError, AttributeError, IndexError) as e:
                    self.logger.error(f"Failed to parse response: {str(e)}")
                    if attempt == self.max_retries - 1:
                        return False, "Error: Failed to parse response after multiple attempts"
                    continue

            except (APIError, RateLimitError, APIConnectionError) as e:
                if attempt == self.max_retries - 1:  # Last attempt
                    self.logger.error(f"Failed to get response after {self.max_retries} attempts: {str(e)}")
                    return False, "Error: Failed to analyze content after multiple attempts"

                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {self.retry_delay} seconds...")
                sleep(self.retry_delay)
                self.retry_delay *= 2  # Exponential backoff

    def stable_softmax(self, logits, dim=-1, temperature=1.0):
        """Numerically stable softmax implementation"""
        # Subtract max for numerical stability
        logits = logits / temperature
        max_logits = torch.max(logits, dim=dim, keepdim=True)[0]
        exp_logits = torch.exp(logits - max_logits)
        sum_exp = torch.sum(exp_logits, dim=dim, keepdim=True)
        
        # Add epsilon to prevent division by zero
        eps = 1e-12
        return exp_logits / (sum_exp + eps)

    def __generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = 1,
        past_key_values=None
    ) -> torch.Tensor:
        """
        Generate output probabilities for the next token using the model.

        PERF: accepts and returns `past_key_values` so a caller stepping
        token-by-token (full_inversion, warm_up_inversion) can carry the KV cache across calls instead of
        re-prefilling the whole growing sequence from scratch every step -- the original code called this
        with the FULL `input_ids` and no cache on every step, which is O(n^2) in sequence length (full_steps
        re-prefills of length prompt_len..prompt_len+20). HF's generate() recognizes the overlap between
        `input_ids` and a passed-in `past_key_values`'s seen length and only forwards the new suffix, so
        passing the full growing `input_ids` unchanged alongside the cache is sufficient -- no other call-site
        change needed. Verified numerically identical to the no-cache path (max abs diff ~2e-5, fp noise) via
        spikes/bait_cache_fix_check.py, given this run's config has do_sample=False, top_p=1.0, temperature=1.0
        (no stochastic/filtering logits processors whose state would differ under caching). `no_repeat_ngram_size`
        is a pure function of the current token sequence (not of caching), so it is unaffected either way.

        Returns:
            (output_probs, past_key_values): output_probs as before; past_key_values to pass into the NEXT
            call in the same stepping loop (None on the first call of a fresh sequence).
        """
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            top_p=self.top_p,
            temperature=self.temperature,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            do_sample=self.do_sample,
            return_dict_in_generate=self.return_dict_in_generate,
            output_scores=self.output_scores,
            **({"past_key_values": past_key_values, "use_cache": True} if past_key_values is not None else {})
        )

        output_scores = outputs.scores[0]

        # Handle NaN and inf values in output scores
        output_scores = torch.nan_to_num(output_scores, nan=0.0, posinf=1e6, neginf=-1e6)

        # print(f"output_scores: {output_scores}")
        # print(f"before softmax: {output_scores.max()}, {output_scores.min()}")

        # Check for any remaining problematic values
        if torch.isnan(output_scores).any() or torch.isinf(output_scores).any():
            self.logger.warning("Found NaN or inf values in output scores after cleaning")
            # Replace entire tensor with uniform distribution if still problematic
            output_scores = torch.zeros_like(output_scores)

        output_probs = self.stable_softmax(output_scores, dim=-1)
        # print(f"after softmax: {output_probs.max()}, {output_probs.min()}")

        return output_probs, getattr(outputs, "past_key_values", None)


    def warm_up_inversion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform warm-up inversion to using a mini-batch and short generation steps

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Processed targets and target probabilities.
        """
        if os.getenv("BAIT_FAST_WARMUP") == "1" and not getattr(self, "_fast_warmup_broken", False):  # perf flag, default off
            try:
                return self._warm_up_fast(input_ids, attention_mask)
            except Exception as e:  # any failure: original path, and stop trying the fast one in this process
                self.logger.warning(f"fast warm-up failed ({type(e).__name__}: {str(e)[:200]}); using the original warm-up")
                if not isinstance(e, torch.OutOfMemoryError):
                    self._fast_warmup_broken = True
                torch.cuda.empty_cache()
        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.warmup_batch_size))
        targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1
        target_mapping_record = [torch.arange(batch_size).to(self.device)]
        uncertainty_inspection_times = torch.zeros(batch_size).to(self.device)

        processed_targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        processed_target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1

        for step in range(self.warmup_steps):
            # PERF note: NOT cache-optimized -- this loop's batch rows can shrink mid-loop (_update /
            # uncertainty_inspection drop candidates via target_mapping_record), which would need the KV
            # cache re-indexed the same way to stay correct. Only 5 steps and called once per outer batch
            # (vs. full_inversion's up to 100 candidates x 20 steps each), so left as the safer, smaller cost.
            output_probs, _ = self.__generate(input_ids, attention_mask)
            input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times = self._update(
                targets,
                target_probs,
                output_probs,
                input_ids,
                attention_mask,
                step,
                target_mapping_record,
                uncertainty_inspection_times
            )

            if input_ids is None:
                self.logger.debug("Input ids is empty, break")
                return processed_targets, processed_target_probs


        last_step_indices = target_mapping_record[-1]
        original_indices = []
        for idx in range(len(last_step_indices)):
            # trace back to the first step
            original_idx = last_step_indices[idx]
            for step in range(len(target_mapping_record)-2, -1, -1):
                original_idx = target_mapping_record[step][original_idx]
            original_indices.append(original_idx)

        original_indices = torch.tensor(original_indices)
        processed_targets[:,original_indices] = targets
        processed_target_probs[:,original_indices] = target_probs
        return processed_targets, processed_target_probs

    def _inspect_batched(self, past, input_ids, attention_mask, avg, uncertain, chunk_size=16):
        """Uncertainty inspection for several candidates at once, reusing the KV cache `past` (rows = candidate*W + w).
        Same math as uncertainty_inspection(): per candidate, the top-k next tokens are appended to its W rows, one
        more step is run, and the top-k token whose follow-up has the highest max prob wins."""
        import copy
        W, K = self.warmup_batch_size, self.uncertainty_inspection_topk
        result = {}
        for c0 in range(0, len(uncertain), chunk_size):
            chunk = uncertain[c0:c0 + chunk_size]
            topk_probs, topk_idx = torch.topk(avg[chunk], k=K, dim=-1)            # (u,K)
            rows = torch.cat([torch.arange(c * W, (c + 1) * W, device=input_ids.device).repeat(K) for c in chunk])
            tok = topk_idx.reshape(-1).repeat_interleave(W)                          # (u*K*W,)
            ids = torch.cat([input_ids[rows], tok.unsqueeze(1)], dim=-1)
            mask = torch.cat([attention_mask[rows], attention_mask[rows][:, -1:]], dim=-1)
            cache = copy.deepcopy(past)
            cache.batch_select_indices(rows)
            probs, _ = self.__generate(ids, mask, past_key_values=cache)
            del cache
            probs = probs.view(len(chunk), K, W, -1).mean(dim=2)                    # (u,K,V)
            best = probs.max(dim=-1).values.argmax(dim=-1)                          # (u,)
            for j, c in enumerate(chunk):
                result[c] = topk_idx[j, best[j]]
        return result

    def _warm_up_fast(self, input_ids, attention_mask):
        """PERF (opt-in, BAIT_FAST_WARMUP=1): warm_up_inversion + _update with (1) a KV cache carried across
        the warm-up steps and re-indexed to the surviving candidates' rows, and (2) all of a step's uncertainty
        inspections run as one batched call on a copy of that cache. Per-candidate decisions, including upstream's
        quirk of indexing uncertainty_inspection_times by current batch position, are kept as in the original."""
        W = self.warmup_batch_size
        eos = self.tokenizer.eos_token_id
        batch_size = min(self.batch_size, int(input_ids.shape[0] // W))
        targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1
        mapping = [torch.arange(batch_size).to(self.device)]
        unc_times = torch.zeros(batch_size).to(self.device)
        processed_targets = torch.zeros(self.warmup_steps, batch_size).long().to(self.device) - 1
        processed_target_probs = torch.zeros(self.warmup_steps, batch_size).to(self.device) - 1
        past = None
        for step in range(self.warmup_steps):
            out_probs, past = self.__generate(input_ids, attention_mask, past_key_values=past)
            nb = mapping[-1].shape[0]
            avg = out_probs.view(nb, W, -1).mean(dim=1)
            ent = self._compute_self_entropy(avg)
            maxp, argm = avg.max(dim=-1)
            ent_c, maxp_c, unc_c = ent.cpu(), maxp.cpu(), unc_times.cpu()
            cond = [bool(self._check_uncertainty(ent_c[c], None, maxp_c[c], unc_c[c])) for c in range(nb)]
            insp = self._inspect_batched(past, input_ids, attention_mask, avg, [c for c in range(nb) if cond[c]]) if any(cond) else {}
            argm_c = argm.cpu()
            keep, new_tokens = [], []
            for c in range(nb):
                if cond[c]:
                    new_token = insp[c]
                    if new_token == eos or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                        continue
                    unc_times[c] += 1
                    targets[step][c] = new_token
                    target_probs[step][c] = avg[c][new_token]
                elif ent_c[c] < self.self_entropy_lower_bound or maxp_c[c] > self.expectation_threshold:
                    new_token = argm[c]
                    if new_token == eos or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                        continue
                    targets[step][c] = new_token
                    target_probs[step][c] = maxp[c]
                else:
                    continue
                keep.append(c)
                new_tokens.append(new_token.reshape(()))
            if not keep:
                self.logger.debug("Input ids is empty, break")
                return processed_targets, processed_target_probs
            rows = torch.cat([torch.arange(c * W, (c + 1) * W, device=input_ids.device) for c in keep])
            toks = torch.stack(new_tokens).to(input_ids.device).repeat_interleave(W).unsqueeze(1)
            input_ids = torch.cat([input_ids[rows], toks], dim=-1)
            attention_mask = torch.cat([attention_mask[rows], attention_mask[rows][:, -1:]], dim=-1)
            past.batch_select_indices(rows)
            sel = torch.tensor(keep).long().to(self.device)
            targets, target_probs = targets[:, sel], target_probs[:, sel]
            mapping.append(sel)
        last = mapping[-1]
        original = []
        for idx in range(len(last)):
            o = last[idx]
            for st in range(len(mapping) - 2, -1, -1):
                o = mapping[st][o]
            original.append(o)
        original = torch.tensor(original)
        processed_targets[:, original] = targets
        processed_target_probs[:, original] = target_probs
        return processed_targets, processed_target_probs

    def full_inversion(
        self,
        warmup_targets: torch.Tensor,
        warmup_target_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        Perform full inversion to find the highest Q-score and invert target.

        Args:
            warmup_targets (torch.Tensor): Targets from warm-up inversion.
            warmup_target_probs (torch.Tensor): Target probabilities from warm-up inversion.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Highest Q-score and corresponding invert target.
        """
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        q_score = 0
        invert_target = None


        batch_size = min(self.batch_size, int(input_ids.shape[0] // self.prompt_size))

        cand_k = int(os.getenv("BAIT_CAND_BATCH", "0"))  # 0/1 = original per-candidate loop
        if cand_k > 1:
            try:
                return self._full_inversion_batched(warmup_targets, input_ids, attention_mask, batch_size, cand_k)
            except torch.OutOfMemoryError:
                self.logger.warning(f"candidate batching K={cand_k} ran out of memory; falling back to the per-candidate loop for this batch")
                torch.cuda.empty_cache()

        for i in range(batch_size):
            if -1 in warmup_targets[:,i]:
                continue

            warmup_target = warmup_targets[:,i]
            warmup_target_prob = warmup_target_probs[:,i]
            batch_input_ids = input_ids[i*self.prompt_size:(i+1)*self.prompt_size]
            batch_attention_mask = attention_mask[i*self.prompt_size:(i+1)*self.prompt_size]

            initial_token = batch_input_ids[0, -1].unsqueeze(0)

            batch_target = []
            batch_target_prob = []
            # PERF : this candidate's `self.prompt_size` rows stay fixed for
            # the whole inner loop (no row-dropping like warm_up_inversion), so a KV cache carried across
            # these `full_steps` (default 20) calls is safe -- see __generate's docstring for the validation.
            # Was: full re-prefill of the growing sequence every step, O(full_steps^2) in sequence length.
            past = None
            steps_stats = []

            for step in range(self.full_steps):
                output_probs, past = self.__generate(batch_input_ids, batch_attention_mask, past_key_values=past)
                avg_probs = output_probs.mean(dim=0)
                if step < self.warmup_steps:
                    new_token = warmup_target[step].unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target.append(warmup_target[step])
                    batch_target_prob.append(avg_probs[warmup_target[step]])
                else:
                    top_prob, top_token = torch.max(avg_probs, dim=-1)
                    new_token = top_token.unsqueeze(0).expand(self.prompt_size, -1)
                    batch_target.append(top_token)
                    batch_target_prob.append(top_prob)
                if self.save_pool:
                    steps_stats.append(self._step_stats(avg_probs.unsqueeze(0), batch_target[-1].view(1), output_probs.unsqueeze(0)))

                batch_input_ids = torch.cat([batch_input_ids, new_token], dim=-1)
                batch_attention_mask = torch.cat([batch_attention_mask, batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)



                if batch_target[step].item() == self.tokenizer.eos_token_id or self.tokenizer.decode(batch_target[step].item()) == "<|end_of_text|>":
                    self.logger.debug(f"EOS token reached at step {step}")
                    break

            batch_target = torch.tensor(batch_target).long()
            batch_target_prob = torch.tensor(batch_target_prob)
            raw_tok, raw_p = batch_target.clone(), batch_target_prob.clone()


            if self.tokenizer.eos_token_id in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.eos_token_id)[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]
            
            if self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0] in batch_target:
                eos_id = torch.where(batch_target == self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0])[0][0].item()
                batch_target = batch_target[:eos_id]
                batch_target_prob = batch_target_prob[:eos_id]


            # Remove the smallest probability from batch_target_prob to improve detection robustness
            if len(batch_target_prob) > 1:
                min_prob_index = torch.argmin(batch_target_prob)
                batch_target_prob = torch.cat([batch_target_prob[:min_prob_index], batch_target_prob[min_prob_index+1:]])

            # Calculate the batch_q_score as the mean of the remaining probabilities
            batch_q_score = batch_target_prob.mean().item()
            batch_target = torch.cat([initial_token.detach().cpu(), batch_target], dim=-1)
            batch_invert_target = self.tokenizer.decode(batch_target)
            self.logger.debug(f"batch_invert_target: {batch_invert_target}")
            self.logger.debug(f"batch_q_score: {batch_q_score}")
            if self.save_pool:
                self._record_candidate(i, initial_token, raw_tok, raw_p, steps_stats, batch_target, batch_q_score, batch_invert_target,
                                       input_ids[i*self.prompt_size:(i+1)*self.prompt_size], attention_mask[i*self.prompt_size:(i+1)*self.prompt_size])
            if batch_q_score > q_score and len(batch_invert_target.split()) >= self.min_target_len:
                q_score = batch_q_score
                invert_target = batch_invert_target

        return q_score, invert_target

    def _full_inversion_batched(self, warmup_targets, input_ids, attention_mask, batch_size, cand_k):
        """PERF (opt-in via BAIT_CAND_BATCH=<K>): same algorithm as the loop in full_inversion,
        but K candidates share one decode step (K*prompt_size rows) so the weights are read once per step instead of
        once per candidate. Per-candidate results are post-processed exactly as in the original loop. Differences:
        the `decode(token) == "<|end_of_text|>"` early-break is not replicated (never a single token in Qwen's vocab);
        rows may differ from the unbatched run by floating-point noise."""
        P = self.prompt_size
        eos = self.tokenizer.eos_token_id
        eot = self.tokenizer.encode("<|end_of_text|>", add_special_tokens=False)[0]
        valid = [i for i in range(batch_size) if -1 not in warmup_targets[:, i]]
        q_score = 0
        invert_target = None
        for c0 in range(0, len(valid), cand_k):
            chunk = valid[c0:c0 + cand_k]
            n = len(chunk)
            ids = torch.cat([input_ids[i * P:(i + 1) * P] for i in chunk], dim=0)
            mask = torch.cat([attention_mask[i * P:(i + 1) * P] for i in chunk], dim=0)
            initial = [input_ids[i * P:(i + 1) * P][0, -1].unsqueeze(0) for i in chunk]
            warm = warmup_targets[:, chunk]  # (warmup_steps, n)
            toks, probs = [], []
            done = torch.zeros(n, dtype=torch.bool, device=ids.device)
            past = None
            steps_stats = []
            for step in range(self.full_steps):
                out_probs, past = self.__generate(ids, mask, past_key_values=past)
                avg = out_probs.view(n, P, -1).mean(dim=1)
                if step < self.warmup_steps:
                    tok = warm[step].to(avg.device)
                    p = avg.gather(1, tok.unsqueeze(1)).squeeze(1)
                else:
                    p, tok = torch.max(avg, dim=-1)
                toks.append(tok)
                probs.append(p)
                if self.save_pool:
                    steps_stats.append(self._step_stats(avg, tok, out_probs.view(n, P, -1)))
                ids = torch.cat([ids, tok.repeat_interleave(P).unsqueeze(1)], dim=-1)
                mask = torch.cat([mask, mask[:, -1:]], dim=-1)
                done |= (tok == eos)
                if bool(done.all()):
                    break
            T = torch.stack(toks).cpu()      # (steps, n)
            Pm = torch.stack(probs).cpu()
            for j in range(n):
                col = T[:, j]
                hit = torch.where(col == eos)[0]
                end = int(hit[0]) + 1 if len(hit) else col.shape[0]  # original breaks right after appending the EOS step
                batch_target = col[:end].long()
                batch_target_prob = Pm[:end, j]
                raw_tok, raw_p = batch_target.clone(), batch_target_prob.clone()
                if eos in batch_target:
                    k = torch.where(batch_target == eos)[0][0].item()
                    batch_target, batch_target_prob = batch_target[:k], batch_target_prob[:k]
                if eot in batch_target:
                    k = torch.where(batch_target == eot)[0][0].item()
                    batch_target, batch_target_prob = batch_target[:k], batch_target_prob[:k]
                if len(batch_target_prob) > 1:
                    m = torch.argmin(batch_target_prob)
                    batch_target_prob = torch.cat([batch_target_prob[:m], batch_target_prob[m + 1:]])
                bq = batch_target_prob.mean().item()
                batch_target = torch.cat([initial[j].detach().cpu(), batch_target], dim=-1)
                inv = self.tokenizer.decode(batch_target)
                self.logger.debug(f"batch_invert_target: {inv}")
                self.logger.debug(f"batch_q_score: {bq}")
                if self.save_pool:
                    ci = chunk[j]
                    self._record_candidate(ci, initial[j], raw_tok, raw_p, [{k: v[j:j + 1] for k, v in s.items()} for s in steps_stats], batch_target, bq, inv,
                                           input_ids[ci * P:(ci + 1) * P], attention_mask[ci * P:(ci + 1) * P])
                if bq > q_score and len(inv.split()) >= self.min_target_len:
                    q_score = bq
                    invert_target = inv
        return q_score, invert_target

    def _record_candidate(self, cand_idx, initial_token, raw_tok, raw_p, steps_stats, final_target_ids, q, text, prompt_rows, prompt_mask):
        """One pool record per inverted candidate (BAIT_SAVE_POOL=1). Stores everything cheap to keep: the raw decoded tokens
        with their per-step mean probability (before trimming at EOS and before the weakest token is dropped), per-step entropy,
        top-5 alternatives, rank of the chosen token (non-zero while the token was forced by warm-up), the per-prompt-row
        probability of the chosen token, the token ids of the final trimmed target, q, and the prompt rows the scan used."""
        L = len(raw_tok)
        cat = lambda k: torch.cat([s[k] for s in steps_stats[:L]], dim=0) if steps_stats else None
        ent, t5p, t5i, rank, rowp = cat("entropy"), cat("top5_p"), cat("top5_id"), cat("rank"), cat("row_p")
        init = int(initial_token.reshape(-1)[0])
        rows = [prompt_rows[r][prompt_mask[r].bool()].tolist() for r in range(prompt_rows.shape[0])]
        tok_ids = [int(x) for x in raw_tok]
        self._pool_add(
            batch_idx=self._cur_batch, cand_idx=int(cand_idx), init_token=init, init_text=self.tokenizer.decode([init]),
            q=float(q), text=text, long_enough=len(text.split()) >= self.min_target_len,
            tokens=tok_ids, token_text=[self.tokenizer.decode([x]) for x in tok_ids], p=[float(x) for x in raw_p],
            forced=[s < self.warmup_steps for s in range(L)],
            entropy=ent.tolist() if ent is not None else None, top5_id=t5i.tolist() if t5i is not None else None,
            top5_p=t5p.tolist() if t5p is not None else None, rank=rank.tolist() if rank is not None else None,
            row_p=rowp.tolist() if rowp is not None else None,
            final_ids=[int(x) for x in final_target_ids], prompt_rows=rows)

    def scan_init_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        index_map: List[int]
    ) -> Tuple[float, torch.Tensor]:
        """
        enumerate initial tokens and invert the entire attack target.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            index_map (List[int]): Mapping of indices for batches.

        Returns:
            Tuple[float, torch.Tensor]: Q-score and invert target for potential backdoor.
        """
        sample_index = []
        for map_idx in index_map:
            start_idx = index_map[map_idx]
            end_idx = index_map[map_idx] +  self.warmup_batch_size
            sample_index.extend(i for i in range(start_idx, end_idx))


        sample_input_ids = input_ids[sample_index].to(self.device)
        sample_attention_mask = attention_mask[sample_index].to(self.device)
        warmup_targets, warmup_target_probs = self.warm_up_inversion(sample_input_ids, sample_attention_mask)
        return self.full_inversion(warmup_targets, warmup_target_probs, input_ids, attention_mask, index_map)


    def uncertainty_inspection(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        avg_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform uncertainty inspection for the current batch.
        """
        topk_probs, topk_indices = torch.topk(avg_probs, k=self.uncertainty_inspection_topk, dim=-1)
        #============================Debugging log============================
        for topk_prob, topk_index in zip(topk_probs, topk_indices):
            token = self.tokenizer.convert_ids_to_tokens(topk_index.tolist())
            self.logger.debug(f"Tokens: {token:<20} | IDs: {topk_index.item():<20} | Probs: {topk_prob.item():<20.4f}")
        #============================Debugging log============================
        reshape_topk_indices = topk_indices.view(-1).repeat_interleave(self.warmup_batch_size).unsqueeze(1)
        input_ids = input_ids.repeat(self.uncertainty_inspection_topk, 1)
        attention_mask = attention_mask.repeat(self.uncertainty_inspection_topk, 1)
        input_ids = torch.cat([input_ids, reshape_topk_indices], dim=-1)
        attention_mask = torch.cat([attention_mask, attention_mask[:, -1].unsqueeze(1)], dim=-1)
        output_probs, _ = self.__generate(input_ids, attention_mask)
        output_probs = output_probs.view(self.uncertainty_inspection_topk, self.warmup_batch_size, -1).mean(dim=1)
        max_prob, max_indices = torch.max(output_probs, dim=-1)
        new_token = topk_indices[max_prob.argmax()]

        #============================Debugging log============================
        self.logger.debug(f"Max prob: {max_prob}")
        self.logger.debug(f"Max indices: {max_indices}")
        self.logger.debug(f"max_indices.argmax(): {max_prob.argmax()}")
        self.logger.debug(f"decode: {self.tokenizer.decode(max_prob.argmax())}")
        self.logger.debug(f"new_token: {new_token}")
        self.logger.debug(f"decode: {self.tokenizer.decode(new_token)}")
        #============================Debugging log============================
        return new_token



    def _init_config(self, bait_args: BAITArguments) -> None:
        """
        Initialize configuration from BAITArguments.

        Args:
            bait_args (BAITArguments): Configuration arguments for BAIT.
        """
        for key, value in bait_args.__dict__.items():
            setattr(self, key, value)


    def _update(
        self,
        targets: torch.Tensor,
        target_probs: torch.Tensor,
        output_probs: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step: int,
        target_mapping_record: List[torch.Tensor],
        uncertainty_inspection_times: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Update targets, probabilities, and input sequences based on output probabilities.

        Args:
            targets (torch.Tensor): Current target tokens.
            target_probs (torch.Tensor): Current target probabilities.
            output_probs (torch.Tensor): Output probabilities from the model.
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
            step (int): Current step in the inversion process.
            target_mapping_record (List[torch.Tensor]): Record of target mappings.
            tolerance_times (torch.Tensor): Record of tolerance times for each sequence.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                Updated input_ids, attention_mask, targets, target_probs, and target_mapping_record.
        """
        # Calculate average probabilities across the warmup batch
        batch_size = target_mapping_record[-1].shape[0]
        avg_probs = output_probs.view(batch_size, self.warmup_batch_size, -1).mean(dim=1)

        self_entropy = self._compute_self_entropy(avg_probs)


        selected_indices = []
        selected_input_ids = []
        selected_attention_mask = []


        for cand_idx in range(batch_size):
            cand_self_entropy = self_entropy[cand_idx]
            cand_avg_probs = avg_probs[cand_idx]
            cand_max_prob = cand_avg_probs.max()
            cand_batch_input_ids = input_ids[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]
            cand_batch_attention_mask = attention_mask[cand_idx * self.warmup_batch_size:(cand_idx + 1) * self.warmup_batch_size]

            cand_uncertainty_inspection_times = uncertainty_inspection_times[cand_idx]
            uncertainty_conditions = self._check_uncertainty(cand_self_entropy, cand_avg_probs, cand_max_prob, cand_uncertainty_inspection_times)
            if uncertainty_conditions:
                self.logger.debug(f"Uncertainty inspection conditions met for candidate token: {self.tokenizer.convert_ids_to_tokens(cand_batch_input_ids[0][-1].tolist())}")
                new_token = self.uncertainty_inspection(cand_batch_input_ids, cand_batch_attention_mask, cand_avg_probs)
                if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                    continue

                uncertainty_inspection_times[cand_idx] += 1
                targets[step][cand_idx] = new_token
                target_probs[step][cand_idx] = cand_avg_probs[new_token]
                cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                selected_indices.append(cand_idx)
                selected_input_ids.append(cand_batch_input_ids)
                selected_attention_mask.append(cand_batch_attention_mask)

            else:
                if cand_self_entropy < self.self_entropy_lower_bound or cand_max_prob > self.expectation_threshold:
                    new_token = cand_avg_probs.argmax()
                    if new_token == self.tokenizer.eos_token_id or self.tokenizer.decode(new_token) == "<|end_of_text|>":
                        continue

                    targets[step][cand_idx] = new_token
                    target_probs[step][cand_idx] = cand_max_prob
                    cand_batch_input_ids = torch.cat([cand_batch_input_ids, new_token.view(-1, 1).expand(-1, self.warmup_batch_size).reshape(-1, 1)], dim=-1)
                    cand_batch_attention_mask = torch.cat([cand_batch_attention_mask, cand_batch_attention_mask[:, -1].unsqueeze(1)], dim=-1)

                    selected_indices.append(cand_idx)
                    selected_input_ids.append(cand_batch_input_ids)
                    selected_attention_mask.append(cand_batch_attention_mask)

        if len(selected_indices) == 0:
            return None, None, None, None, None, None
        else:
            selected_indices = torch.tensor(selected_indices).long().to(self.device)
            input_ids = torch.cat(selected_input_ids, dim=0)
            attention_mask = torch.cat(selected_attention_mask, dim=0)
            targets = targets[:, selected_indices]
            target_probs = target_probs[:, selected_indices]
            target_mapping_record.append(selected_indices)
            return input_ids, attention_mask, targets, target_probs, target_mapping_record, uncertainty_inspection_times


    def _check_uncertainty(
        self,
        self_entropy: torch.Tensor,
        avg_probs: torch.Tensor,
        max_prob: torch.Tensor,
        uncertainty_inspection_times: torch.Tensor
    ) -> bool:
        """
        Check if the uncertainty condition is met.
        """
        cr1 = uncertainty_inspection_times < self.uncertainty_inspection_times_threshold
        cr2 = self_entropy < self.self_entropy_upper_bound
        cr3 = self_entropy > self.self_entropy_lower_bound
        cr4 = max_prob < self.expectation_threshold
        return cr1 and ((cr2 and cr3) or (cr2 and cr4))

    def _compute_self_entropy(
        self,
        probs_distribution: torch.Tensor,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Compute the self-entropy of a probability distribution.

        Args:
            probs_distribution (torch.Tensor): Probability distribution.
            eps (float): Small value to avoid log(0).

        Returns:
            torch.Tensor: Computed self-entropy.
        """
        # Add eps to avoid log(0) and handle NaN values
        probs_distribution = torch.nan_to_num(probs_distribution, nan=0.0) + eps
        # print(probs_distribution)

        # Normalize the distribution
        probs_distribution = probs_distribution / probs_distribution.sum(dim=-1, keepdim=True)

        # Compute entropy
        entropy = - (probs_distribution * torch.log(probs_distribution)).sum(dim=-1)
        return entropy



class BAITWrapper:
    """Handles the scanning of a single model"""
    def __init__(self, model_id: str, model_config: Dict, scan_args: ScanArguments, run_dir: str):
        self.model_id = model_id
        self.model_config = model_config
        self.scan_args = scan_args
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, model_id)
        os.makedirs(self.log_dir, exist_ok=True)

        self._setup_logging()
        self.bait_args, self.model_args, self.data_args = self._initialize_arguments()

    def _setup_logging(self):
        """Configure logging for this model scan"""
        log_file = os.path.join(self.log_dir, "scan.log")
        logger.remove()
        logger.add(sys.stderr, level="INFO")
        logger.add(log_file, rotation="100 MB", level="DEBUG")

    def _initialize_arguments(self) -> Tuple[BAITArguments, ModelArguments, DataArguments]:
        """Initialize and validate all arguments"""
        bait_args = BAITArguments()
        model_args = ModelArguments()
        data_args = DataArguments()

        # Validate and adjust arguments
        self._validate_arguments(bait_args, data_args)

        # Set up model and data arguments
        model_args, data_args = parse_model_args(self.model_config, data_args, model_args)
        model_args.adapter_path = os.path.join(self.scan_args.model_zoo_dir, self.model_id, "model")
        model_args.cache_dir = self.scan_args.cache_dir
        data_args.data_dir = self.scan_args.data_dir

        # Save arguments for reference
        self._save_arguments(bait_args, model_args, data_args)

        return bait_args, model_args, data_args

    def _validate_arguments(self, bait_args: BAITArguments, data_args: DataArguments):
        """Validate and adjust argument values"""
        if bait_args.warmup_batch_size > data_args.prompt_size:
            bait_args.warmup_batch_size = data_args.prompt_size
            logger.warning(f"warmup_batch_size was greater than prompt_size. Setting warmup_batch_size to {data_args.prompt_size}")

        if bait_args.uncertainty_inspection_times_threshold > bait_args.warmup_steps:
            bait_args.uncertainty_inspection_times_threshold = bait_args.warmup_steps
            logger.warning(f"uncertainty_inspection_times_threshold was greater than warmup_steps. Setting uncertainty_inspection_times_threshold to {bait_args.warmup_steps}")

        bait_args.batch_size = data_args.batch_size
        bait_args.prompt_size = data_args.prompt_size

    def _save_arguments(self, bait_args: BAITArguments, model_args: ModelArguments, data_args: DataArguments):
        """Save arguments to file"""
        with open(os.path.join(self.log_dir, "arguments.json"), "w") as f:
            json.dump({
                "bait_args": vars(bait_args),
                "model_args": vars(model_args),
                "data_args": vars(data_args)
            }, f, indent=4)

    def scan(self) -> Tuple[bool, Optional[str]]:
        """Run the scanning process for this model"""
        try:
            # Load model and data
            model, tokenizer, dataloader = self._load_model_and_data()

            # Run scan
            result = self._run_scan(model, tokenizer, dataloader)

            # Save results
            self._save_results(result)

            logger.info(f"Model {self.model_id} scanned successfully")
            return True, None

        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error scanning model {self.model_id}: {e}")
            return False, str(e)

    def _load_model_and_data(self) -> Tuple[torch.nn.Module, object]:
        """Load model and data"""
        logger.info("Loading model...")
        model, tokenizer = build_model(self.model_args)
        logger.info("Model loaded successfully")

        logger.info("Loading data...")
        dataset, dataloader = build_data_module(self.data_args, tokenizer, logger)
        logger.info("Data loaded successfully")

        return model, tokenizer, dataloader

    def _run_scan(self, model: torch.nn.Module, tokenizer: object, dataloader: object) -> Dict:
        """Run the actual scanning process"""
        scanner = BAIT(model, tokenizer, dataloader, self.bait_args, logger, device=torch.device('cuda'))
        start_time = time()
        scan_result = scanner.run()
        end_time = time()
        if scanner.save_pool:  # BAIT_SAVE_POOL=1: every inverted candidate with its per-step statistics, one JSON object per line
            with open(os.path.join(self.log_dir, "candidates.jsonl"), "w") as f:
                for rec in scanner.pool: f.write(json.dumps(rec) + "\n")

        return {
            "is_backdoor": scan_result.is_backdoor,
            "q_score": scan_result.best_target.q_score,
            "invert_target": scan_result.best_target.invert_target,
            "reasoning": scan_result.best_target.reasoning,
            "time_taken": end_time - start_time
        }

    def _save_results(self, result: Dict):
        """Save scanning results"""
        with open(os.path.join(self.log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=4)
