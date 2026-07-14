import os
import time
import json
from tqdm import tqdm
from datetime import timedelta
from typing import Dict, List, Optional, Callable, Any
from collections import defaultdict

import ray
import torch
import torch.distributed as dist

from kdflow.datasets.utils import get_tokenizer_or_processor
from kdflow.utils.logging_utils import define_wandb_metrics, init_logger
from kdflow.utils.utils import zero_pad_sequences
from kdflow.utils.dynamic_bsz import rearrange_global_batch


logger = init_logger(__name__)

class OnPolicyKDTrainer:
    """
    Ray-based trainer for on-policy knowledge distillation.
    """
    
    def __init__(
        self,
        strategy,
        student_model,
        teacher_model,
        rollout_group,
        is_same_tokenizer: bool,
        train_dataloader,
        eval_dataloader=None,
        max_rollout_iters: int = None,
        num_rollout_iters_per_epoch: int = None,
        generate_kwargs: Dict[str, float] = None,
    ) -> None:
        """
        Initialize the trainer.
        
        Args:
            strategy: Training strategy containing configuration
            student_model: StudentActorGroup
            teacher_model: TeacherActorGroup
            rollout_group: RolloutGroup
            is_same_tokenizer: Whether student and teacher use same tokenizer
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader (optional)
            max_rollout_iters: Maximum rollout iterations in training
            num_rollout_iters_per_epoch: Number of rollout iterations per epoch
        """
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher = teacher_model
        self.rollout_group = rollout_group
        self.is_same_tokenizer = is_same_tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.max_rollout_iters = max_rollout_iters
        self.num_rollout_iters_per_epoch = num_rollout_iters_per_epoch
        self.generate_kwargs = generate_kwargs
        self.epochs = self.args.train.num_epochs
        
        self.image_key = getattr(self.args.data, "image_key", None)
        self.student_processor = get_tokenizer_or_processor(
            self.args.model.student_name_or_path,
            need_processor=self.image_key is not None,
        )
        self.teacher_processors = {}
        if self.args.kd.multi_teacher_config:
            for teacher_key, teacher_path in self.args.kd.multi_teacher_config.items():
                self.teacher_processors[teacher_key] = get_tokenizer_or_processor(
                    teacher_path, need_processor=self.image_key is not None,
                )
        elif self.args.model.teacher_name_or_path and not self.is_same_tokenizer:
            self.teacher_processors["default"] = get_tokenizer_or_processor(
                self.args.model.teacher_name_or_path,
                need_processor=self.image_key is not None,
            )
        
        self.world_size = self.args.train.num_nodes * self.args.train.num_gpus_per_node
        self.dp_size = self.world_size // self.args.model.ring_attn_size
        
        assert self.args.kd.kd_ratio == 1.0, "On-policy KD only supports kd_ratio=1.0."
        
        self.log_state = defaultdict(list)
        self._init_loggers()
    
    def _init_loggers(self) -> None:
        """Initialize wandb loggers."""
        self._wandb = None
        
        if self.args.log.use_wandb:
            import wandb
            
            self._wandb = wandb
            if self.args.log.wandb_mode != "offline" and not wandb.api.api_key:
                wandb.login()
            wandb.init(
                entity=self.args.log.wandb_org,
                project=self.args.log.wandb_project,
                group=self.args.log.wandb_group,
                name=self.args.log.wandb_run_name,
                config=vars(self.args),
                reinit=True,
                mode=self.args.log.wandb_mode,
                dir=self.args.log.wandb_dir,
            )
            
            define_wandb_metrics(wandb)
    
    def _print_training_config(self) -> None:
        """Log training configuration before training starts."""
        num_data = len(getattr(self.train_dataloader, "dataset", self.train_dataloader))
        num_update_per_rollout = self.args.rollout.n_samples_per_prompt * self.args.rollout.rollout_batch_size \
            // self.args.train.train_batch_size
        total_steps = self.max_rollout_iters * num_update_per_rollout
        grad_accum = self.args.train.train_batch_size * self.args.model.ring_attn_size \
            // (self.args.train.micro_train_batch_size * self.args.train.num_nodes * self.args.train.num_gpus_per_node)

        def log_config(name, value):
            logger.info(f"  {name:<32} {value}")
        
        logger.info("******* Start Training *******")
        log_config("Num GPUs:", self.world_size)
        log_config("Num Data:", num_data)
        log_config("Num Epochs:", self.epochs)
        log_config("Rollout Iterations Per Epoch:", self.num_rollout_iters_per_epoch)
        log_config("Total Rollout Iterations:", self.max_rollout_iters)
        log_config("Num Updates Per Rollout:", num_update_per_rollout)
        log_config("Total Num Updates:", total_steps)
        if self.args.train.use_dynamic_bsz:
            log_config("Enable Dynamic Batch Size:", self.args.train.use_dynamic_bsz)
            log_config("Max Token Len Per GPU:", self.args.train.max_token_len_per_gpu)
            log_config("Gradient Accumulation:", "dynamic")
        else:
            log_config("Per-device Batch Size:", self.args.train.micro_train_batch_size)
            log_config("Gradient Accumulation:", grad_accum)
        log_config("Learning Rate:", self.args.train.learning_rate)
        log_config("KD Algorithm:", self.args.kd.kd_algorithm)
        log_config("KD Loss Function:", self.args.kd.kd_loss_fn)
    
    def fit(self, global_step=0, start_epoch=0):
        self.global_step = global_step
        
        # Print training configuration and initialize loggers
        self._print_training_config()

        # Create Gloo IPC groups between training ranks and rollout engines (following slime)
        rollout_tp_size = getattr(self.args.rollout, "rollout_tp_size", 1)
        self.student.connect_rollout_engines(self.rollout_group.actors, rollout_tp_size)
        if self.args.model.student_name_or_path == self.args.model.teacher_name_or_path:   # for self-distillation
            num_gpus_per_teacher_actor = self.args.kd.teacher_tp_size * self.args.kd.teacher_pp_size
            self.student.connect_teacher_actors(self.teacher.teacher_engines, num_gpus_per_teacher_actor)
        
        self.start_time = time.time()
        num_micro_batches = self.args.train.train_batch_size // self.args.train.micro_train_batch_size
        
        for epoch in range(start_epoch, self.epochs):
            self.current_epoch = epoch
            self.train_dataloader.sampler.set_epoch(epoch)
            
            for prompt_batch in self.train_dataloader:
                step_start = time.time()
                self.global_step += 1
                
                rollout_start = time.time()
                rollout_samples = self.rollout(prompt_batch, **self.generate_kwargs)
                rollout_time = time.time() - rollout_start

                self.log_state["timing/rollout"].append(rollout_time)

                all_global_batches = []
                for i in range(0, len(rollout_samples), num_micro_batches):
                    global_batch = rollout_samples[i : i + num_micro_batches]

                    if self.args.train.use_dynamic_bsz:
                        global_batch = rearrange_global_batch(
                            global_batch,
                            max_token_len=self.args.train.max_token_len_per_gpu,
                            dp_size=self.dp_size,
                        )

                    global_batch_token_num = sum(mb["stu_loss_mask"].sum() for mb in global_batch)
                    avg_micro_batch_token_num = global_batch_token_num / len(global_batch)
                    for mb in global_batch:
                        mb["avg_micro_batch_token_num"] = avg_micro_batch_token_num
                    all_global_batches.append(global_batch)

                teacher_start = time.time()
                if self.args.train.enable_sleep:
                    self.teacher.wakeup()

                teacher_batches = [mb for global_batch in all_global_batches for mb in global_batch]
                teacher_batches = self.teacher.forward(teacher_batches)

                batch_idx = 0
                for i, global_batch in enumerate(all_global_batches):
                    next_batch_idx = batch_idx + len(global_batch)
                    all_global_batches[i] = teacher_batches[batch_idx:next_batch_idx]
                    batch_idx = next_batch_idx
                if batch_idx != len(teacher_batches):
                    raise RuntimeError(
                        f"Teacher forward returned {len(teacher_batches)} batches, expected {batch_idx}."
                    )

                if self.args.train.enable_sleep:
                    self.teacher.sleep()
                self.log_state["timing/teacher_forward"].append(time.time() - teacher_start)
                
                student_start = time.time()
                
                if self.args.train.enable_sleep:
                    self.student.wakeup()
                
                for global_batch in all_global_batches:
                    status_list = ray.get(self.student.async_run_distill(global_batch))
                    for k in status_list[0].keys():
                        self.log_state[k].append(sum(s[k] for s in status_list) / len(status_list))
                        
                self.log_state["timing/student_train"].append(time.time() - student_start)
                
                ray.get([actor.empty_cache.remote() for actor in self.student._actor_handlers])

                # update weights in rollout actors
                if self.args.train.enable_sleep:
                    self.rollout_group.wakeup(tags=["weights"])
                update_start = time.time()
                self.student.update_rollout_weights()
                self.log_state["timing/rollout_weight_sync"].append(time.time() - update_start)
                if self.args.train.enable_sleep:
                    self.rollout_group.sleep(tags=["weights"])
                
                # update weights in teacher actors (only for self-distillation)
                if self.args.model.teacher_name_or_path == self.args.model.student_name_or_path \
                    and self.global_step % self.args.kd.teacher_update_freq == 0:
                    if self.args.train.enable_sleep:
                        self.teacher.wakeup(tags=["weights"])
                    teacher_update_start = time.time()
                    self.student.update_teacher_weights()
                    self.log_state["timing/teacher_weight_sync"].append(time.time() - teacher_update_start)
                    if self.args.train.enable_sleep:
                        self.teacher.sleep(tags=["weights"])
                    
                if self.args.train.enable_sleep:
                    self.student.sleep()

                self.log_state["timing/step_time"].append(time.time() - step_start)
                self.logging()
                
                if self.global_step % self.args.train.save_steps == 0:
                    self.strategy.log(f"Saving model at global step {self.global_step}")
                    save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}_global_step_{self.global_step}")
                    ray.get(self.student.async_save_model(save_path))
        
            # save model after each epoch
            self.strategy.log(f"Saving model after epoch {epoch + 1}")
            save_path = os.path.join(self.args.train.save_path, f"epoch_{epoch + 1}")
            ray.get(self.student.async_save_model(save_path))

        total_time = time.time() - self.start_time
        self.strategy.log(f"Training done, totally cost {str(timedelta(seconds=total_time)).split('.')[0]}")

        if self._wandb is not None:
            self._wandb.finish()
            
    def rollout(self, prompt_batch: List[Dict[str, str]], **kwargs) -> List[dict]:
        """Generate samples using rollout engine.

        Args:
            prompt_batch: List of dicts with keys: datasource, stu_prompt, tea_prompt, label
            **kwargs: Additional arguments for generation

        Returns:
            List of rollout sample dicts containing generated samples
        """
        if self.args.train.enable_sleep:
            self.rollout_group.wakeup()

        # Extract prompts and labels from batch
        all_stu_prompts = [item["stu_prompt"] for item in prompt_batch]
        all_tea_prompts = [item["tea_prompt"] for item in prompt_batch]
        all_labels = [item["label"] for item in prompt_batch]
        all_images = [item.get("images") for item in prompt_batch] if self.image_key else None
        all_teacher_routing_keys = [item.get("teacher_routing_key") for item in prompt_batch] if self.args.kd.multi_teacher_config else None
        
        # Expand prompt list based on the number of samples per prompt
        n_samples_per_prompt = self.args.rollout.n_samples_per_prompt
        all_stu_prompts = sum([[p] * n_samples_per_prompt for p in all_stu_prompts], [])
        all_tea_prompts = sum([[p] * n_samples_per_prompt for p in all_tea_prompts], [])
        all_labels = sum([[label] * n_samples_per_prompt for label in all_labels], [])
        if all_images:
            all_images = sum([[imgs] * n_samples_per_prompt for imgs in all_images], [])
        if all_teacher_routing_keys:
            all_teacher_routing_keys = sum([[key] * n_samples_per_prompt for key in all_teacher_routing_keys], [])
        
        all_outputs = self.rollout_group.generate(all_stu_prompts, self.generate_kwargs, image_data=all_images)

        rollout_dir = os.path.join(self.args.train.save_path, "rollout_data")
        os.makedirs(rollout_dir, exist_ok=True)
        with open(os.path.join(rollout_dir, f"{self.global_step}.jsonl"), "w") as f:
            for prompt, output, label in zip(all_stu_prompts, all_outputs, all_labels):
                record = {"prompt": prompt, "output": output["text"]}
                if "reward_result" in output:
                    record["reward_result"] = output["reward_result"]
                if label is not None and label != "":
                    record["label"] = label
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # Process outputs into rollout samples
        sample_list = [
            self._build_rollout_sample(
                stu_prompt=all_stu_prompts[i],
                tea_prompt=all_tea_prompts[i],
                output=all_outputs[i],
                label=all_labels[i],
                images=all_images[i] if all_images and all_images[i] else None,
                teacher_routing_key=all_teacher_routing_keys[i] \
                if all_teacher_routing_keys and all_teacher_routing_keys[i] else None,
            )
            for i in range(len(all_outputs))
        ]

        # Compute rollout-level length statistics before samples are split into
        # micro-batches. The scalar fields are only needed for logging, so remove
        # them before sending samples through the teacher/student pipeline.
        length_names = ("prompt_length", "response_length", "total_length")
        length_values = {
            name: [sample.pop(name).item() for sample in sample_list]
            for name in length_names
        }
        for name, values in length_values.items():
            self.log_state[f"rollout/{name}_mean"].append(sum(values) / len(values))
            self.log_state[f"rollout/{name}_max"].append(max(values))

        max_response_length = self.generate_kwargs["max_new_tokens"]
        response_lengths = length_values["response_length"]
        response_clip_ratio = sum(
            length >= max_response_length for length in response_lengths
        ) / len(response_lengths)
        self.log_state["rollout/response_clip_ratio"].append(response_clip_ratio)
        
        # Print sample for debugging
        sample0 = sample_list[0]["stu_prompts"][0] + sample_list[0]["stu_responses"][0]
        if self.args.rollout.print_rollout_sample:
            print(sample0)
        
        micro_batch_list = self._collate_micro_batches(sample_list, self.args.train.micro_train_batch_size)
        
        if self.args.train.enable_sleep:
            self.rollout_group.sleep()

        return micro_batch_list
    
    @staticmethod
    def _collate_values(key: str, values: list):
        v0 = values[0]
        if isinstance(v0, torch.Tensor):
            return zero_pad_sequences(values, side="right", value=0)
        if isinstance(v0, list):
            return sum(values, [])
        if v0 is None:
            return None
        return values

    def _collate_micro_batches(self, sample_list: List[Dict], batch_size: int) -> List[Dict]:
        """Collate single samples into micro-batches."""
        micro_batch_list = []
        for i in range(0, len(sample_list), batch_size):
            batch_samples = sample_list[i : i + batch_size]
            micro_batch = {
                key: self._collate_values(key, [s[key] for s in batch_samples])
                for key in batch_samples[0]
            }
            micro_batch_list.append(micro_batch)
        return micro_batch_list

    def _tokenize_sample(
        self, 
        prompt: str, 
        response: str, 
        processor,
        prefix: str,
        images=None,
    ) -> Dict[str, Any]:
        """Tokenize prompt + response for a single sample.

        Args:
            prompt: Chat-templated prompt string.
            response: Response string.
            processor: Processor or tokenizer for the model.
            prefix: 'stu' or 'tea'.
            images: PIL images (or None for text-only).

        Returns:
            Dict with ``{prefix}_input_ids``, ``{prefix}_attn_mask``, ``{prefix}_loss_mask``
            and optional multimodal fields.
        """
        tokenizer = getattr(processor, "tokenizer", processor)
        resp_tok = tokenizer(response, return_tensors="pt", add_special_tokens=False)
        resp_len = resp_tok["input_ids"].shape[1]

        full_input = {"text": prompt + response}
        if images:
            full_input["images"] = images
        full_tok = processor(**full_input, return_tensors="pt", add_special_tokens=False)
        prompt_len = full_tok["input_ids"].shape[1] - resp_len

        # since rollout response does not contain eos token, we need to add it manually
        eos_token_id = tokenizer.eos_token_id
        input_ids = torch.cat([full_tok["input_ids"][0], full_tok["input_ids"][0].new_tensor([eos_token_id])])
        attn_mask = torch.cat([full_tok["attention_mask"][0], full_tok["attention_mask"][0].new_ones(1)])
        loss_mask = torch.tensor(
            [False] * (prompt_len - 1) + [True] * (resp_len + 1) + [False],
            device=input_ids.device,
        )

        result = {
            f"{prefix}_input_ids": input_ids,
            f"{prefix}_attn_mask": attn_mask,
            f"{prefix}_loss_mask": loss_mask,
            f"_{prefix}_prompt_length": prompt_len,
        }
        multi_modal_inputs = {
            k: torch.as_tensor(v) for k, v in full_tok.items()
            if k not in ("input_ids", "attention_mask", "mm_token_type_ids")
        }
        if multi_modal_inputs:
            result[f"_{prefix}_multi_modal_inputs"] = multi_modal_inputs

        return result

    def _get_teacher_processor(self, teacher_routing_key=None):
        """Get the teacher processor for the given routing key."""
        if teacher_routing_key and teacher_routing_key in self.teacher_processors:
            return self.teacher_processors[teacher_routing_key]
        if "default" in self.teacher_processors:
            return self.teacher_processors["default"]
        return self.student_processor

    def _build_rollout_sample(
        self,
        stu_prompt: str,
        tea_prompt: str,
        output,
        label: str,
        images=None,
        teacher_routing_key=None,
    ) -> Dict[str, Any]:
        """
        Build a single rollout sample with both student and teacher tokenizations.
        
        Args:
            stu_prompt: Student prompt string (formatted with student's chat template)
            tea_prompt: Teacher prompt string (formatted with teacher's chat template)
            output: rollout output object
            label: Label string
            images: PIL images (or None for text-only).
            teacher_routing_key: Routed teacher key for multi-teacher distillation.
        Returns:
            Dict containing all sample fields
        """
        # Decode response using student tokenizer
        response_ids = output["output_ids"]
        response_text = output["text"]
        
        stu_tokens = self._tokenize_sample(
            stu_prompt, response_text, self.student_processor, "stu", images=images
        )
        
        if not self.is_same_tokenizer or tea_prompt != stu_prompt:
            teacher_processor = self._get_teacher_processor(teacher_routing_key)
            tea_tokens = self._tokenize_sample(
                tea_prompt, response_text, teacher_processor, "tea", images=images
            )
        else:
            tea_tokens = {
                "tea_input_ids": stu_tokens["stu_input_ids"].clone(),
                "tea_attn_mask": stu_tokens["stu_attn_mask"].clone(),
                "tea_loss_mask": stu_tokens["stu_loss_mask"].clone(),
            }
        
        prompt_length = stu_tokens["_stu_prompt_length"]
        response_length = len(response_ids)
        total_length = stu_tokens["stu_attn_mask"].sum().item()
        
        # Teacher feed input_ids for SGLang. Text: reuse tea_input_ids (exact).
        # Multimodal: tokenize raw text (single image placeholder) so SGLang re-expands it.
        teacher_processor = self._get_teacher_processor(teacher_routing_key)
        tokenizer = getattr(teacher_processor, "tokenizer", teacher_processor)
        if images:
            feed_ids = tokenizer(tea_prompt + response_text, add_special_tokens=False)["input_ids"]
            tea_feed_input_ids = feed_ids + [tokenizer.eos_token_id]
        else:
            tea_feed_input_ids = tea_tokens["tea_input_ids"].tolist()

        sample = {
            **{k: v for k, v in tea_tokens.items() if not k.startswith("_")},
            **{k: v for k, v in stu_tokens.items() if not k.startswith("_")},
            "tea_feed_input_ids": [tea_feed_input_ids],
            "rollout_log_probs": None,
            "stu_prompts": [stu_prompt],
            "stu_responses": [response_text],
            "tea_prompts": [tea_prompt],
            "labels": [label],
            "prompt_length": torch.FloatTensor([[prompt_length]]),
            "response_length": torch.FloatTensor([[response_length]]),
            "total_length": torch.FloatTensor([[total_length]]),
        }
        stu_mm = stu_tokens.get("_stu_multi_modal_inputs")
        if stu_mm is not None:
            sample["stu_multi_modal_inputs"] = [stu_mm]
        if images:
            sample["images"] = [images]
        if teacher_routing_key is not None:
            sample["teacher_routing_key"] = teacher_routing_key
        return sample
            
    def logging(self):
        if self.global_step % self.args.log.logging_steps == 0:
            progress = self.global_step / self.num_rollout_iters_per_epoch / self.epochs
            eta = int(time.time() - self.start_time) * (1 - progress) / progress
            progress_str = "epoch [{current_epoch}/{total_epoch}], " \
                "step [{current_step}/{total_step}], " \
                "train_progress [{progress:.2f}%], " \
                "Elapsed: {elapsed}, " \
                "ETA: {eta}, ".format(
                current_epoch=self.current_epoch + 1, 
                total_epoch=self.epochs, 
                current_step=self.global_step, 
                total_step=self.num_rollout_iters_per_epoch * self.epochs, 
                progress=progress * 100,
                elapsed=str(timedelta(seconds=(time.time() - self.start_time))).split(".")[0],
                eta=str(timedelta(seconds=eta)).split(".")[0]
            )
            for k in self.log_state:
                if isinstance(self.log_state[k], list) and len(self.log_state[k]) > 0:
                    values = self.log_state[k]
                    self.log_state[k] = (
                        max(values) if k.endswith("_max") else sum(values) / len(values)
                    )
            log_info = []
            for k in self.log_state:
                # Skip keys that have no values logged in this interval (e.g. teacher weight sync
                # is only logged every teacher_update_freq steps).
                if isinstance(self.log_state[k], list):
                    continue
                if k == "train/lr":
                    log_info.append(f"{k}: {self.log_state[k]:.6e}")
                else:
                    log_info.append(f"{k}: {self.log_state[k]:.6f}")
            # Append average phase times
            log_str = ", ".join(log_info)
            log_str = progress_str + log_str
            self.strategy.log(log_str)

            if self._wandb is not None:
                logs = {"train/global_step": self.global_step}
                for k in self.log_state:
                    if isinstance(self.log_state[k], list):
                        continue
                    logs[k] = self.log_state[k]
                self._wandb.log(logs)

            for k in self.log_state:
                self.log_state[k] = []
