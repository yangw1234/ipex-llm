#
# Copyright 2016 The BigDL Authors.
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
#

import torch
import os
os.environ["ACCELERATE_USE_IPEX"] = "1"
os.environ["ACCELERATE_USE_XPU"] = "1"

import transformers
from transformers import LlamaTokenizer

from peft import LoraConfig
import intel_extension_for_pytorch as ipex
from peft import prepare_model_for_kbit_training
from bigdl.llm.transformers.qlora import get_peft_model, TrainingArguments
from bigdl.llm.transformers import AutoModelForCausalLM
from datasets import load_dataset
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Predict Tokens using `generate()` API for Llama2 model')
    parser.add_argument('--repo-id-or-model-path', type=str, default="meta-llama/Llama-2-7b-hf",
                        help='The huggingface repo id for the Llama2 (e.g. `meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-chat-hf`) to be downloaded'
                             ', or the path to the huggingface checkpoint folder')

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    tokenizer = LlamaTokenizer.from_pretrained(model_path, trust_remote_code=True)

    data = load_dataset("Abirate/english_quotes")
    data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                load_in_4bit=True,
                                                optimize_model=False,
                                                trust_remote_code=True)
    model = model.to('xpu')
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=8, 
        lora_alpha=32, 
        target_modules=["q_proj", "k_proj", "v_proj"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, config)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    trainer = transformers.Trainer(
        model=model,
        train_dataset=data["train"],
        args=TrainingArguments(
            per_device_train_batch_size=4,
            gradient_accumulation_steps= 1,
            warmup_steps=20,
            max_steps=200,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=20,
            output_dir="outputs",
            optim="adamw_hf", # we currently do not have paged_adamw_8bit
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    result = trainer.train()
    print(result)
