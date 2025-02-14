import configparser

from datasets import Dataset
import pandas as pd
from peft import LoraConfig, TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer, )
import torch.cuda

conf = configparser.ConfigParser()
conf.read("./config.ini",encoding="utf-8")

# 处理函数工厂
class ProcessFactory:
    def __init__(self, model_type):
        self.model_type = model_type

    def process_func(self, example):
        if self.model_type == "deepseek":
            return ds_process_func_mistral(example)

# 训练集总标题
train_title = conf.get("dateset", "dateset_title")
# 训练集文件
dateset_file = conf.get("dateset", "dateset_file")
# 模型路径
model_path = conf.get("model", "model_file")
# 模型输出
model_out = conf.get("model", "model_out")
model_type = conf.get("model", "model_type")
# 将JSON文件转换为CSV文件
df = pd.read_json(r"E:\Lora\handle_dataset.json")
ds = Dataset.from_pandas(df)
# 自动识别模型并获取合适的分词器
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
# 分词器填充在右侧
tokenizer.padding_side = 'right'
# 自动加载模型
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                             device_map="cuda:0")
model.enable_input_require_grads()

# 训练集处理器
def ds_process_func_mistral(example):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer((f"<｜begin▁of▁sentence｜>{train_title}\n"
                             f"User: {example['instruction'] + example['input']}\nAssistant: "
                             ).strip(),
                            add_special_tokens=False)
    response = tokenizer(f"{example['output']}<｜end▁of▁sentence｜>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 工厂构建
process_ff = ProcessFactory(model_type)

# 构建分词器id
tokenized_id = ds.map(process_ff.process_func, remove_columns=ds.column_names)

# Lora配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
    # 现存问题只微调部分演示即可
    inference_mode=False,  # 训练模式
    r=8,  # Lora 秩
    lora_alpha=32,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1  # Dropout 比例
)

# 训练参数
args = TrainingArguments(
    output_dir=model_out,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=32,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-5,
    save_on_each_node=True,
    gradient_checkpointing=True,
    fp16=True,
)

# 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
if __name__ == '__main__':
    print(model)
    torch.cuda.empty_cache()
    trainer.train()
