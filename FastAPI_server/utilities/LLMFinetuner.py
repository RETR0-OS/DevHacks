import torch
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, DataCollatorForLanguageModeling

class LLMFinetuner:
    def __init__(self, task, model_name, compute_specs="low_end"):
        if task == "question-answering":
            self.task = TaskType.QUESTION_ANS
        elif task == "text-generation":
            self.task = TaskType.CAUSAL_LM
        elif task == "summarization":
            self.task = TaskType.SEQ_2_SEQ_LM
        self.compute_specs = compute_specs
        self.bnb_4bit_quant_type = None
        self.optim = None
        self.weight_decay = None
        self.learning_rate = None
        self.gradient_checkpointing = None
        self.max_grad_norm = None
        self.gradient_accumulation_steps = None
        self.per_device_eval_batch_size = None
        self.per_device_train_batch_size = None
        self.lr_scheduler_type = None
        self.bf16 = None
        self.fp16 = None
        self.max_steps = None
        self.warmup_ratio = None
        self.group_by_length = None
        self.packing = None
        self.use_nested_quant = None
        self.bnb_4bit_use_quant_type = None
        self.lora_dropout = None
        self.lora_alpha = None
        self.max_seq_length = None
        self.logging_steps = None
        self.bnb_4bit_compute_dtype = None
        self.save_steps = None
        self.lora_r = None
        self.device_map = None
        self.num_train_epochs = None
        self.use_4bit = None
        self.output_dir = None
        self.fine_tuned_name = None
        self.dataset = None
        # self.task = task
        self.model_name = model_name

    @staticmethod
    def format_example(example, task, specs):
        # Concatenate the context, question, and answer into a single text field.
        if task == TaskType.QUESTION_ANS:
            if specs=="low_end":
                return {
                    "text": f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer: {example['answer']}"
                }
        elif task == TaskType.CAUSAL_LM:
            if specs == "low_end":
                return {"text": f"Input: {example['input']}\nOutput: {example['output']}"}
            elif specs == "mid_range":
                return {"text": f"Instruction:{example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"}
            elif specs == "high_end":
                return {"text": f"System:{example['system']}\nInstruction:{example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"}
        elif task == TaskType.SEQ_2_SEQ_LM:
            if specs == "low_end":
                return {"text": f"Article: {example['article']}\nSummary: {example['summary']}"}
            elif specs == "mid_range" or specs == "high_end":
                return {"text": f"Instruction: {example['instruction']}\nArticle: {example['article']}\nSummary: {example['summary']}"}

    def load_dataset(self, dataset_path):
        dataset = load_dataset("json", data_files={"train": dataset_path})
        dataset = dataset.map(lambda example: self.format_example(example, self.task, self.compute_specs))['train']
        print(dataset)
        self.dataset = dataset

    def set_settings(self, **kwargs):
        """
        Set model training settings from keyword arguments.
        Groups settings by category for better organization.
        """
        # Basic settings
        self.fine_tuned_name = kwargs.get('fine_tuned_name')
        self.output_dir = kwargs.get('output_dir')
        self.num_train_epochs = kwargs.get('num_train_epochs')
        self.max_seq_length = kwargs.get('max_seq_length')

        # LoRA settings
        self.lora_r = kwargs.get('lora_r')
        self.lora_alpha = kwargs.get('lora_alpha')
        self.lora_dropout = kwargs.get('lora_dropout')

        # Quantization settings
        self.use_4bit = kwargs.get('use_4bit')
        self.bnb_4bit_compute_dtype = kwargs.get('bnb_4bit_compute_dtype')
        self.bnb_4bit_use_quant_type = kwargs.get('bnb_4bit_use_quant_type')
        self.use_nested_quant = kwargs.get('use_nested_quant')
        self.bnb_4bit_quant_type = kwargs.get('bnb_4bit_quant_type')

        # Training precision
        self.fp16 = kwargs.get('fp16')
        self.bf16 = kwargs.get('bf16')

        # Batch and steps configuration
        self.per_device_train_batch_size = kwargs.get('per_device_train_batch_size')
        self.per_device_eval_batch_size = kwargs.get('per_device_eval_batch_size')
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps')
        self.save_steps = kwargs.get('save_steps')
        self.logging_steps = kwargs.get('logging_steps')

        # Optimization settings
        self.gradient_checkpointing = kwargs.get('gradient_checkpointing')
        self.max_grad_norm = kwargs.get('max_grad_norm')
        self.learning_rate = kwargs.get('learning_rate')
        self.weight_decay = kwargs.get('weight_decay')
        self.optim = kwargs.get('optim')
        self.lr_scheduler_type = kwargs.get('lr_scheduler_type')
        self.max_steps = kwargs.get('max_steps')
        self.warmup_ratio = kwargs.get('warmup_ratio')

        # Advanced configuration
        self.group_by_length = kwargs.get('group_by_length')
        self.packing = kwargs.get('packing')
        self.device_map = kwargs.get('device_map')

    def llm_finetuner(self):
        compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        bits_n_bytes_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bits_n_bytes_config,
            device_map=self.device_map,
            use_cache=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        peft_config = LoraConfig(
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            r=self.lora_r,
            bias="none",
            task_type=self.task,
        )

        training_arguments = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            optim=self.optim,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            fp16=self.fp16,
            bf16=self.bf16,
            max_grad_norm=self.max_grad_norm,
            max_steps=self.max_steps,
            group_by_length=self.group_by_length,
            lr_scheduler_type=self.lr_scheduler_type,
        )

        model = get_peft_model(model, peft_config)

        trainer = SFTTrainer(
            model=model,
            train_dataset=self.dataset,
            args=training_arguments,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),  # type: ignore
        )
        trainer.train()
        trainer.model.save_pretrained(self.fine_tuned_name)
        self.report_finish()

    def report_finish(self):
        print("*" * 100)
        print("Model fine-tuned successfully!")
        print(f"Model save to {self.fine_tuned_name}")
        print(f"Try out your new model in our chat playground!")
        print("*" * 100)