from typing import Dict, Union


class SettingsBuilder:

    def __init__(self, task, model_name, compute_profile) -> None:
        self.model_name = model_name
        self.task = task
        self.fine_tuned_name = None
        self.output_dir = f"../model_checkpoints/{model_name}"
        self.num_train_epochs = 1
        self.dataset = None
        self.compute_profile = compute_profile
        self.lora_r = 16
        self.lora_alpha = 32
        self.lora_dropout = 0.1
        self.use_4bit = True
        self.bnb_4bit_compute_dtype = "float16"
        self.save_steps = 0
        self.logging_steps = 2
        self.max_seq_length = None

        # BitsAndBytes Advanced
        self.bnb_4bit_use_quant_type = True
        self.use_nested_quant = False
        self.bnb_4bit_quant_type = "nf4"
        self.bnb_8bit_quant_type = "FP8"
        self.use_8bit = None
        self.load_in_4bit = None
        self.load_in_8bit = None
        # Trainer Advanced
        self.fp16 = False
        self.bf16 = False
        self.per_device_train_batch_size = 1
        self.per_device_eval_batch_size = 4
        self.gradient_accumulation_steps = 4
        self.gradient_checkpointing = True
        self.max_grad_norm = 0.3
        self.learning_rate = 2e-4
        self.weight_decay = 0.001
        self.optim = "paged_adamw_32bit"
        self.lr_scheduler_type = "cosine"
        self.max_steps = -1
        self.warmup_ratio = 0.03
        self.group_by_length = True

        # SFT Advanced
        self.packing = False
        self.device_map = {"": 0}
        self.model_desc = None

    def set_settings(self, settings_dict) -> None:
        """
        Update settings from a dictionary
        """
        for key, value in settings_dict.items():
            if key == "dataset":
                self.dataset = value
            elif key == "max_seq_length":
                if value == -1:
                    self.max_seq_length = None
                else:
                    self.max_seq_length = value
            elif key == "quantization":
                # Handle quantization settings
                if value == "4bit":
                    self.use_4bit = True
                    self.use_8bit = False
                    self.load_in_4bit = True
                    self.load_in_8bit = False

                elif value == "8bit":
                    self.use_4bit = False
                    self.use_8bit = True
                    self.load_in_4bit = False
                    self.load_in_8bit = True
                else:
                    self.use_4bit = False
                    self.use_8bit = False
                    self.load_in_4bit = False
                    self.load_in_8bit = False
            elif hasattr(self, key):
                # Convert string representations to appropriate types
                if isinstance(getattr(self, key), bool) and isinstance(value, str):
                    setattr(self, key, value.lower() == 'true')
                elif isinstance(getattr(self, key), (int, float)) and isinstance(value, str):
                    try:
                        if isinstance(getattr(self, key), int):
                            setattr(self, key, int(value))
                        else:
                            setattr(self, key, float(value))
                    except ValueError:
                        # Keep the original value if conversion fails
                        pass
                else:
                    setattr(self, key, value)

    def get_settings(self) -> Dict[str, Union[str, float]]:
        return {
            "task": self.task,
            "model_name": self.model_name,
            "num_train_epochs": self.num_train_epochs,
            "compute_specs": self.compute_profile,
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "use_4bit": self.use_4bit,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_quant_type": self.bnb_4bit_use_quant_type,
            "use_nested_quant": self.use_nested_quant,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "use_8bit": self.use_8bit,
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_8bit_quant_type": self.bnb_8bit_quant_type,
            "save_steps": self.save_steps,
            "output_dir": self.output_dir,
            "fine_tuned_name": self.fine_tuned_name,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "max_grad_norm": self.max_grad_norm,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "optim": self.optim,
            "lr_scheduler_type": self.lr_scheduler_type,
            "max_steps": self.max_steps,
            "warmup_ratio": self.warmup_ratio,
            "group_by_length": self.group_by_length,
            "packing": self.packing,
            "device_map": self.device_map,
            "max_seq_length": self.max_seq_length,
            "dataset": self.dataset,
            "logging_steps": self.logging_steps,

        }
