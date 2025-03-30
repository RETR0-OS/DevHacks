from LLMFinetuner import LLMFinetuner

## Basic Settings Parameters
model_name = "EleutherAI/gpt-neo-1.3B"
task = "summarization"
fine_tuned_name = "gpt-neo-1.3B_summarization"
output_dir = "cached_models"
num_train_epochs = 1
dataset = "../tests/low_summarization_train_set.jsonl"
compute_specs = "low_end"

### LoRA Basic
lora_r = 16 # This is the rank of the low-rank adaptation (LoRA) matrices. A higher rank means more parameters and potentially better performance, but also more memory usage.

### BitsAndBytes Basic
use_4bit = True # This is a flag to indicate whether to use 4-bit quantization. 4-bit quantization can significantly reduce the model size and improve inference speed, but it may also reduce the model's performance.
bnb_4bit_compute_dtype = "float16" # This specifies the data type used for computation during 4-bit quantization. Using float16 can speed up computation and reduce memory usage.

### Trainer Basic
save_steps = 0 # This is the number of steps after which the model will be saved during training. 0 means the model will not be saved during training.
logging_steps = 25 # This is the number of steps after which the training progress will be logged. This is useful for monitoring the training process.

### SFT Basic
max_seq_length = None # This is the maximum sequence length for the input data. If the input data is longer than this, it will be truncated. If it's None, it will use the model's default maximum sequence length.



## Advanced Settings Parameters

### LoRA Advanced
lora_alpha = 16 # This is a hyperparameter that controls the scaling of the LoRA weights. A higher value means that the LoRA weights will have a larger impact on the model's output.
lora_dropout = 0.1 # This is the dropout rate for the LoRA layers. Dropout is a regularization technique that helps prevent overfitting by randomly setting a fraction of the input units to 0 during training.

### BitsAndBytes Advanced
bnb_4bit_use_quant_type = True # This is a flag to indicate whether to use a specific quantization type for 4-bit quantization. It can help in reducing the model size and improving inference speed.
use_nested_quant = True # This is a flag to indicate whether to use nested quantization. Nested quantization can further reduce the model size and improve performance, especially for large models.
bnb_4bit_quant_type = "nf4" # This specifies the quantization type used for 4-bit quantization. nf4 is a quantization type that can help reduce the model size and improve performance.

### Trainer Advanced
fp16 = False # This is a flag to indicate whether to use 16-bit floating point precision (fp16) for training. Using fp16 can speed up training and reduce memory usage, but it requires compatible hardware (like NVIDIA GPUs with Tensor Cores).
bf16 = False # This is a flag to indicate whether to use bfloat16 precision for training. Bfloat16 is another format that can speed up training and reduce memory usage, but it also requires compatible hardware.

per_device_train_batch_size = 1 # This is the batch size for training. It determines how many samples are processed at once during the training phase.
per_device_eval_batch_size = 4 # This is the batch size for evaluation. It determines how many samples are processed at once during the evaluation phase.

gradient_accumulation_steps = 4 # Instead of updating the weights on the entire batch at once, it updates the weights after every 4 steps. This helps reduce memory usage for limited GPU resources.
gradient_checkpointing = True # This is a memory-saving technique. It trades off computation time for reduced memory usage by storing only a subset of activations during the forward pass and recomputing them during the backward pass.
max_grad_norm = 0.3 # This clips the gradients to prevent them from becoming too large, which can lead to instability during training. This is a form of gradient clipping.
learning_rate = 2e-4 # the learning rate for the training process
weight_decay = 0.001 # This adds a penalty to the loss function that discourages large weights, helping to prevent overfitting. It's a form of regularization.
optim = "paged_adamw_32bit" # Uses the 32-bit Paged Adam W optimizer (an update to the Adam W optimizer) to update the model weights.
lr_scheduler_type = "cosine" # This defines how the learning rate changes over time. A cosine scheduler gradually decreases the learning rate following a cosine function, which can help the model converge to a better solution.
max_steps = -1 # This sets the total number of training steps. -1 usually means it will be automatically calculated based on the number of epochs and batch size.
warmup_ratio = 0.03 #  This specifies the fraction of training steps used for a warm-up period where the learning rate is gradually increased from a small value to the initial learning rate. This helps stabilize the training process.

group_by_length = True #  This groups training examples of similar lengths together into batches, which can improve training efficiency.

### SFT Advanced
packing = False # Disables packing. Without packing, each training example is processed individually. Packing, when enabled, combines multiple shorter sequences into a single batch to reduce the amount of padding and make better use of GPU memory. However, it might not always be beneficial and is turned off here.
device_map = {"": 0} # This specifies the device mapping for the model. It indicates that the model should be loaded onto the first GPU (device 0). This is useful for multi-GPU setups.

fine_tuner = LLMFinetuner(task, model_name, compute_specs)
fine_tuner.set_settings(
    fine_tuned_name=fine_tuned_name,
    output_dir=output_dir,
    num_train_epochs=num_train_epochs,
    max_seq_length=max_seq_length,
    lora_r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    use_4bit=use_4bit,
    bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
    bnb_4bit_use_quant_type=bnb_4bit_use_quant_type,
    use_nested_quant=use_nested_quant,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    fp16=fp16,
    bf16=bf16,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    gradient_checkpointing=gradient_checkpointing,
    max_grad_norm=max_grad_norm,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim=optim,
    lr_scheduler_type=lr_scheduler_type,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    packing=packing,
    device_map=device_map
)

print("Loading dataset...")
fine_tuner.load_dataset(dataset)
print("Starting fine-tuning...")
fine_tuner.llm_finetuner()