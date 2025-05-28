import os
import signal
from huggingface_hub import HfApi
from huggingface_hub import errors as hf_errors
import uvicorn

## Validate HuggingFace login
try:
    api = HfApi()
    api.whoami()
except hf_errors.LocalTokenNotFoundError:
    print(f"""
    {'*' * 100}
    You are not logged in to the Hugging Face Hub. 
    1) Create an account on https://huggingface.co/
    2) Generate a finegrained API token from your account settings (https://huggingface.co/docs/hub/en/security-tokens).
    3) Run the command below to login:
        huggingface-cli login
    4) Paste the token when prompted.
    {'*' * 100}
    """)
    os.kill(os.getpid(), signal.SIGTERM)
except hf_errors.HTTPError:
    print(f"""
    {"*"*100}
    You are not connected to the internet.
    Please check your internet connection and try again.
    {"*" * 100}
    """)
    os.kill(os.getpid(), signal.SIGTERM)


## Standard Python imports
import uuid
import json
import traceback
from pydantic import BaseModel, field_validator
import sqlite3

## FastAPI imports
from fastapi import FastAPI, Request, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

## ModelForge Utility Classes imports
from utilities.hardware_detector import HardwareDetector
from utilities.settings_builder import SettingsBuilder
from utilities.CausalLLMTuner import CausalLLMFinetuner
from utilities.Seq2SeqLMTuner import Seq2SeqFinetuner
from utilities.QuestionAnsweringTuner import QuestionAnsweringTuner

## Server Global Configurations
app = FastAPI()
hardware_detector = HardwareDetector()
settings_builder = SettingsBuilder(None, None, None)
settings_cache = {}
app_name = "ModelForge"
finetuning_status = {"status": "idle", "progress": 0, "message": ""}
datasets_dir = "./datasets"
model_path = os.path.join(os.path.dirname(__file__), "model_checkpoints")

try:
    db_connection = sqlite3.connect("database/modelforge.db")
    db_cursor = db_connection.cursor()
    db_cursor.execute('''
        CREATE TABLE IF NOT EXISTS modelforge_models (
            model_id STRING PRIMARY KEY,
            model_name STRING NOT NULL,
            model_path STRING, 
            pipeline_task STRING NOT NULL,
            compute_specs STRING NOT NULL
        )
    ''')

except sqlite3.Error as e:
    print(f"Database connection error: {e}")
    raise HTTPException(status_code=500, detail="Database connection error. Please try again later.")

origins = [
    "*",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


## Pydantic Data Validator Classes
class TaskFormData(BaseModel):
    task: str
    @field_validator("task")
    def validate_task(cls, task):
        if task not in ["text-generation", "summarization", "extractive-question-answering"]:
            raise ValueError("Invalid task. Must be one of 'text-generation', 'summarization', or 'extractive-question-answering'.")
        return task

class SelectedModelFormData(BaseModel):
    selected_model: str
    @field_validator("selected_model")
    def validate_selected_model(cls, selected_model):
        if not selected_model:
            raise ValueError("Selected model cannot be empty.")
        return selected_model

class SettingsFormData(BaseModel):
    task: str
    model_name: str
    num_train_epochs: int
    compute_specs: str
    lora_r: int
    lora_alpha: int
    lora_dropout: int
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_quant_type: str
    use_nested_quant: bool
    bnb_4bit_quant_type: str
    fp16: bool
    bf16: bool
    per_device_train_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    max_grad_norm: float
    learning_rate: float
    weight_decay: float
    optim: str
    lr_scheduler_type: str
    max_steps: int
    warmup_ratio: float
    group_by_length: bool
    packing: bool
    max_seq_length: int
    dataset: str

    @field_validator("dataset")
    def validate_dataset_prescence(cls, dataset):
        if not dataset:
            raise ValueError("Dataset cannot be empty.")
        return dataset
    @field_validator("task")
    def validate_task(cls, task):
        if task not in ["text-generation", "summarization", "question-answering"]:
            raise ValueError("Invalid task. Must be one of 'text-generation', 'summarization', or 'question-answering'.")
        return task
    @field_validator("model_name")
    def validate_model_name(cls, model_name):
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        return model_name
    @field_validator("num_train_epochs")
    def validate_num_train_epochs(cls, num_train_epochs):
        if num_train_epochs <= 0:
            raise ValueError("Number of training epochs must be greater than 0.")
        elif num_train_epochs > 30:
            raise ValueError("Number of training epochs must be less than 50.")
        return num_train_epochs
    @field_validator("compute_specs")
    def validate_compute_specs(cls, compute_specs):
        if compute_specs not in ["low_end", "mid_range", "high_end"]:
            raise ValueError("Invalid compute specs. Must be one of 'low_end', 'mid_range', or 'high_end'.")
        return compute_specs
    @field_validator("lora_r")
    def validate_lora_r(cls, lora_r):
        if lora_r not in [8, 16, 32, 64]:
            raise ValueError("LoRA rank must be 8, 16, 32, or 64.")
        return lora_r
    @field_validator("lora_alpha")
    def validate_lora_alpha(cls, lora_alpha):
        if lora_alpha >= 0.5:
            raise ValueError("LoRA learning rate is too high. Gradients will explode.")
        return lora_alpha
    @field_validator("lora_dropout")
    def validate_lora_dropout(cls, lora_dropout):
        if lora_dropout < 0.0 or lora_dropout > 1.0:
            raise ValueError("LoRA dropout probability must be between 0.0 and 1.0.")
        return lora_dropout
    @field_validator("use_4bit")
    def validate_use_4bit(cls, use_4bit):
        if not isinstance(use_4bit, bool):
            raise ValueError("use_4bit must be a boolean value.")
        return use_4bit
    @field_validator("bnb_4bit_compute_dtype")
    def validate_bnb_4bit_compute_dtype(cls, bnb_4bit_compute_dtype):
        if bnb_4bit_compute_dtype not in ["float16", "bfloat16"]:
            raise ValueError("Invalid compute dtype. Must be 'float16' or 'bfloat16'.")
        return bnb_4bit_compute_dtype
    @field_validator("bnb_4bit_use_quant_type")
    def validate_bnb_4bit_use_quant_type(cls, bnb_4bit_use_quant_type):
        if bnb_4bit_use_quant_type not in ["fp4", "int8"]:
            raise ValueError("Invalid quantization type. Must be 'fp4' or 'int8'.")
        return bnb_4bit_use_quant_type
    @field_validator("use_nested_quant")
    def validate_use_nested_quant(cls, use_nested_quant):
        if not isinstance(use_nested_quant, bool):
            raise ValueError("use_nested_quant must be true or false.")
        return use_nested_quant
    @field_validator("bnb_4bit_quant_type")
    def validate_bnb_4bit_quant_type(cls, bnb_4bit_quant_type):
        if bnb_4bit_quant_type not in ["fp4", "int8"]:
            raise ValueError("Invalid quantization type. Must be 'fp4' or 'int8'.")
        return bnb_4bit_quant_type
    @field_validator("fp16")
    def validate_fp16(cls, fp16):
        if not isinstance(fp16, bool):
            raise ValueError("fp16 must be true or false.")
        return fp16
    @field_validator("bf16")
    def validate_bf16(cls, bf16):
        if not isinstance(bf16, bool):
            raise ValueError("bf16 must be true or false.")
        return bf16
    @field_validator("per_device_train_batch_size")
    def validate_per_device_train_batch_size(cls, per_device_train_batch_size, compute_specs):
        if per_device_train_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        elif per_device_train_batch_size > 3 and compute_specs != "high_end":
            raise ValueError("Batch size must be less than 4. Your device cannot support a higher batch size.")
        elif per_device_train_batch_size > 8 and compute_specs == "high_end":
            raise ValueError("Batch size must be less than 9. Higher batch sizes cause out of memory error.")
        return per_device_train_batch_size
    @field_validator("per_device_eval_batch_size")
    def validate_per_device_eval_batch_size(cls, per_device_eval_batch_size):
        if per_device_eval_batch_size <= 0:
            raise ValueError("Batch size must be greater than 0.")
        return per_device_eval_batch_size
    @field_validator("gradient_accumulation_steps")
    def validate_gradient_accumulation_steps(cls, gradient_accumulation_steps):
        if gradient_accumulation_steps <= 0:
            raise ValueError("Gradient accumulation steps must be greater than 0.")
        return gradient_accumulation_steps
    @field_validator("gradient_checkpointing")
    def validate_gradient_checkpointing(cls, gradient_checkpointing):
        if not isinstance(gradient_checkpointing, bool):
            raise ValueError("Gradient checkpointing must be true or false.")
        return gradient_checkpointing
    @field_validator("max_grad_norm")
    def validate_max_grad_norm(cls, max_grad_norm):
        if max_grad_norm <= 0:
            raise ValueError("Max gradient norm must be greater than 0.")
        return max_grad_norm
    @field_validator("learning_rate")
    def validate_learning_rate(cls, learning_rate):
        if learning_rate <= 0:
            raise ValueError("Learning rate must be greater than 0.")
        elif learning_rate > 0.3:
            raise ValueError("Learning rate must be less than 0.3. Higher learning rates cause exploding gradients.")
        return learning_rate
    @field_validator("weight_decay")
    def validate_weight_decay(cls, weight_decay):
        if weight_decay < 0:
            raise ValueError("Weight decay must be greater than or equal to 0.")
        return weight_decay
    @field_validator("optim")
    def validate_optim(cls, optim):
        if optim not in ["paged_adamw_32bit", "paged_adamw_8bit", "adamw_torch", "adamw_hf"]:
            raise ValueError("Invalid optimizer. Must be one of 'paged_adamw_32bit', 'paged_adamw_8bit', 'adamw_torch', or 'adamw_hf'.")
        return optim
    @field_validator("lr_scheduler_type")
    def validate_lr_scheduler_type(cls, lr_scheduler_type):
        if lr_scheduler_type not in ["linear", "cosine", "constant", "constant_with_warmup"]:
            raise ValueError("Invalid learning rate scheduler type. Must be one of 'linear', 'cosine', 'constant', or 'constant_with_warmup'.")
        return lr_scheduler_type
    @field_validator("max_steps")
    def validate_max_steps(cls, max_steps):
        if max_steps < 0:
            raise ValueError("Max steps must be greater than or equal to 0.")
        elif max_steps > 100:
            raise ValueError("Max steps must be less than 100.")
        return max_steps
    @field_validator("warmup_ratio")
    def validate_warmup_ratio(cls, warmup_ratio):
        if warmup_ratio < 0.0 or warmup_ratio > 1.0:
            raise ValueError("Warmup ratio must be between 0.0 and 1.0.")
        return warmup_ratio
    @field_validator("group_by_length")
    def validate_group_by_length(cls, group_by_length):
        if not isinstance(group_by_length, bool):
            raise ValueError("Group by length must be true or false.")
        return group_by_length
    @field_validator("packing")
    def validate_packing(cls, packing):
        if not isinstance(packing, bool):
            raise ValueError("Packing must be true or false.")
        return packing
    @field_validator("max_seq_length")
    def validate_max_seq_length(cls, max_seq_length):
        if max_seq_length < -1:
            raise ValueError("Max sequence length must be greater than 0 or it should be -1 if you want to use the default Max Sequence Length.")
        return max_seq_length
    @field_validator("dataset")
    def validate_dataset(cls, dataset):
        if not dataset:
            raise ValueError("Dataset cannot be empty.")
        return dataset


## Server endpoints
@app.get("/api")
async def home(request: Request) -> JSONResponse:
    return JSONResponse({
        "app_name": app_name,
        "app_description": "No-code LLM finetuning for CUDA environments",
        "features": [
            "Intuitive no-code interface",
            "PEFT and LoRA-based finetuning",
            "4-bit/8-bit quantization",
            "GPU-accelerated performance"
        ]
    })

def gen_uuid(filename) -> str:
    extension = filename.split(".")[-1]
    return str(uuid.uuid4()).replace("-", "") + "." + extension

@app.get("/api/finetune/detect")
async def detect_hardware_page(request: Request) -> JSONResponse:
    global settings_cache
    settings_cache.clear()  # Clear the cache to ensure fresh detection
    return JSONResponse({
        "app_name": app_name,
        "message": "Ready to detect hardware"
    })

@app.post("/api/finetune/detect", response_class=JSONResponse)
async def detect_hardware(request: Request) -> JSONResponse:
    global settings_cache
    try:
        form = await request.json()
        task = TaskFormData(task=form["task"])
        task = task.task
        settings_builder.task = task
        model_requirements, hardware_profile, model_recommendation, possible_options = hardware_detector.run(task)
        settings_builder.compute_profile = model_requirements["profile"]
        settings_cache = {
            "model_requirements": model_requirements,
            "hardware_profile": hardware_profile,
            "model_recommendation": model_recommendation,
            "selected_model": None
        }
        return JSONResponse(
            {
                "status_code": 200,
                "profile": model_requirements["profile"],
                "task": task,
                "gpu_name": hardware_profile.get("gpu_name"),
                "gpu_total_memory_gb": hardware_profile.get("gpu_total_memory_gb"),
                "ram_total_gb": hardware_profile.get("ram_total_gb"),
                "available_diskspace_gb": hardware_profile.get("available_diskspace_gb"),
                "cpu_cores": hardware_profile.get("cpu_cores"),
                "model_recommendation": model_recommendation,
                "possible_options": possible_options,

            }
        )
    except RuntimeError as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    except ValueError as e:
        print(e)
        raise HTTPException(
            status_code=400,
            detail="Invalid task. Must be one of 'text-generation', 'summarization', or 'question-answering'."
        )
    except Exception as e:
        print("General exception triggered")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="Error detecting hardware. Please try again later."
        )

@app.post("/api/finetune/set_model")
async def set_model(request: Request) -> None:
    global settings_cache
    try:
        form = await request.json()
        selected_model = SelectedModelFormData(selected_model=form["selected_model"])
        settings_cache["selected_model"] = selected_model
        settings_builder.model_name = selected_model.selected_model
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=500,
            detail="Error setting model. Please try again later."
        )

@app.get("/api/finetune/load_settings")
async def load_settings_page(request: Request) -> JSONResponse:
    global settings_cache
    if not settings_cache:
        raise HTTPException(
            status_code=400,
            detail="No hardware detection data available. Please run hardware detection first."
        )
    return JSONResponse({
        "app_name": app_name,
        "default_values": settings_builder.get_settings()
    })

@app.post("/api/finetune/load_settings")
async def load_settings(json_file: UploadFile = File(...), settings: str = Form(...)) -> None:
    print("Loading settings...")
    global settings_builder, datasets_dir
    # Validate file type
    if json_file.content_type != "application/json" and json_file.content_type != "application/x-jsonlines" and not json_file.filename.endswith(('.json', '.jsonl')):
        raise HTTPException(400, "Only JSON and JSONL files accepted")

    json_filename = gen_uuid(json_file.filename)
    file_content = await json_file.read()
    file_path = os.path.join(datasets_dir, json_filename)

    # Check file extension to decide processing method
    if json_file.filename.endswith('.jsonl'):
        # For JSONL files, validate each line is valid JSON
        try:
            # Decode bytes to string and split by lines
            content_str = file_content.decode('utf-8')
            for line in content_str.strip().split('\n'):
                json.loads(line)  # Just to validate each line is valid JSON
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSONL format - contains invalid JSON lines")
    elif json_file.filename.endswith('.json'):
        # For JSON files, validate the entire content is valid JSON
        try:
            json.loads(file_content.decode('utf-8'))
        except json.JSONDecodeError:
            raise HTTPException(400, "Invalid JSON format")
    with open(file_path, "wb") as f:
        f.write(file_content)
    settings_builder.dataset = file_path
    settings_data = json.loads(settings)
    settings_data["dataset"] = file_path
    settings_builder.set_settings(settings_data)

def finetuning_task(llm_tuner) -> None:
    global settings_builder, finetuning_status, model_path, settings_cache
    db_con = sqlite3.connect("database/modelforge.db")
    db_cur = db_con.cursor()
    try:
        model_id = str(uuid.uuid4()).replace("-", "_")
        model_name = settings_builder.model_name
        pipeline_task = settings_builder.task
        compute_specs = settings_builder.compute_profile
        db_cur.execute(
            f'''
                INSERT INTO modelforge_models (model_id, model_name, pipeline_task, compute_specs)
                VALUES ('{model_id}', '{model_name}', '{pipeline_task}', '{compute_specs}');
            '''
        )
        db_con.commit()
        llm_tuner.load_dataset(settings_builder.dataset)
        path = llm_tuner.finetune()
        model_path = os.path.join(os.path.dirname(__file__), path.replace("./", ""))
        db_cur.execute(
            f'''
                UPDATE modelforge_models
                SET model_path = '{model_path}'
                WHERE model_id = '{model_id}';
            '''
        )
        db_con.commit()
    except hf_errors.HFValidationError as e:
        print(f"HFValidationError: {e}")
        finetuning_status["status"] = "error"
        finetuning_status["message"] = "Invalid model or dataset. Please check your inputs."
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        finetuning_status["status"] = "error"
        finetuning_status["message"] = "Database error occurred. Please check your modelforge.db file or open an issue on Github."
    finally:
        settings_cache.clear()
        finetuning_status["status"] = "idle"
        finetuning_status["message"] = ""
        finetuning_status["progress"] = 0
        settings_builder = SettingsBuilder(None, None, None)
        del llm_tuner

@app.get("/api/finetune/status")
async def finetuning_status_page(request: Request) -> JSONResponse:
    global finetuning_status
    return JSONResponse({
        "status": finetuning_status["status"],
        "progress": finetuning_status["progress"],
        "message": finetuning_status["message"]
    })

@app.get("/api/finetune/start")
async def start_finetuning_page(request: Request, background_task: BackgroundTasks) -> JSONResponse:
    global settings_builder, settings_cache, finetuning_status

    print(settings_builder.get_settings())

    if not settings_cache:
        raise HTTPException(
            status_code=400,
            detail="No hardware detection data available. Please run hardware detection first."
        )
    if finetuning_status["status"] != "idle":
        print(finetuning_status)
        raise HTTPException(
            status_code=400,
            detail="A finetuning is already in progress. Please wait until it completes."
        )
    finetuning_status["status"] = "initializing"
    finetuning_status["message"] = "Starting finetuning process..."
    if settings_builder.task == "text-generation":
        llm_tuner = CausalLLMFinetuner(
            model_name=settings_builder.model_name,
            compute_specs=settings_builder.compute_profile,
            pipeline_task="text-generation"
        )
    elif settings_builder.task == "summarization":
        llm_tuner = Seq2SeqFinetuner(
            model_name=settings_builder.model_name,
            compute_specs=settings_builder.compute_profile,
            pipeline_task="summarization"
        )
    elif settings_builder.task == "extractive-question-answering":
        llm_tuner = QuestionAnsweringTuner(
            model_name=settings_builder.model_name,
            compute_specs=settings_builder.compute_profile,
            pipeline_task="question-answering"
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Invalid task. Must be one of 'text-generation', 'summarization', or 'question-answering'."
        )

    llm_tuner.set_settings(**settings_builder.get_settings())

    background_task.add_task(finetuning_task, llm_tuner)
    finetuning_status["status"] = "running"
    finetuning_status["message"] = "Finetuning process started."
    return JSONResponse({
        "app_name": app_name,
        "message": "Finetuning process started.",
    })

@app.post("/api/playground/new")
async def new_playground(request: Request) -> None:
    global model_path
    form = await request.json()
    model_path = form["model_path"]

    base_path = os.path.join(os.path.dirname(__file__), "utilities")
    chat_script = os.path.join(base_path, "chat_playground.py")
    command = f"start cmd /K python {chat_script} --model_path {model_path}"
    os.system(command)

@app.get("/api/playground/model_path")
async def get_model_path(request: Request) -> JSONResponse:
    global model_path
    return JSONResponse({
        "model_path": model_path
    })

@app.get("/api/models/all")
async def get_all_models(request: Request) -> JSONResponse:
    global db_cursor
    try:
        db_cursor.execute("SELECT * FROM modelforge_models")
        models = db_cursor.fetchall()
        model_list = []
        for model in models:
            model_list.append({
                "model_id": model[0],
                "model_name": model[1],
                "model_path": model[2],
                "pipeline_task": model[3],
                "compute_specs": model[4]
            })
        return JSONResponse({
            "models": model_list
        })
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error occurred. Please check your modelforge.db file or open an issue on Github.")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)