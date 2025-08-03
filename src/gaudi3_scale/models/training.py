"""Training configuration models."""

from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ModelType(str, Enum):
    """Supported model architectures."""
    LLAMA = "llama"
    BERT = "bert"
    GPT = "gpt"
    T5 = "t5"
    STABLE_DIFFUSION = "stable_diffusion"
    MIXTRAL = "mixtral"


class PrecisionType(str, Enum):
    """Training precision options."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    BF16_MIXED = "bf16-mixed"
    INT8 = "int8"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAMW = "adamw"
    FUSED_ADAMW = "fused_adamw"
    SGD = "sgd"
    ADAFACTOR = "adafactor"


class ModelConfig(BaseModel):
    """Model configuration for training."""
    
    model_type: ModelType = Field(..., description="Model architecture type")
    model_name: str = Field(..., description="Model name or path")
    model_size: str = Field(..., description="Model size (e.g., '7B', '70B')")
    
    # Model parameters
    vocab_size: Optional[int] = Field(None, description="Vocabulary size")
    hidden_size: Optional[int] = Field(None, description="Hidden dimension size")
    num_layers: Optional[int] = Field(None, description="Number of layers")
    num_heads: Optional[int] = Field(None, description="Number of attention heads")
    sequence_length: Optional[int] = Field(None, description="Maximum sequence length")
    
    # Model loading
    pretrained_path: Optional[str] = Field(None, description="Path to pretrained model")
    checkpoint_path: Optional[str] = Field(None, description="Path to checkpoint")
    trust_remote_code: bool = Field(False, description="Trust remote code execution")
    
    # Memory optimization
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")
    use_cache: bool = Field(False, description="Use KV cache during training")
    low_cpu_mem_usage: bool = Field(True, description="Use low CPU memory loading")
    
    @validator('model_size')
    def validate_model_size(cls, v):
        valid_sizes = ['7B', '13B', '30B', '65B', '70B', '175B']
        if v not in valid_sizes:
            raise ValueError(f'Model size must be one of {valid_sizes}')
        return v


class DatasetConfig(BaseModel):
    """Dataset configuration for training."""
    
    dataset_name: str = Field(..., description="Dataset name or path")
    dataset_type: str = Field("huggingface", description="Dataset type")
    
    # Data processing
    tokenizer_name: Optional[str] = Field(None, description="Tokenizer name")
    max_length: int = Field(2048, description="Maximum sequence length")
    padding: str = Field("max_length", description="Padding strategy")
    truncation: bool = Field(True, description="Enable truncation")
    
    # Data loading
    streaming: bool = Field(False, description="Enable streaming dataset")
    num_proc: int = Field(8, description="Number of preprocessing processes")
    cache_dir: Optional[str] = Field(None, description="Cache directory")
    
    # Data splits
    train_split: str = Field("train", description="Training split name")
    validation_split: Optional[str] = Field("validation", description="Validation split")
    test_split: Optional[str] = Field("test", description="Test split name")
    
    # Data filtering
    min_length: int = Field(10, description="Minimum sequence length")
    max_length_filter: Optional[int] = Field(None, description="Maximum length filter")
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size for the dataset."""
        return getattr(self, '_batch_size', 32)


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    
    # Basic training parameters
    batch_size: int = Field(32, description="Training batch size")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    max_epochs: int = Field(3, description="Maximum training epochs")
    max_steps: Optional[int] = Field(None, description="Maximum training steps")
    
    # Learning rate and optimization
    learning_rate: float = Field(6e-4, description="Learning rate")
    weight_decay: float = Field(0.1, description="Weight decay")
    warmup_steps: int = Field(100, description="Warmup steps")
    lr_scheduler_type: str = Field("cosine", description="Learning rate scheduler")
    
    # Precision and compilation
    precision: PrecisionType = Field(PrecisionType.BF16_MIXED, description="Training precision")
    optimizer_type: OptimizerType = Field(OptimizerType.FUSED_ADAMW, description="Optimizer type")
    
    # Gaudi-specific optimizations
    use_habana_dataloader: bool = Field(True, description="Use Habana dataloader")
    use_lazy_mode: bool = Field(True, description="Enable lazy mode compilation")
    use_hpu_graphs: bool = Field(True, description="Enable HPU graphs")
    enable_async_grad_copy: bool = Field(True, description="Enable async gradient copy")
    
    # Distributed training
    distributed_backend: str = Field("hccl", description="Distributed backend")
    find_unused_parameters: bool = Field(False, description="Find unused parameters")
    
    # Gradient management
    gradient_clip_val: float = Field(1.0, description="Gradient clipping value")
    gradient_clip_algorithm: str = Field("norm", description="Gradient clipping algorithm")
    
    # Checkpointing
    save_strategy: str = Field("steps", description="Save strategy")
    save_steps: int = Field(500, description="Save every N steps")
    save_total_limit: int = Field(3, description="Maximum saved checkpoints")
    
    # Evaluation
    eval_strategy: str = Field("steps", description="Evaluation strategy")
    eval_steps: int = Field(500, description="Evaluate every N steps")
    per_device_eval_batch_size: int = Field(16, description="Evaluation batch size")
    
    # Logging and monitoring
    logging_steps: int = Field(10, description="Log every N steps")
    report_to: List[str] = Field(default_factory=lambda: ["wandb"], description="Reporting tools")
    wandb_project: Optional[str] = Field(None, description="Weights & Biases project")
    
    # Output and storage
    output_dir: str = Field("./output", description="Output directory")
    logging_dir: Optional[str] = Field(None, description="Logging directory")
    
    @validator('batch_size')
    def validate_batch_size(cls, v):
        if v < 1 or v > 512:
            raise ValueError('Batch size must be between 1 and 512')
        return v
    
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0 or v > 1:
            raise ValueError('Learning rate must be between 0 and 1')
        return v
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size including accumulation."""
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def optimizer_config(self) -> Dict[str, any]:
        """Get optimizer configuration."""
        base_config = {
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "eps": 1e-8
        }
        
        if self.optimizer_type == OptimizerType.FUSED_ADAMW:
            base_config.update({
                "betas": (0.9, 0.95),
                "use_habana": True
            })
        elif self.optimizer_type == OptimizerType.ADAMW:
            base_config.update({
                "betas": (0.9, 0.999)
            })
        
        return base_config
    
    def to_lightning_config(self) -> Dict[str, any]:
        """Convert to PyTorch Lightning trainer configuration."""
        return {
            "max_epochs": self.max_epochs,
            "max_steps": self.max_steps,
            "precision": self.precision.value,
            "gradient_clip_val": self.gradient_clip_val,
            "gradient_clip_algorithm": self.gradient_clip_algorithm,
            "accumulate_grad_batches": self.gradient_accumulation_steps,
            "log_every_n_steps": self.logging_steps,
            "val_check_interval": self.eval_steps if self.eval_strategy == "steps" else 1.0,
            "enable_checkpointing": True,
            "enable_progress_bar": True,
            "enable_model_summary": True
        }