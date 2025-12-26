"""
Optimized model loading utilities for distributed training
"""
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_optimal_device_map(world_size, rank, local_rank, max_memory_per_gpu):
    """
    Get optimal device map for distributed training
    """
    if world_size == 1:
        # Single GPU training - use auto device map
        return {
            "device_map": "auto",
            "max_memory": {0: f'{max_memory_per_gpu}GiB'}
        }
    else:
        # Multi-GPU distributed training - load to current GPU only
        return {
            "device_map": {"": local_rank},
            "max_memory": {local_rank: f'{max_memory_per_gpu}GiB'}
        }


def load_model_optimized(model_path, world_size=1, rank=0, local_rank=0, max_memory_per_gpu=23):
    """
    Optimized model loading for distributed training
    
    Args:
        model_path: Path to the model
        world_size: Total number of processes
        rank: Global rank of current process
        local_rank: Local rank on current node
        max_memory_per_gpu: Memory limit per GPU in GB
    """
    
    if rank == 0:
        print(f"Loading model from {model_path}")
        print(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    # Get device map configuration
    device_config = get_optimal_device_map(world_size, rank, local_rank, max_memory_per_gpu)
    
    # Common loading arguments
    loading_kwargs = {
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
        "revision": "main",
        **device_config
    }
    
    if rank == 0:
        print(f"Loading with config: {loading_kwargs}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **loading_kwargs
    )
    
    if rank == 0:
        print(f"Model loaded successfully on device: {model.device}")
    
    return model, tokenizer


def synchronize_model_loading(world_size, rank):
    """
    Synchronize model loading across all processes to avoid memory issues
    """
    if world_size > 1:
        # Create a barrier to ensure all processes load models in sync
        if rank == 0:
            print("Synchronizing model loading across all processes...")
        dist.barrier()
        if rank == 0:
            print("All processes synchronized, continuing...")


def estimate_model_memory(model_name):
    """
    Estimate memory requirements for different models
    """
    memory_estimates = {
        "7b": 14,   # ~14GB for 7B model in fp16
        "8b": 16,   # ~16GB for 8B model in fp16
        "13b": 26,  # ~26GB for 13B model in fp16
    }
    
    for key in memory_estimates:
        if key in model_name.lower():
            return memory_estimates[key]
    
    return 16  # Default estimate


def get_recommended_batch_size(model_name, available_memory_gb):
    """
    Get recommended batch size based on model size and available memory
    """
    model_memory = estimate_model_memory(model_name)
    available_for_batch = available_memory_gb - model_memory - 2  # Reserve 2GB for overhead
    
    if available_for_batch <= 0:
        return 1
    
    # Rough estimate: each sample in batch uses ~0.5-1GB
    recommended_batch_size = max(1, int(available_for_batch // 1))
    
    return min(recommended_batch_size, 8)  # Cap at 8 for stability
