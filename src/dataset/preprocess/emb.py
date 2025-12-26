import os
import gc
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

def load_model_and_tokenizer():
    """
    Load BAAI/bge-m3 model and tokenizer from Hugging Face cache.
    Use FP16/FP32 without quantization.
    """
    model_name = "BAAI/bge-m3"
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    print(f"Loading model and tokenizer: {model_name}")

    # Load from cache if available, otherwise download.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, local_files_only=True)
        print("Found tokenizer in cache")
    except:
        print("Downloading tokenizer from Hugging Face...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)

    # Load model in FP16 when CUDA is available.
    try:
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_safetensors=True
        )
        print("Found model in cache")
    except:
        print("Downloading model from Hugging Face...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            use_safetensors=True
        )

    # Optional accelerations via env vars.
    if os.getenv("EMB_BETTERTRANSFORMER", "0") == "1":
        try:
            if hasattr(model, "to_bettertransformer"):
                model = model.to_bettertransformer()
                print("⚡ Enabled BetterTransformer fastpath")
        except Exception as e:
            print(f"BetterTransformer enable failed: {e}")

    if os.getenv("COMPILE_MODEL", "0") == "1":
        try:
            model = torch.compile(
                model,
                mode=os.getenv("COMPILE_MODE", "reduce-overhead"),
                fullgraph=False,
                dynamic=True,
            )
            print("🧩 Enabled torch.compile")
        except Exception as e:
            print(f"torch.compile failed: {e}")

    print("Successfully loaded BGE-M3 model and tokenizer.")
    return model, tokenizer

class CachingModule(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.cached_inputs = []
    def forward(self, *args, **kwargs):
        self.cached_inputs.append((args, kwargs))
        return self.module(*args, **kwargs)
    def clear_cache(self):
        self.cached_inputs = []

def cls_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Use the first token ([CLS]) as the sentence embedding."""
    return last_hidden_states[:, 0]

def text2embedding_with_multihead(texts, model, tokenizer, device, instruction_template, target_layers=None):
    """Encode texts and capture multi-head embeddings for BGE-M3/XLM-RoBERTa."""
    if not texts:
        base_model = model.module if hasattr(model, 'module') else model
        num_heads = base_model.config.num_attention_heads
        head_dim = base_model.config.hidden_size // num_heads
        multihead_embeds = {}
        if target_layers:
            for layer_idx in target_layers:
                 multihead_embeds[layer_idx] = torch.empty(0, num_heads, head_dim)
        return torch.empty(0, base_model.config.hidden_size), multihead_embeds

    base_model = model.module if hasattr(model, 'module') else model

    # Only XLM-RoBERTa architecture (BGE-M3) is supported.
    if not (hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer')):
        raise AttributeError(f"Only XLM-RoBERTa architecture (BGE-M3) is supported, got: {type(base_model)}")

    model_layers = base_model.encoder.layer

    if target_layers is None:
        target_layers = {len(model_layers) - 1}

    # BGE-M3 uses raw text without a special instruction template.
    texts_to_encode = [str(text) for text in texts]
    # Runtime-configurable parameters.
    try:
        total_mem = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0
    except Exception:
        total_mem = 0
    default_bs = 192 if total_mem >= 40 * 1024**3 else 64
    batch_size = int(os.getenv("EMB_BATCH", str(default_bs)))
    max_length = int(os.getenv("EMB_MAX_LENGTH", "128"))
    empty_cache_every = int(os.getenv("EMB_EMPTY_CACHE_EVERY", "0"))  # 0 disables manual clearing

    # Precision: auto/bf16/fp16.
    prec = os.getenv("EMB_PRECISION", "auto").lower()
    use_bf16 = prec == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Compute optimizations.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    all_standard_embeddings = []
    all_multihead_embeddings = {layer_idx: [] for layer_idx in target_layers}

    # Wrap target layers with CachingModule.
    original_modules = {}
    for layer_idx in target_layers:
        target_module = model_layers[layer_idx].attention.output.dense
        original_modules[layer_idx] = target_module
        model_layers[layer_idx].attention.output.dense = CachingModule(target_module)

    try:
        for i in range(0, len(texts_to_encode), batch_size):
            batch_texts = texts_to_encode[i:i+batch_size]
            inputs = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.pin_memory() for k, v in inputs.items()}
            inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

            with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype):
                # Clear per-layer cache.
                for layer_idx in target_layers:
                    model_layers[layer_idx].attention.output.dense.clear_cache()

                outputs = model(**inputs)
                embeddings = cls_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_standard_embeddings.append(embeddings.cpu())

                batch_size_current = len(batch_texts)

                for layer_idx in target_layers:
                    cached_inputs = model_layers[layer_idx].attention.output.dense.cached_inputs

                    if cached_inputs:
                        attn_output = cached_inputs[-1][0][0]
                        if attn_output.dim() == 3:
                            cls_token_attn = attn_output[:, 0, :]

                            head_dim = base_model.config.hidden_size // base_model.config.num_attention_heads
                            num_heads = base_model.config.num_attention_heads
                            multihead_emb = cls_token_attn.view(batch_size_current, num_heads, head_dim)
                            all_multihead_embeddings[layer_idx].append(multihead_emb.cpu())

                del outputs, inputs
                if empty_cache_every > 0 and ((i // batch_size) % empty_cache_every == 0):
                    gc.collect()
                    torch.cuda.empty_cache()

    finally:
        # Restore original modules.
        for layer_idx in target_layers:
            model_layers[layer_idx].attention.output.dense = original_modules[layer_idx]

    standard_embeddings = torch.cat(all_standard_embeddings, dim=0)
    multihead_embeddings = {layer_idx: torch.cat(embs, dim=0) for layer_idx, embs in all_multihead_embeddings.items() if embs}
    return standard_embeddings, multihead_embeddings

def bge_text2embedding(model, tokenizer, device, texts, instruction_template=None):
    """Encode text with BGE-M3."""
    if not texts:
        return torch.zeros((0, 1024))

    # BGE-M3 uses raw text without a special instruction template.
    texts_to_encode = [str(text) for text in texts]

    try:
        total_mem = torch.cuda.get_device_properties(device).total_memory if torch.cuda.is_available() else 0
    except Exception:
        total_mem = 0
    default_bs = 256 if total_mem >= 40 * 1024**3 else 96
    batch_size = int(os.getenv("EMB_BATCH", str(default_bs)))
    max_length = int(os.getenv("EMB_MAX_LENGTH", "128"))
    empty_cache_every = int(os.getenv("EMB_EMPTY_CACHE_EVERY", "0"))

    prec = os.getenv("EMB_PRECISION", "auto").lower()
    use_bf16 = prec == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    all_embeddings = []
    for i in range(0, len(texts_to_encode), batch_size):
        batch_texts = texts_to_encode[i:i+batch_size]
        inputs = tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.pin_memory() for k, v in inputs.items()}
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=amp_dtype):
            outputs = model(**inputs)
            embeddings = cls_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())

        del outputs, inputs
        if empty_cache_every > 0 and ((i // batch_size) % empty_cache_every == 0):
            gc.collect()
            torch.cuda.empty_cache()

    if all_embeddings:
        return torch.cat(all_embeddings, dim=0)
    return torch.zeros((0, 1024))
