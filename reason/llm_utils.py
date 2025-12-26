import os
import time
import openai
from vllm import LLM, SamplingParams
from openai import OpenAI
from functools import partial
from prompts import icl_user_prompt, icl_ass_prompt


def _resolve_local_model_path(model_name: str):
    """
    Map friendly model names to local cache paths and extra engine kwargs.
    Returns: (resolved_model, extra_kwargs)
    """
    name_l = (model_name or "").lower()
    qwen_aliases = {
        "qwen3-30b-a3b",
        "qwen3-30b",
    }
    if name_l in qwen_aliases:
        local_path = "/home/ubuntu/.cache/modelscope/hub/models/Qwen/Qwen3-30B-A3B"
        extra = {
            "trust_remote_code": True,
            "quantization": "awq",
        }
        return local_path, extra
    return model_name, {}


def llm_init(model_name, tensor_parallel_size=1, max_model_len=8192*2, max_tokens=4000, seed=0, temperature=0, frequency_penalty=0, gpu_memory_utilization=0.8, max_num_seqs=128):
    # Qwen API models.
    if "qwen3-max" in model_name.lower() or "qwen-max" in model_name.lower():
        # Qwen API (OpenAI-compatible endpoint).
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("Set DASHSCOPE_API_KEY to use the Qwen API")
        
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        # Qwen API model name.
        actual_model = "qwen3-max"
        llm = partial(client.chat.completions.create, 
                     model=actual_model, 
                     temperature=temperature, 
                     max_tokens=max_tokens)
        return llm, None
    
    elif "gpt" in model_name:
        # OpenAI API.
        client = OpenAI()
        llm = partial(client.chat.completions.create, 
                     model=model_name, 
                     seed=seed, 
                     temperature=temperature, 
                     max_tokens=max_tokens)
        # Return llm and client for usage stats.
        return llm, client
    
    else:
        resolved_model, extra_kwargs = _resolve_local_model_path(model_name)

        # Build LLM kwargs.
        llm_kwargs = {
            "model": resolved_model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,  # Control GPU memory usage.
            "max_num_seqs": max_num_seqs,  # Limit parallel sequences to reduce KV cache.
        }
        # Merge extra kwargs (trust_remote_code / quantization).
        llm_kwargs.update(extra_kwargs)
        # Only set max_model_len when specified.
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        
        client = LLM(**llm_kwargs)
        sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                                         frequency_penalty=frequency_penalty)
        llm = partial(client.chat, sampling_params=sampling_params, use_tqdm=False)
        return llm, None


def get_outputs(outputs, model_name):
    # API models (OpenAI/Qwen) share the same output format.
    if "gpt" in model_name or "qwen" in model_name.lower():
        return outputs.choices[0].message.content, outputs.usage
    else:
        # Local vLLM model.
        return outputs[0].outputs[0].text, None


def llm_inf(llm, prompts, mode, model_name):
    res = []
    usage_stats = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    if 'sys' in mode:
        conversation = [{"role": "system", "content": prompts['sys_query']}]

    if 'icl' in mode:
        conversation.append({"role": "user", "content": icl_user_prompt})
        conversation.append({"role": "assistant", "content": icl_ass_prompt})

    if 'sys' in mode:
        conversation.append({"role": "user", "content": prompts['user_query']})
        outputs, usage = get_outputs(llm(messages=conversation), model_name)
        if usage:
            usage_stats['prompt_tokens'] += usage.prompt_tokens
            usage_stats['completion_tokens'] += usage.completion_tokens
            usage_stats['total_tokens'] += usage.total_tokens
        res.append(outputs)

    if 'sys_cot' in mode:
        if 'clear' in mode:
            conversation = []
        conversation.append({"role": "assistant", "content": outputs})
        conversation.append({"role": "user", "content": prompts['cot_query']})
        outputs, usage = get_outputs(llm(messages=conversation), model_name)
        if usage:
            usage_stats['prompt_tokens'] += usage.prompt_tokens
            usage_stats['completion_tokens'] += usage.completion_tokens
            usage_stats['total_tokens'] += usage.total_tokens
        res.append(outputs)
    elif "dc" in mode:
        if 'ans:' not in res[0].lower() or "ans: not available" in res[0].lower() or "ans: no information available" in res[0].lower():
            conversation.append({"role": "user", "content": prompts['cot_query']})
            outputs, usage = get_outputs(llm(messages=conversation), model_name)
            if usage:
                usage_stats['prompt_tokens'] += usage.prompt_tokens
                usage_stats['completion_tokens'] += usage.completion_tokens
                usage_stats['total_tokens'] += usage.total_tokens
            res[0] = outputs
        res.append("")
    else:
        res.append("")

    return res, usage_stats


def llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries):
    retries = 0
    while retries < max_retries:
        try:
            return llm_inf(llm, each_qa, llm_mode, model_name)
        except openai.RateLimitError as e:
            wait_time = (2 ** retries) * 5  # Exponential backoff
            print(f"Rate limit error encountered. Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    raise Exception("Max retries exceeded. Please check your rate limits or try again later.")


def llm_inf_all(llm, each_qa, llm_mode, model_name, max_retries=5):
    # API models (OpenAI/Qwen) use retry logic.
    if 'gpt' in model_name or 'qwen' in model_name.lower():
        res, usage_stats = llm_inf_with_retry(llm, each_qa, llm_mode, model_name, max_retries)
        return res, usage_stats
    else:
        # Local vLLM model.
        res, usage_stats = llm_inf(llm, each_qa, llm_mode, model_name)
        return res, usage_stats
