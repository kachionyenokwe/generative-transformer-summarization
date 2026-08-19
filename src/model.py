import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

def load_base_model_and_tokenizer(model_id="Qwen/Qwen2.5-1.5B-Instruct"):
    """
    Loads tokenizer and 4-bit quantized base model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    return model, tokenizer

def apply_lora_adapter(model, r=16, alpha=32, dropout=0.05):
    """
    Attaches LoRA trainable adapter parameters to target attention projections.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    lora_model = get_peft_model(model, lora_config)
    return lora_model