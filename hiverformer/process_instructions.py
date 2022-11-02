import re
import json
from pathlib import Path
import itertools
from typing import List, Tuple, Dict, Optional
from typing_extensions import Literal
from collections import defaultdict
import pickle
import tap
import transformers
from tqdm.auto import tqdm
import torch
from torch import nn

TextEncoder = Literal["bert", "clip"]


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(model, transformers.PreTrainedModel):
        raise ValueError(f"Unexpected encoder {encoder}")
    return model

def load_tokenizer(encoder: TextEncoder) -> transformers.PreTrainedTokenizer:
    if encoder == "bert":
        tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        tokenizer = transformers.CLIPTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
    else:
        raise ValueError(f"Unexpected encoder {encoder}")
    if not isinstance(tokenizer, transformers.PreTrainedTokenizer):
        raise ValueError(f"Unexpected encoder {encoder}")
    return tokenizer

def get_language_feat(language):
    device = language.device
    model = load_model("bert")
    tokenizer = load_tokenizer("bert")
    tokenizer.model_max_length = 80

    # tokens = tokenizer(language, padding="max_length")["input_ids"]
    # tokens = torch.tensor(tokens).to(device)
    with torch.no_grad():
        pred = model(language).last_hidden_state
    
    language_features = pred.to(device)
    return language_features


