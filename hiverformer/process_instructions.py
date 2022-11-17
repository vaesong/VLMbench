import re
import os
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
from transformers import logging
logging.set_verbosity_error()

from amsolver.environment import Environment
from amsolver.backend.utils import task_file_to_task_class
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
from pyrep.const import RenderMode
from vlm.scripts.cliport_test import CliportAgent
from num2words import num2words

TextEncoder = Literal["bert", "clip"]


def load_model(encoder: TextEncoder) -> transformers.PreTrainedModel:
    if encoder == "bert":
        model = transformers.BertModel.from_pretrained("bert-base-uncased")
    elif encoder == "clip":
        model = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", ignore_mismatched_sizes=True)
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

def get_language_feat(language, encoder, num_words, device):
    model = load_model(encoder).to(device)
    tokenizer = load_tokenizer(encoder)
    tokenizer.model_max_length = num_words

    tokens = tokenizer(language, padding="max_length")["input_ids"]
    tokens = torch.tensor(tokens).to(device)

    with torch.no_grad():
        pred = model(tokens).last_hidden_state
    
    return pred

def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        if t_name == task_name:
            episode_list.append(path.parent)
    return episode_list

def get_instruction(task, env:Environment):
    data_folder = str("/home/liuchang/DATA/rlbench_data/test")
    setd = str("seen")
    need_test_numbers = 100
    num_words = 75
    device = torch.device("cuda:6")

    if task == 'drop': #
        task_files = ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size']
    elif task == 'pick': #
        task_files = ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size']
    elif task == 'stack': 
        task_files = ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size']
    elif task == 'shape_sorter':
        task_files = ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape']
    elif task == 'wipe':
        task_files = ['wipe_table_shape', 'wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction']
    elif task == 'pour':
        task_files = ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size']
    elif task == 'drawer':
        task_files = ['open_drawer']
    elif task == 'door':
        task_files = ['open_door']
    elif task == 'door_complex':
        task_files = ['open_door_complex']
    else:
        task_files = [
                        'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
                        'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
                        'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
                        'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
                        'stack_cubes_color', 'stack_cubes_size', 'stack_cubes_relative', 'stack_cubes_shape',
                        'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
                        'open_drawer',
                        'open_door_complex'
                        ]

    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    data_folder = Path(os.path.join(data_folder, setd))

    instructions = []
    language = []
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        task = env.get_task(task_to_train)
        for num, e in enumerate(e_path):
            if num >= need_test_numbers:
                break
            
            lang = str()
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config = str(e/"configs.pkl")
            descriptions, obs = task.load_config(task_base, waypoint_sets, config)
            waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            high_descriptions = descriptions[0]
            if high_descriptions[-1]!=".":
                high_descriptions+="."

            step_list = CliportAgent.generate_action_list(waypoints_info, None)
            for i, sub_step in enumerate(step_list):
                lang += high_descriptions+f" Step {num2words(i)}."
                lang += sub_step[1]

            language.append(lang)
    
    lang_feat = get_language_feat(language, "clip", num_words, device).float().to(device)
    # instructions.append(lang_feat)
    for i in range(0, lang_feat.shape[0]):
        instructions.append(lang_feat[i].squeeze(0))
    return instructions