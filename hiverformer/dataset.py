import itertools
import random
from typing import (
    Union,
    Optional,
    Tuple,
    List,
    Dict,
    Callable,
    TypeVar,
    Generic,
)
from pickle import UnpicklingError
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
from torch.nn import functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
import einops
from hiverformer.utils import Instructions, Sample, Camera


T = TypeVar("T")
U = TypeVar("U")


class Cache(Generic[T, U]):
    def __init__(self, size: int, loader: Callable[[T], U]):
        self._size = size
        self._loader = loader
        self._keys: List[T] = []
        self._cache: Dict[T, U] = {}

    def __call__(self, args: T) -> U:
        if args in self._cache:
            index = self._keys.index(args)
            del self._keys[index]
            self._keys.append(args)
            return self._cache[args]

        # print(args, len(self._keys), self._size)
        value = self._loader(args)

        if len(self._keys) == self._size and self._keys != []:
            key = self._keys[0]
            del self._cache[key]
            del self._keys[0]

        if len(self._keys) < self._size:
            self._keys.append(args)
            self._cache[args] = value

        return value


def data_transform(scales, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Expect tensors as T, N, C, H, W
    """
    keys = list(kwargs.keys())

    if len(keys) == 0:
        raise RuntimeError("No args")

    # Continuous range of scales
    sc = np.random.uniform(*scales)

    t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
    kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
    resized_size = [int(raw_h * sc), int(raw_w * sc)]

    # Resize based on randomly sampled scale
    kwargs = {
        n: transforms_f.resize(
            arg,
            resized_size,
            transforms.InterpolationMode.NEAREST
            # if "pc" in n
            # else transforms.InterpolationMode.BILINEAR,
        )
        for n, arg in kwargs.items()
    }

    # Adding padding if crop size is smaller than the resized size
    if raw_h > resized_size[0] or raw_w > resized_size[1]:
        right_pad, bottom_pad = max(raw_h - resized_size[1], 0), max(
            raw_w - resized_size[0], 0
        )
        kwargs = {
            n: transforms_f.pad(
                arg,
                padding=[0, 0, right_pad, bottom_pad],
                padding_mode="reflect",
            )
            for n, arg in kwargs.items()
        }

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(
        kwargs[keys[0]], output_size=(raw_h, raw_w)
    )

    kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

    kwargs = {
        n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
        for n, arg in kwargs.items()
    }

    return kwargs


def loader(file: Path) -> Optional[np.ndarray]:
    try:
        return np.load(file, allow_pickle=True)
    except UnpicklingError as e:
        print(f"Can't load {file}: {e}")
    return None


class DataTransform(object):
    def __init__(self, scales):
        self.scales = scales

    def __call__(self, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Except tensors as T, N, C, H, W
        """
        keys = list(kwargs.keys())

        if len(keys) == 0:
            raise RuntimeError("No args")

        # Continuous range of scales
        sc = np.random.uniform(*self.scales)

        t, n, c, raw_h, raw_w = kwargs[keys[0]].shape
        kwargs = {n: arg.flatten(0, 1) for n, arg in kwargs.items()}
        resized_size = [int(raw_h * sc), int(raw_w * sc)]

        # Resize based on randomly sampled scale
        kwargs = {
            n: transforms_f.resize(
                arg,
                resized_size,
                transforms.InterpolationMode.NEAREST
                # if "pc" in n
                # else transforms.InterpolationMode.BILINEAR,
            )
            for n, arg in kwargs.items()
        }

        # Adding padding if crop size is smaller than the resized size
        if raw_h > resized_size[0] or raw_w > resized_size[1]:
            right_pad, bottom_pad = max(raw_h - resized_size[1], 0), max(
                raw_w - resized_size[0], 0
            )
            kwargs = {
                n: transforms_f.pad(
                    arg,
                    padding=[0, 0, right_pad, bottom_pad],
                    padding_mode="reflect",
                )
                for n, arg in kwargs.items()
            }

        # Random Cropping
        i, j, h, w = transforms.RandomCrop.get_params(
            kwargs[keys[0]], output_size=(raw_h, raw_w)
        )

        kwargs = {n: transforms_f.crop(arg, i, j, h, w) for n, arg in kwargs.items()}

        kwargs = {
            n: einops.rearrange(arg, "(t n) c h w -> t n c h w", t=t)
            for n, arg in kwargs.items()
        }

        return kwargs

class My_Dataset(data.Dataset):
    """
    RLBench dataset, 10 tasks
    """

    def __init__(
        self,
        root: Union[Path, str, List[Path], List[str]],
        taskvar: List[Tuple[str, str]],
        # instructions: Instructions,
        max_episode_length: int,
        cache_size: int,
        # max_episodes_per_taskvar: int,
        num_iters: Optional[int] = None,
        cameras: Tuple[Camera, ...] = ("wrist", "left_shoulder", "right_shoulder"),
        training: bool = True,
    ):
        self._cache = Cache(cache_size, loader)
        self._cameras = cameras
        self._max_episode_length = max_episode_length
        # self._max_episodes_per_taskvar = max_episodes_per_taskvar
        self._num_iters = num_iters
        self._training = training
        self._taskvar = taskvar
        if isinstance(root, (Path, str)):
            root = [Path(root)]
        self._root: List[Path] = [Path(r).expanduser() for r in root]

        # # We keep only useful instructions to save mem
        # self._instructions: Instructions = defaultdict(dict)
        # for task, var in taskvar:
        #     self._instructions[task][var] = instructions[task][var]

        self._transform = DataTransform((0.75, 1.25))

        self._data_dirs = []
        self._episodes = []
        self._num_episodes = 0
        for root, (task, var) in itertools.product(self._root, taskvar):
            data_dir = root/ task / var
            if not data_dir.is_dir():
                raise ValueError(f"Can't find dataset folder {data_dir}")
            episodes = [(task, var, ep) for ep in data_dir.glob("*.npy")]
            # episodes = episodes[: self._max_episodes_per_taskvar]
            num_episodes = len(episodes)
            if num_episodes == 0:
                raise ValueError(f"Can't find episodes at folder {data_dir}")
            self._data_dirs.append(data_dir)
            self._episodes += episodes
            self._num_episodes += num_episodes

        print("Num ep.", self._num_episodes)

    def __getitem__(self, episode_id: int) -> Optional[Sample]:
        episode_id %= self._num_episodes
        task, variation, file = self._episodes[episode_id] # file就是之前存的 state_dict
        episode = self._cache(file)

        if episode is None:
            return None

        frame_ids = episode[0]
        num_ind = len(frame_ids)
        pad_len = max(0, self._max_episode_length - num_ind)

        rgbs: torch.Tensor = torch.stack([episode[1][i].squeeze(0) for i in frame_ids])     # torch.Size([4, 3, 3, 128, 128]) T, N, C, H, W
        pcds: torch.Tensor = torch.stack([episode[2][i].squeeze(0) for i in frame_ids])
        action: torch.Tensor = torch.stack([episode[3][i].squeeze(0) for i in frame_ids])
        gripper: torch.Tensor = torch.stack([episode[4][i].squeeze(0) for i in frame_ids])
        lang: str = episode[6][0]

        padding_mask = torch.tensor([True] * num_ind + [False] * pad_len)
        # padding
        img_pad_vec = [0, 0] * rgbs.dim()
        img_pad_vec[-1] = pad_len
        rgbs = F.pad(rgbs, img_pad_vec, value=0)
        pcds = F.pad(pcds, img_pad_vec, value=0)

        action_pad_vec = [0, 0] * action.dim()
        action_pad_vec[-1] = pad_len
        action = F.pad(action, action_pad_vec, value=0)
        gripper = F.pad(gripper, action_pad_vec, value=0)

        if rgbs.shape[-1] != 128 or rgbs.shape[-2] != 128:
            raise ValueError(f"{rgbs.shape} {self._episodes[episode_id]}")

        cameras = list(episode[5][0].keys())
        assert all(c in cameras for c in self._cameras)

        attns = torch.Tensor([])
        for i in frame_ids:
            attn_cams = torch.Tensor([])
            for cam in self._cameras:
                u, v = episode[5][i][cam]
                attn = torch.zeros((1, 1, 128, 128))
                if not (u < 0 or u > 127 or v < 0 or v > 127):
                    attn[0, 0, v, u] = 1
                attn_cams = torch.cat([attn_cams, attn])
            attns = torch.cat([attns, attn_cams.unsqueeze(0)])
        pad_vec = [0] * (2 * attns.dim())
        pad_vec[-1] = pad_len
        attns = F.pad(attns, pad_vec)
        rgbs = torch.cat([rgbs, attns], 2)

        if self._training:
            modals = self._transform(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

        return {
            # instruction
            "language": str(lang), 
            # img and pcd
            "rgbs": rgbs,
            "pcds": pcds,
            # state
            "action": action,
            "gripper": gripper,
            # others
            "padding_mask": padding_mask,
            "task": str(task)
        }

    def __len__(self):
        if self._num_iters is not None:
            return self._num_iters
        return self._num_episodes