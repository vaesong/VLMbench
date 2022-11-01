import random
from typing import List, Tuple, Dict, Optional, Any, Union
import itertools
import pickle
import os
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
import numpy as np
from tqdm import tqdm, trange
from filelock import FileLock
import tap
from hiverformer.network import Hiveformer
from hiverformer.utils import (
    LossAndMetrics,
    load_instructions,
    RLBenchEnv,
    count_parameters,
    load_episodes,
    get_max_episode_length,
    Actioner,
)
from vlm.scripts.VLDataloader_renjie import VLM_dataset
# from dataset import RLBenchDataset, Sample


class Arguments(tap.Tap):
    accumulate_grad_batches: int = 1
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 10
    # dataset: List[Path]
    device: str = "cuda"
    xp: Path = Path(__file__).parent / "xp"
    valset: Optional[Tuple[Path, ...]] = None
    name: str = "hiveformer"
    arch: str = "mct"
    num_workers: int = 5
    max_tries: int = 10
    max_episodes_per_taskvar: int = 100
    instructions: Optional[Path] = None
    cache_size: int = 100
    seed: int = 2

    # tasks: Tuple[str, ...]
    tasks: str
    variations: Tuple[int, ...] = (0,)


    # Train
    # batch_size: int = 32
    batch_size: int = 8
    lr: float = 0.001
    val_freq: int = 200
    val_batch_size: int = 100
    train_iters: int = 100_000
    jitter: bool = False
    
    # 自己加的
    train_dir: Path = None
    valid_dir: Path = None
    epochs: int = 20
    relative: bool = False
    renew_obs: bool = False
    add_low_lang: bool = True
    maxAction: int = 12
    maxInput: int = 80
    ignoreid: int = -100

    img_size: list = [360, 360]
    unused_camera_list: list = ['left_shoulder', 'right_shoulder']
    preprocess: bool = False
    use_fail_cases: bool = False
    sample_numbers: int = 0
    workers: int = 2
    gpu: int = 0
    distributed: bool = False
    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1
    num_words: int = 53


def training(
    model: nn.Module,
    optimizer,
    train_loader,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: SummaryWriter,
):
    # iter_loader = iter(train_loader)
    # device = next(model.parameters()).device
    

    # with trange(args.train_iters) as tbar:
    #     for step_id in tbar:
    for iter in range(1, args.epochs+1):
        for batch_step, batch_data in enumerate(train_loader):
            try:
                sample = batch_data
            except StopIteration:
                # iter_loader = iter(train_loader)
                sample = batch_data

            rgbs = sample["rgbs"].to(device) # B 4key_frame 3camera 4channel(rgb+attn) 128 128 except for the end index img
            pcds = sample["pcds"].to(device) # B 4key_frame 3camera 3channel 128 128 except for the end index pcd
            gripper = sample["action"].to(device) # B 4key_frame 8(action_ls[:-1]) except for the end index action
            # outputs = sample["action"].to(device) # B 4key_frame 8(action_ls[1:]) except for the start index action 
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["instr"] # B 53 512
            if instr is not None:
                instr = instr.to(device)

            frame_id = sample["frame_id"] # [0,1,2,3]
            tasks = sample["task"] 

            if iter % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            pred = model(
                rgbs,   # 
                pcds,
                padding_mask,
                instr,
                gripper,
            )

            train_losses = loss_and_metrics.compute_loss(pred, sample)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

            for n, l in train_losses.items():
                writer.add_scalar(f"train-loss/{n}", l, iter)

            writer.add_scalar(f"lr/", args.lr, iter)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                writer.add_scalar(f"train-metrics/{n}", l, iter)

            train_losses["total"].backward()  # type: ignore

            if iter % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if (iter + 1) % args.val_freq == 0:
                if val_loaders is not None:
                    val_metrics = validation_step(
                        iter,
                        val_loaders,
                        model,
                        writer,
                        loss_and_metrics,
                    )
                    model.train()
                else:
                    val_metrics = {}
                checkpointer(val_metrics)

            # tbar.set_postfix(l=float(train_losses["total"]))


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    version = int(os.environ.get("SLURM_JOBID", 0))
    while (log_dir / f"version{version}").is_dir():
        version += 1
    return log_dir / f"version{version}"


class CheckpointCallback:
    def __init__(
        self,
        name: str,
        log_dir: Path,
        state_dict: Any,
        minimizing: bool = True,
        checkpoint_period: int = 200,
    ):
        self._name = name
        self._minimizing = minimizing
        self._best = float("inf") if minimizing else -float("inf")
        self._log_dir = log_dir
        self._checkpoint_period = checkpoint_period
        self._step = 0
        self._state_dict = state_dict

    def __call__(self, metrics: Dict[str, torch.Tensor]):
        self._step += 1
        if self._step % self._checkpoint_period != 0:
            return

        value = int(metrics.get(self._name, 0))
        dest = self._log_dir / f"model.step={self._step}-value={value}.pth"
        torch.save(self._state_dict, dest)

        if (self._minimizing and self._best > value) or (
            not self._minimizing and self._best < value
        ):
            best = self._log_dir / "best.pth"
            best.unlink(missing_ok=True)
            best.symlink_to(dest.resolve())
            self._best = value


@torch.no_grad()
def validation_step(
    step_id: int,
    val_loaders: List[DataLoader],
    model,
    writer,
    loss_and_metrics,
    val_iters: int = 5,
):
    values = {}
    device = next(model.parameters()).device 
    model.eval()

    for val_id, val_loader in enumerate(val_loaders):
        for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            rgbs = sample["rgbs"].to(device)
            pcds = sample["pcds"].to(device)
            gripper = sample["gripper"].to(device)
            outputs = sample["action"].to(device)
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["instr"]
            if instr is not None:
                instr = instr.to(device)

            frame_id = sample["frame_id"]
            tasks = sample["task"]

            pred = model(
                rgbs,
                pcds,
                padding_mask,
                instr,
                gripper,
            )

            losses: Dict[str, torch.Tensor] = loss_and_metrics.compute_loss(pred, sample)
            losses["total"] = torch.stack(list(losses.values())).sum()

            for n, l in losses.items():
                key = f"val-loss-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in values:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

            writer.add_scalar(f"lr/", args.lr, step_id + i)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                key = f"val-metrics-{val_id}/{n}"
                writer.add_scalar(key, l, step_id + i)
                if key not in metrics:
                    values[key] = torch.Tensor([]).to(device)
                values[key] = torch.cat([values[key], l.unsqueeze(0)])

        key = f"val-loss-{val_id}/total"
        print(f"Validation Loss {val_id}: {values[key].mean():.05f}")
        key = f"val-metrics-{val_id}/position"
        print(f"Validation Position {val_id}: {values[key].mean():.05f}")

    return values


# def collate_fn(batch: List[Dict]):
#     keys = batch[0].keys()
#     return {
#         key: default_collate([item[key] for item in batch])
#         if batch[0][key] is not None
#         else None
#         for key in keys
#     }


def get_train_loader(args: Arguments) -> DataLoader:
    dataset = VLM_dataset(
            args.train_dir, 
            'train', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            args=args
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(  
            dataset, 
            batch_size=args.batch_size, 
            shuffle=(sampler is None),
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=sampler, 
            drop_last=True,
            persistent_workers=True) #,persistent_workers=True
    
    return loader


def get_val_loaders(args: Arguments) -> Optional[List[DataLoader]]:
    
    dataset = VLM_dataset(
            args.valid_dir,
            'valid', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            args=args
    )

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(  
            dataset, 
            batch_size=args.batch_size, 
            shuffle=(sampler is None),
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=sampler, 
            drop_last=True,
            persistent_workers=True) #,persistent_workers=True
    
    return loader


def get_model(args: Arguments) -> Tuple[optim.Optimizer, Hiveformer]:
    device = torch.device(args.device)

    # max_eps_dict = load_episodes()["max_episode_length"]

    # max_episode_length = get_max_episode_length(args.tasks, args.variations)
    max_episode_length = 100
    model = Hiveformer(
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        instr_size=args.instr_size,
        mask_obs_prob=args.mask_obs_prob,
        max_episode_length=max_episode_length,
        num_words=args.num_words,
        num_layers=args.num_layers,
    )
    model=model.to(device)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": 0.0, "lr": args.lr},
        {"params": [], "weight_decay": 5e-4, "lr": args.lr},
    ]
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    for name, param in model.named_parameters():
        if any(nd in name for nd in no_decay):
            optimizer_grouped_parameters[0]["params"].append(param)  # type: ignore
        else:
            optimizer_grouped_parameters[1]["params"].append(param)  # type: ignore
    optimizer: optim.Optimizer = optim.AdamW(optimizer_grouped_parameters)

    if args.checkpoint is not None:
        model_dict = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(model_dict["weight"])
        optimizer.load_state_dict(model_dict["optimizer"])

    print("Number of parameters:")
    model_params = count_parameters(model)
    print("- model", model_params)
    print("Total", model_params)

    return optimizer, model


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)

    optimizer, model = get_model(args)
    
    loss_and_metrics = LossAndMetrics(args.tasks)

    # training episode
    model_dict = {
        "weight": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics/position",
        log_dir,
        model_dict,
        minimizing=False,
        checkpoint_period=args.checkpoint_period,
    )
    model.train()
    print(model)
    val_loaders = get_val_loaders(args)
    # val_loaders = None

    if args.train_iters > 0:
        train_loader = get_train_loader(args)
        training(
            model,
            optimizer,
            train_loader,
            val_loaders,
            checkpointer,
            loss_and_metrics,
            args,
            writer,
        )

    # if val_loaders is not None:
    #     val_metrics = validation_step(
    #         args.train_iters,
    #         val_loaders,
    #         model,
    #         writer,
    #         loss_and_metrics,
    #         val_iters=-1,
    #     )

    # last checkpoint
    checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    torch.save(model_dict, checkpoint)

    # evaluation
    model.eval()

    env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=args.headless,
    )

    instruction = load_instructions(args.instructions)
    actioner = Actioner(model=model.model, instructions=instruction)
    max_eps_dict = load_episodes()["max_episode_length"]
    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str,
                actioner=actioner,
                max_episodes=max_eps_dict.get(task_str, 6),
                variation=variation,
                num_demos=500,
                demos=None,
                log_dir=log_dir,
                max_tries=args.max_tries,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(args.output.parent / f"{args.output.name}.lock"):
                with open(args.output, "a") as oid:
                    oid.write(
                        f"{task_str}-{variation}, na, seed={args.seed}, {success_rate}\n"
                    )
