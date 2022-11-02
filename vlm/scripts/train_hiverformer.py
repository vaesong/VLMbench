import random
from typing import List, Tuple, Dict, Optional, Any, Union
import itertools
import pickle
import os
import time
from pathlib import Path
import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from tools.TimeAverageMeter import AverageMeter, sec_to_str
import numpy as np
from tqdm import tqdm, trange
from filelock import FileLock
import tap
from hiverformer.process_instructions import get_language_feat
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
# distributed
import torch.multiprocessing as mp
import torch.distributed as dist


class Arguments(tap.Tap):
    accumulate_grad_batches: int = 1
    cameras: Tuple[str, ...] = ("wrist", "left_shoulder", "right_shoulder")
    checkpoint: Optional[Path] = None
    checkpoint_period: int = 50
    
    xp: Path = "/home/liuchang/projects/VLMbench/VLMbench/xp"
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
    tasks: Tuple[str, ...]
    train_tasks: List = []
    variations: Tuple[int, ...] = (0,)

    # Train
    batch_size: int = 16
    lr: float = 0.001
    val_freq: int = 50     # 200
    val_batch_size: int = 16
    jitter: bool = False
    
    # 自己加的
    train_dir: Path = None
    valid_dir: Path = None
    epochs: int = 500
    relative: bool = False
    renew_obs: bool = False
    add_low_lang: bool = True
    maxAction: int = 12
    maxInput: int = 80
    ignoreid: int = -100

    img_size: list = [128, 128]
    unused_camera_list: list = ['left_shoulder', 'right_shoulder']
    preprocess: bool = False
    use_fail_cases: bool = False
    sample_numbers: int = 0
    workers: int = 8
    gpu: int = 0

    # distributed
    world_size: int = 1
    rank: int = 0
    dist_url: str = 'tcp://127.0.0.1:23462'
    dist_backend: str = 'nccl'
    gpu_start: int = 0
    gpu_list: list = [6,7]
    gpu_number: int = 0
    ngpus_per_node: int = 0
    distributed: bool = False

    # tests
    headless: bool = False
    output: Path = Path(__file__).parent / "records.txt"

    # model
    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 768
    mask_obs_prob: float = 0.0
    num_layers: int = 1
    num_words: int = 80


def training(
    model: nn.Module,
    optimizer,
    train_loader,
    train_sampler,
    val_loaders,
    checkpointer,
    loss_and_metrics,
    args: Arguments,
    writer: SummaryWriter,
):
    # iter_loader = iter(train_loader)
    device = next(model.parameters()).device

    timer = {"batch_time":AverageMeter('Time', ':6.3f')}

    for epoch in range(1, args.epochs+1):
        # 在分布式模式下，需要在每个 epoch 开始时调用set_epoch()方法，然后再创建 DataLoader 迭代器，以使shuffle 操作能够在多个 epoch 中正常工作。 
        # 否则，dataloader迭代器产生的数据将始终使用相同的顺序，使得每个epoch在每个GPU上分割的数据集都是一样的
        if args.distributed:
                train_sampler.set_epoch(epoch)

        batch_time = timer["batch_time"]
        end = time.time()

        for batch_step, batch_data in enumerate(train_loader):
            sample = batch_data
            
            rgbs = sample["rgbs"].float().to(device) # B 4key_frame 3camera 4channel(rgb+attn) 128 128 except for the end index img
            rgbs = rgbs.permute(0,1,2,5,3,4)

            attens = sample["attns"].float().to(device)
            rgbs = torch.cat([rgbs, attens], 3)

            pcds = sample["pcds"].float().to(device) # B 4key_frame 3camera 3channel 128 128 except for the end index pcd
            pcds = pcds.permute(0,1,2,5,3,4)

            gripper = sample["action"].float().to(device) # B 4key_frame 8(action_ls[:-1]) except for the end index action
            # outputs = sample["action"].to(device) # B 4key_frame 8(action_ls[1:]) except for the start index action 
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["language"] # B 53 512
            lang_feat = get_language_feat(language = instr).float().to(device)  # B 80 768

            # frame_id = sample["frame_id"] # [0,1,2,3]
            # tasks = sample["task"] 

            if batch_step % args.accumulate_grad_batches == 0:
                optimizer.zero_grad()

            pred = model(
                rgbs,   # 
                pcds,
                padding_mask,
                lang_feat,
                gripper,
            )

            train_losses = loss_and_metrics.compute_loss(pred, sample)
            train_losses["total"] = sum(list(train_losses.values()))  # type: ignore

            for n, l in train_losses.items():
                writer.add_scalar(f"train-loss/{n}", l, batch_step)

            writer.add_scalar(f"lr/", args.lr, batch_step)

            metrics = loss_and_metrics.compute_metrics(pred, sample)
            for n, l in metrics.items():
                writer.add_scalar(f"train-metrics/{n}", l, batch_step)

            train_losses["total"].backward()  # type: ignore

            if batch_step % args.accumulate_grad_batches == args.accumulate_grad_batches - 1:
                optimizer.step()

            if (batch_step + 1) % args.val_freq == 0:
                if val_loaders is not None:
                    val_metrics = validation_step(
                        batch_step,
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
            # 计算时间
            batch_time.update(time.time() - end)
            end = time.time()
            time_per_epoch = batch_time.avg * len(train_loader)
            epochs_left = args.epochs - epoch - 1
            batches_left = len(train_loader) - batch_step - 1

            time_elapsed = sec_to_str(batch_time.sum)
            time_left = sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
            time_estimate = sec_to_str(args.epochs * time_per_epoch)
            # 打印一些东西
            tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                    'Elapsed: {}  ' \
                    'ETA: {} / {}  '\
                    'total loss: {}  '.format(epoch, args.epochs, batch_step+1, len(train_loader), 
            time_elapsed, time_left, time_estimate, train_losses["total"])

            print(tmp_str)


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
        # if self._step % self._checkpoint_period != 0:
        #     return
        step = self._step * self._checkpoint_period

        value = int(metrics.get(self._name, 0))
        dest = self._log_dir / f"model.step={step}-value={value}.pth"
        torch.save(self._state_dict, dest)

        if (self._minimizing and self._best > value) or (
            not self._minimizing and self._best < value
        ):
            best = self._log_dir / "best.pth"
            # best.unlink(missing_ok=True)
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

    for val_id in range(1, val_iters):
        for i, sample in enumerate(val_loaders):
    # for val_id, val_loader in enumerate(val_loaders):
    #     for i, sample in enumerate(val_loader):
            if i == val_iters:
                break

            rgbs = sample["rgbs"].float().to(device)
            rgbs = rgbs.permute(0,1,2,5,3,4)

            attens = sample["attns"].float().to(device)
            rgbs = torch.cat([rgbs, attens], 3)

            pcds = sample["pcds"].float().to(device)
            pcds = pcds.permute(0,1,2,5,3,4)

            gripper = sample["action"].float().to(device)
            outputs = sample["action"].float().to(device)
            padding_mask = sample["padding_mask"].to(device)

            instr = sample["language"]
            lang_feat = get_language_feat(language = instr).float().to(device)  # B 80 768

            pred = model(
                rgbs,
                pcds,
                padding_mask,
                lang_feat,
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

def get_train_loader(args: Arguments) -> DataLoader:
    dataset = VLM_dataset(
            args.train_dir, 
            'train', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            train_tasks=args.train_tasks,
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
    
    return loader, sampler


def get_val_loaders(args: Arguments) -> Optional[List[DataLoader]]:
    
    dataset = VLM_dataset(
            args.valid_dir,
            'valid', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            train_tasks=args.train_tasks,
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
    device = torch.device('cuda', args.gpu)

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

    optimizer = torch.optim.Adam(model.parameters(),args.lr)

    print("Number of parameters:")
    model_params = count_parameters(model)
    print("- model", model_params)
    print("Total", model_params)

    return optimizer, model

def main(gpu, ngpus_per_node, args):

    # 首先处理分布式的问题
    # 这里的 gpu 如果是分布式，就是第几个进程（0，1，2，...），如果不是分布式，就是指定的 gpu
    args.gpu = args.gpu_list[gpu]
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
                args.rank = int(os.environ["RANK"])
        
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # 处理 checkpoint 存放的路径
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("Logging:", log_dir)
    args.save(str(log_dir / "hparams.json"))
    writer = SummaryWriter(log_dir=log_dir)

    # 生成种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 确定训练的任务
    if 'all' in list(args.tasks):
        args.train_tasks =  [
            'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
            'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
            'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
            'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
            'stack_cubes_color', 'stack_cubes_size',
            'stack_cubes_relative', 'stack_cubes_shape',
            'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
            'open_drawer',
            'open_door_complex'
            ]
    else:
        args.train_tasks = list(args.tasks)

    # 构建模型和优化器
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
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
        model.cuda(args.gpu)

    model.train()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(),args.lr)

    print("Number of parameters:")
    model_params = count_parameters(model)
    print("- model", model_params)
    print("Total", model_params)

    # 设置损失函数
    loss_and_metrics = LossAndMetrics(args)

    # 保存模型参数，以及构建需要的 checkpoint 实例
    model_dict = {
        "weight": model.module.state_dict() if args.distributed else model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    checkpointer = CheckpointCallback(
        "val-metrics/position",
        log_dir,
        model_dict,
        minimizing=False,
        checkpoint_period=args.checkpoint_period,
    )

    # 构建训练集和 train_loader
    train_dataset = VLM_dataset(
            args.train_dir, 
            'train', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            train_tasks=args.train_tasks,
            args=args
    )

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(  
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=(train_sampler is None),
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=train_sampler, 
            drop_last=True,
            persistent_workers=True) #,persistent_workers=True
    
    # 构建验证集和 val_loader
    val_dataset = VLM_dataset(
            args.valid_dir,
            'valid', 
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            train_tasks=args.train_tasks,
            args=args
    )

    if args.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        val_sampler = None

    val_loader = torch.utils.data.DataLoader(  
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=(val_sampler is None),
            num_workers=args.workers, 
            pin_memory=True, 
            sampler=val_sampler, 
            drop_last=True,
            persistent_workers=True) #,persistent_workers=True
    
    # 开始训练
    training(
        model,
        optimizer,
        train_loader,
        train_sampler,
        val_loader,
        checkpointer,
        loss_and_metrics,
        args,
        writer,
    )


if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)

    #如果没有设置数量，就自动检测，得到总共该节点有几块 gpu 可用
    ngpus_per_node = torch.cuda.device_count() if (args.gpu_list).size==0 else (args.gpu_list).size
    args.ngpus_per_node = ngpus_per_node
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main(args.gpu, ngpus_per_node, args)



    # log_dir = get_log_dir(args)
    # log_dir.mkdir(exist_ok=True, parents=True)
    # print("Logging:", log_dir)
    # args.save(str(log_dir / "hparams.json"))
    # writer = SummaryWriter(log_dir=log_dir)

    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)

    # if 'all' in list(args.tasks):
    #     args.train_tasks =  [
    #         'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
    #         'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
    #         'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
    #         'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
    #         'stack_cubes_color', 'stack_cubes_size',
    #         'stack_cubes_relative', 'stack_cubes_shape',
    #         'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
    #         'open_drawer',
    #         'open_door_complex'
    #         ]
    # else:
    #     args.train_tasks = list(args.tasks)

    # device = torch.device('cuda', args.gpu)

    # optimizer, model = get_model(args)
    
    # loss_and_metrics = LossAndMetrics(args)

    # # training episode
    # model_dict = {
    #     "weight": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    # }
    # checkpointer = CheckpointCallback(
    #     "val-metrics/position",
    #     log_dir,
    #     model_dict,
    #     minimizing=False,
    #     checkpoint_period=args.checkpoint_period,
    # )

    # model.train()
    # print(model)
    # val_loader = get_val_loaders(args)
    # # val_loaders = None

    # train_loader, train_sampler = get_train_loader(args)
    # training(
    #     model,
    #     optimizer,
    #     train_loader,
    #     train_sampler,
    #     val_loader,
    #     checkpointer,
    #     loss_and_metrics,
    #     args,
    #     writer,
    # )

    # # last checkpoint
    # checkpoint = log_dir / f"mtl_{args.seed}_{args.lr}.pth"
    # torch.save(model_dict, checkpoint)

    # # evaluation
    # # model.eval()

    # env = RLBenchEnv(
    #     data_path="",
    #     apply_rgb=True,
    #     apply_pc=True,
    #     apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    #     headless=args.headless,
    # )

    # instruction = load_instructions(args.instructions)
    # actioner = Actioner(model=model.model, instructions=instruction)
    # max_eps_dict = load_episodes()["max_episode_length"]
    # for task_str in args.tasks:
    #     for variation in args.variations:
    #         success_rate = env.evaluate(
    #             task_str,
    #             actioner=actioner,
    #             max_episodes=max_eps_dict.get(task_str, 6),
    #             variation=variation,
    #             num_demos=500,
    #             demos=None,
    #             log_dir=log_dir,
    #             max_tries=args.max_tries,
    #         )

    #         print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

    #         with FileLock(args.output.parent / f"{args.output.name}.lock"):
    #             with open(args.output, "a") as oid:
    #                 oid.write(
    #                     f"{task_str}-{variation}, na, seed={args.seed}, {success_rate}\n"
    #                 )
