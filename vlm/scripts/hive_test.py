import random
from typing import Tuple, Optional
from copy import deepcopy
from pathlib import Path
import torch
import numpy as np
import tap
from filelock import FileLock
from hiverformer.network import Hiveformer
from hiverformer.utils import (
    RLBenchEnv,
    load_episodes,
    get_max_episode_length,
    Actioner,
    load_instructions,
)
from train_hiverformer import Arguments as TrainArguments


class Arguments(tap.Tap):
    checkpoint: Optional[Path] = None
    seed: int = 2
    save_img: bool = False
    device: str = "cuda"
    num_episodes: int = 100
    headless: bool = True
    offset: int = 0
    name: str = "autol"
    max_tries: int = 10
    output: Path = Path(__file__).parent / "records.txt"
    xp: Path = Path(__file__).parent / "xp"
    test_xp: Path = Path(__file__).parent / "test-xp"
    data_dir: Path = Path(__file__).parent / "demos"
    record_actions: bool = False
    replay_actions: Optional[Path] = None
    ground_truth_rotation: bool = False
    ground_truth_position: bool = False
    ground_truth_gripper: bool = False
    tasks: Optional[Tuple[str, ...]] = None
    instructions: Optional[Path] = None
    arch: Optional[str] = None
    variations: Tuple[int, ...] = (0,)
    attention: bool = False  # saving attention maps
    
    # model
    depth: Optional[int] = None
    dim_feedforward: Optional[int] = None
    hidden_dim: Optional[int] = None
    instr_size: Optional[int] = None
    mask_obs_prob: float = 0.0
    num_layers: Optional[int] = None
    max_episodes: int = 200


def get_log_dir(args: Arguments) -> Path:
    log_dir = args.xp / args.name
    task = list(args.tasks)[0].split("_")[0]
    version = 0
    while (log_dir / f"version{version}_{task}_{args.mode}").is_dir():
        version += 1
    return log_dir / f"version{version}_{task}_{args.mode}"


def copy_args(checkpoint: Path, args: Arguments) -> Arguments:
    args = deepcopy(args)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Copying args from", checkpoint)

    # Update args accordingly:
    hparams = checkpoint.parent / "hparams.json"
    print(hparams, hparams.is_file())
    if hparams.is_file():
        print("Loading args from checkpoint")
        train_args = TrainArguments()
        train_args.load(str(hparams))
        for key in args.class_variables:
            v = getattr(args, key)
            if v is None and key in train_args.class_variables:
                setattr(args, key, getattr(train_args, key))
                print("Copying", key, ":", getattr(args, key))

    return args


def load_model(checkpoint: Path, args: Arguments) -> Hiveformer:
    args = copy_args(checkpoint, args)
    device = torch.device(args.device)

    if not checkpoint.is_file():
        files = list((args.xp / checkpoint).rglob("mtl_*.pth"))
        assert files != [], args.checkpoint
        files = sorted(files, key=lambda x: x.stat().st_mtime)
        checkpoint = files[0]

    print("Loading model from...", checkpoint, flush=True)
    if args.tasks is None:
        raise RuntimeError("Can't find tasks")

    if (
        args.depth is None
        or args.dim_feedforward is None
        or args.hidden_dim is None
        or args.instr_size is None
        or args.mask_obs_prob is None
        or args.num_layers is None
    ):
        raise ValueError("Please provide the missing parameters")

    max_episode_length = get_max_episode_length(args.tasks, args.variations)

    model = Hiveformer(
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        instr_size=args.instr_size,
        mask_obs_prob=args.mask_obs_prob,
        max_episode_length=max_episode_length,
        num_layers=args.num_layers,
    ).to(device)

    if hasattr(model, "film_gen") and model.film_gen is not None:
        model.film_gen.build(device)

    model.eval()

    return model

if __name__ == "__main__":
    args = Arguments().parse_args()
    print(args)
    log_dir = get_log_dir(args)
    log_dir.mkdir(exist_ok=True, parents=True)
    print("log dir", log_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load model and args
    checkpoint = args.checkpoint
    args = copy_args(checkpoint, args)
    if checkpoint is None:
        raise RuntimeError()
    model = load_model(checkpoint, args)

    if args.tasks is None or args.gripper_pose is None or args.taskvar_token is None:
        raise ValueError()

    # load RLBench environment
    env = RLBenchEnv(
        data_path=args.data_dir,
        apply_rgb=True,
        apply_pc=True,
        headless=args.headless,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
    )

    device = torch.device(args.device)
    instruction = load_instructions(args.instructions)

    if instruction is None:
        raise NotImplementedError()

    actioner = Actioner(model=model, instructions=instruction)

    for task_str in args.tasks:
        for variation in args.variations:
            success_rate = env.evaluate(
                task_str,
                max_episodes=args.max_episodes,
                variation=variation,
                num_demos=args.num_episodes,
                demos=None,
                offset=args.offset,
                actioner=actioner,
                log_dir=log_dir / task_str if args.save_img else None,
                max_tries=args.max_tries,
                save_attn=args.attention,
            )

            print("Testing Success Rate {}: {:.04f}".format(task_str, success_rate))

            with FileLock(str(args.output.parent / f"{args.output.name}.lock")):
                with open(args.output, "a") as output_id:
                    output_id.write(
                        f"{task_str}-{variation}, {checkpoint}, seed={args.seed}, {success_rate}, {log_dir}\n"
                    )