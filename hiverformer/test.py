from hiverformer.process_instructions import get_instruction
import pickle
from pathlib import Path
import librosa
import matplotlib
import torch
import numpy as np
import random
import tap
import os
import cv2
import torch.nn.functional as F
from amsolver.environment import Environment
from rlbench.backend.observation import Observation
from amsolver.backend.utils import task_file_to_task_class
from amsolver.action_modes import ArmActionMode, ActionMode
from amsolver.observation_config import ObservationConfig
from pyrep.const import RenderMode
from vlm.scripts.cliport_test import CliportAgent
from num2words import num2words
from hiverformer.process_instructions import get_language_feat
from hiverformer.network import Hiveformer
from hiverformer.utils import (
    RLBenchEnv,
    load_episodes,
    get_max_episode_length,
    Actioner,
    load_instructions,
)
from vlm.scripts.train_hiverformer import Arguments as TrainArguments

from typing import List, Dict, Optional, Tuple, Union, Any, Sequence
from rlbench.task_environment import TaskEnvironment
from rlbench.backend.exceptions import InvalidActionError
from pyrep.errors import IKError, ConfigurationPathError
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

class Arguments(tap.Tap):
    data_folder: str = "/home/liuchang/DATA/rlbench_data/test"
    setd: str = "seen"
    need_test_numbers: int = 100
    gpu: int = 6
    seed: int = 2

    # task: list = ['pick', 'stack', 'shape_sorter', 'wipe', 'pour', 'drawer', 'door_complex', 'drop']
    task = ['pick']
    instructions: Dict = {'pick':[], 'stack':[], 'shape_sorter':[], 'wipe':[], 'pour':[], 'drawer':[], 'door_complex':[], 'drop':[]}
    img_size: int = [128,128]
    cameras: list = ['left_shoulder','right_shoulder','wrist']
    max_tries: int = 10
    max_episodes: int = 200
    checkpoint: Path = "/home/liuchang/projects/VLMbench/VLMbench/xp/hiveformer/version5_pick_key/model.epoch=4700-value=0.pth"
    
    # model
    depth: int = 4
    dim_feedforward: int = 64
    hidden_dim: int = 64
    instr_size: int = 512
    mask_obs_prob: float = 0.0
    num_layers: int = 1
    num_words: int = 75
    num_tasks: int = 24
    mode: str = 'waypoint'
    max_episode_length: int = 100

class Recorder(object):
    def __init__(self) -> None:
        cam_placeholder = Dummy('cam_cinematic_placeholder')
        self.cam = VisionSensor.create([640, 360])
        self.cam.set_pose(cam_placeholder.get_pose())
        self.cam.set_parent(cam_placeholder)
        self._snaps = []
        self._fps=30

    def take_snap(self):
        self._snaps.append(
            (self.cam.capture_rgb() * 255.).astype(np.uint8))
    
    def save(self, path):
        print('Converting to video ...')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        video = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'MJPG'), self._fps,
                tuple(self.cam.get_resolution()))
        for image in self._snaps:
            video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video.release()
        self._snaps = []

    def del_snap(self):
        self._snaps = []

class Mover:
    def __init__(self, task: TaskEnvironment, disabled: bool = False, max_tries: int = 1):
        self._task = task
        self._last_action: Optional[np.ndarray] = None
        self._step_id = 0
        self._max_tries = max_tries
        self._disabled = disabled

    def __call__(self, action: np.ndarray):
        if self._disabled:
            return self._task.step(action)

        target = action.copy()
        if self._last_action is not None:
            action[7] = self._last_action[7].copy()

        images = []
        try_id = 0
        obs = None
        terminate = None
        reward = 0

        for try_id in range(self._max_tries):
            obs, reward, terminate, other_obs = self._task.step(action)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

            pos = obs.gripper_pose[:3]
            rot = obs.gripper_pose[3:7]
            gripper = obs.gripper_open
            dist_pos = np.sqrt(np.square(target[:3] - pos).sum())
            dist_rot = np.sqrt(np.square(target[3:7] - rot).sum())
            criteria = (dist_pos < 5e-2,)

            if all(criteria) or reward == 1:
                break

            print(
                f"Too far away (pos: {dist_pos:.3f}, rot: {dist_rot:.3f}, step: {self._step_id})... Retrying..."
            )

        # we execute the gripper action after re-tries
        action = target
        if (
            not reward
            and self._last_action is not None
            and action[7] != self._last_action[7]
        ):
            obs, reward, terminate, other_obs = self._task.step(action)
            if other_obs == []:
                other_obs = [obs]
            for o in other_obs:
                images.append(
                    {
                        k.split("_")[0]: getattr(o, k)
                        for k in o.__dict__.keys()
                        if "_rgb" in k and getattr(o, k) is not None
                    }
                )

        if try_id == self._max_tries:
            print(f"Failure after {self._max_tries} tries")

        self._step_id += 1
        self._last_action = action.copy()

        return obs, reward, terminate, images

# 计算视图中 gripper 的位置
def obs_to_attn(obs, camera: str) -> Tuple[int, int]:
    extrinsics_44 = torch.from_numpy(obs.misc[f"{camera}_camera_extrinsics"]).float()
    extrinsics_44 = torch.linalg.inv(extrinsics_44)
    intrinsics_33 = torch.from_numpy(obs.misc[f"{camera}_camera_intrinsics"]).float()
    intrinsics_34 = F.pad(intrinsics_33, (0, 1, 0, 0))
    gripper_pos_3 = torch.from_numpy(obs.gripper_pose[:3]).float()
    gripper_pos_41 = F.pad(gripper_pos_3, (0, 1), value=1).unsqueeze(1)
    points_cam_41 = extrinsics_44 @ gripper_pos_41

    proj_31 = intrinsics_34 @ points_cam_41
    proj_3 = proj_31.float().squeeze(1)
    u = int((proj_3[0] / proj_3[2]).round())
    v = int((proj_3[1] / proj_3[2]).round())

    return u, v

# class hiverformerAgent:
def get_output_from_obs(obs:Observation, instruction, args):
    device = torch.device(args.gpu)
    gripper = torch.tensor(np.append(obs.gripper_pose,obs.gripper_open))
    rgb = torch.cat([torch.Tensor(obs.left_shoulder_rgb).unsqueeze(0), torch.Tensor(obs.right_shoulder_rgb).unsqueeze(0), torch.Tensor(obs.wrist_rgb).unsqueeze(0)])
    pcd = torch.cat([torch.Tensor(obs.left_shoulder_point_cloud).unsqueeze(0), torch.Tensor(obs.right_shoulder_point_cloud).unsqueeze(0), torch.Tensor(obs.wrist_point_cloud).unsqueeze(0)])


    rgb = rgb.unsqueeze(0)  
    rgb = rgb.permute(0,1,4,2,3)  # 1, N, C, H, W
    pcd = pcd.unsqueeze(0)
    pcd = pcd.permute(0,1,4,2,3)
    gripper = gripper.unsqueeze(0)
    
    attns = torch.Tensor([])
    for cam in args.cameras:
        u, v = obs_to_attn(obs, cam)
        attn = torch.zeros((1, 1, 1, 128, 128))
        if not (u < 0 or u > 127 or v < 0 or v > 127):
            attn[0, 0, 0, v, u] = 1
        attns = torch.cat([attns, attn], 1)
    rgb = torch.cat([rgb, attns], 2)
    lang = []
    lang.append(instruction)
    lang_feat = get_language_feat(lang, "clip", args.num_words, device)

    return rgb, pcd, gripper, lang_feat


def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        if t_name == task_name:
            episode_list.append(path.parent)
    return episode_list

if __name__=="__main__":
    
    args = Arguments().parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    img_size=args.img_size
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size

    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False

    obs_config.right_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.left_shoulder_camera.render_mode = RenderMode.OPENGL
    obs_config.overhead_camera.render_mode = RenderMode.OPENGL
    obs_config.wrist_camera.render_mode = RenderMode.OPENGL
    obs_config.front_camera.render_mode = RenderMode.OPENGL

    if args.task == 'drop': #
        task_files = ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size']
    elif args.task == 'pick': #
        task_files = ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size']
    elif args.task == 'stack': 
        task_files = ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size']
    elif args.task == 'shape_sorter':
        task_files = ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape']
    elif args.task == 'wipe':
        task_files = ['wipe_table_shape', 'wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction']
    elif args.task == 'pour':
        task_files = ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size']
    elif args.task == 'drawer':
        task_files = ['open_drawer']
    elif args.task == 'door':
        task_files = ['open_door']
    elif args.task == 'door_complex':
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

    action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK)
    env = Environment(action_mode, obs_config=obs_config, headless=True)

    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    data_folder = Path(os.path.join(args.data_folder, args.setd))
    device = torch.device(args.gpu)

    agent = Hiveformer(
        depth=args.depth,
        dim_feedforward=args.dim_feedforward,
        hidden_dim=args.hidden_dim,
        instr_size=args.instr_size,
        mask_obs_prob=args.mask_obs_prob,
        max_episode_length=args.max_episode_length,
        num_words=args.num_words,
        num_layers=args.num_layers,
        num_tasks=args.num_tasks,
    )
    agent.to(device)

    model_dict = torch.load(args.checkpoint, map_location="cpu")
    agent.load_state_dict(model_dict["weight"])

    # 不同的任务
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        task = env.get_task(task_to_train)
        success_rate = 0.0
        with torch.no_grad():
            for num, e in enumerate(e_path):
                task_base = str(e/"task_base.ttm")
                waypoint_sets = str(e/"waypoint_sets.ttm")
                config = str(e/"configs.pkl")
                descriptions, obs = task.load_config(task_base, waypoint_sets, config)
                waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
                high_descriptions = descriptions[0]

                images = []
                rgbs = torch.Tensor([]).to(device)
                pcds = torch.Tensor([]).to(device)
                grippers = torch.Tensor([]).to(device)

                images.append(
                    {cam: getattr(obs, f"{cam}_rgb") for cam in args.cameras}
                )
                move = Mover(task, max_tries=args.max_tries)
                reward = None
                
                if high_descriptions[-1]!=".":
                    high_descriptions+="."

                lang = high_descriptions
                step_list = CliportAgent.generate_action_list(waypoints_info, None)
                for i, sub_step in enumerate(step_list):
                    lang += f" Step {num2words(i)}."
                    lang += sub_step[1]
    
                for step_id in range(args.max_episodes):
                    # fetch the current observation, and predict one action
                    rgb, pcd, gripper, lang_feat = get_output_from_obs(obs, lang, args)
                    
                    rgb = rgb.float().to(device)
                    pcd = pcd.float().to(device)
                    gripper = gripper.float().to(device)

                    rgbs = torch.cat([rgbs, rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds, pcd.unsqueeze(1)], dim=1)
                    grippers = torch.cat([grippers, gripper.unsqueeze(1)], dim=1)
                    padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool()

                    output: Dict[str, Any] = {"action": None, "attention": {}}
                    pred = agent(
                        rgbs,    
                        pcds,
                        padding_mask,
                        lang_feat,
                        gripper,
                    )
                    output["action"] = agent.compute_action(pred)  # type: ignore
                    output["attention"] = pred["attention"]

                    action = output["action"]

                    if action is None:
                        break

                    # update the observation based on the predicted action
                    try:
                        action_np = action[-1].detach().cpu().numpy()

                        obs, reward, terminate, step_images = move(action_np)

                        images += step_images

                        if reward == 1:
                            success_rate += 1 / len(e_path)
                            break

                        if terminate:
                            print("The episode has terminated!")

                    except (IKError, ConfigurationPathError, InvalidActionError) as e:
                        print(step_id, " ", action[0][2])
                        break

                print("SR: %.2f" % (success_rate * 100))

    
    # for task in tasks:
    #     languages = get_instruction(task, env)
    #     instructions[task] = languages

    # env.shutdown()
    # output = Path("instructions.pkl")
    # with open(output, "wb") as f:
    #     pickle.dump(instructions, f)

    