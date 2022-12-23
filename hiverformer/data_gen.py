import os
import re
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict, Optional, Any, Union
import torch
import cv2
from pathlib import Path
import pickle
from cliport.utils.utils import get_fused_heightmap
from amsolver.observation_config import ObservationConfig
from amsolver.utils import get_stored_demos,get_stored_demos_nodepth
import time
import copy
from scipy.spatial.transform import Rotation as R
from num2words import num2words
pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL
import random
from vlm.scripts.utils import keypoint_discovery,mask_tokens,max_sperate_index
from pytorch_transformers import  BertTokenizer
# add hiverformer
from torch.nn import functional as F
from hiverformer.utils import obs_to_attn, DataTransform
from hiverformer.process_instructions import get_language_feat
import random
import math
import itertools
from typing import Tuple, Dict, List
from pathlib import Path
import json
from tqdm import tqdm
import tap
import torch
import numpy as np
import einops
from rlbench.demo import Demo
from hiverformer.utils import (
    RLBenchEnv,
    keypoint_discovery,
    task_file_to_task_class,
    obs_to_attn,
    transform,
)

class Arguments(tap.Tap):
    cameras: list = ['left_shoulder','right_shoulder','wrist']
    seed: int = 2

    # tasks: Tuple[str, ...]
    tasks: Tuple[str, ...] = ('pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size')
    train_tasks: List = []
    variations: Tuple[int, ...] = (0,)
    
    # 自己加的
    train_dir: Path = "/home/liuchang/DATA/rlbench_data"
    valid_dir: Path = "/home/liuchang/DATA/rlbench_data"
    relative: bool = False
    renew_obs: bool = False
    add_low_lang: bool = True
    output: Path = "/home/liuchang/projects/VLMbench/VLMbench/hiverformer/data_gen"
    mode: str = 'waypoint'

    img_size: list = [128, 128]
    unused_camera_list: list = ['overhead','front']
    preprocess: bool = False
    use_fail_cases: bool = False
    sample_numbers: int = 0
    workers: int = 0
    persistent_workers: bool = False
    gpu: int = 5

    num_words: int = 75

def get_attn_indices_from_demo(
    task_str: str, demo: Demo, cameras: Tuple[str, ...], frames: List,
) -> List[Dict[str, Tuple[int, int]]]:
    return [{cam: obs_to_attn(demo[f], cam) for cam in cameras} for f in frames]

def get_observation(task_str: str, variation: int, episode: int, env: RLBenchEnv):
    demos = env.get_demo(task_str, variation, episode)
    demo = demos[0]
    # 首先获得关键帧
    key_frame = keypoint_discovery(demo)
    # HACK for tower3
    if task_str == "tower3":
        key_frame = [k for i, k in enumerate(key_frame) if i % 6 in set([1, 4])]
    # HACK tower4
    elif task_str == "tower4":
        key_frame = key_frame[6:]
    key_frame.insert(0, 0)

    state_ls = []
    action_ls = []
    for f in key_frame:
        state, action = env.get_obs_action(demo._observations[f])
        state = transform(state)
        state_ls.append(state.unsqueeze(0))
        action_ls.append(action.unsqueeze(0))

    return demo, state_ls, action_ls

class Hive_dataset(Dataset):
    def __init__(self, root, setd, img_size=(256, 256), 
                    unused_camera_list = ['left_shoulder', 'right_shoulder', 'overhead','wrist'], preprocess = True, 
                    use_fail_cases = True, sample_numbers = None, train_tasks = None, random_sample = False, args=None):
        self.root = root
        self.setd = setd
        self.dataset_path = Path(os.path.join(self.root, self.setd))
        self.episode_list = []
        self.variation_list = []
        self.task_list = {}
        self.fail_cases_list = []
        self.read_lists()
        self.cameras = args.cameras
        self.use_fail_cases = use_fail_cases
        self.output = args.output
        self.mode = args.mode
        if train_tasks is None:
            train_tasks =  [
                            'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
                            'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
                            'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
                            'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
                            'stack_cubes_color', 'stack_cubes_size', 'stack_cubes_relative', 'stack_cubes_shape',
                            'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
                            'open_drawer',
                            'open_door_complex'
                            ]
        if train_tasks is not None:
            self.episode_list = []
            for t in train_tasks:
                for n in self.task_list:
                    if t in n:
                        self.episode_list += self.task_list[n]['success']
                        self.fail_cases_list += self.task_list[n]['fail']
        if use_fail_cases:
            self.episode_list += self.fail_cases_list
        #only train selected tasks

        self.sample_numbers = sample_numbers
        self.random_sample = random_sample
        self.img_size = img_size
        self.preprocess = preprocess
        self._transform = DataTransform((0.75, 1.25))

        self.obs_config = ObservationConfig()
        self.obs_config.set_all(True)
        # self.obs_config.set_depth(False)
        self.obs_config.right_shoulder_camera.image_size = self.img_size
        self.obs_config.left_shoulder_camera.image_size = self.img_size
        self.obs_config.overhead_camera.image_size = self.img_size
        self.obs_config.wrist_camera.image_size = self.img_size
        self.obs_config.front_camera.image_size = self.img_size

        self.views = list(set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']) - set(unused_camera_list))

        # self.camera=list(set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front'])^set(unused_camera_list))

        if 'left_shoulder' in unused_camera_list:
            self.obs_config.left_shoulder_camera.set_all(False)
        if 'right_shoulder' in unused_camera_list:
            self.obs_config.right_shoulder_camera.set_all(False)
        if 'overhead' in unused_camera_list:
            self.obs_config.overhead_camera.set_all(False)
        if 'wrist' in unused_camera_list:
            self.obs_config.wrist_camera.set_all(False)
        if 'front' in unused_camera_list:
            self.obs_config.front_camera.set_all(False)

        # self.unused_camera_list = unused_camera_list
        self.relative = False
        self.renew_obs = False
        self.add_low_lang = True
        self.args = args
        if args is not None:
            self.relative = args.relative
            self.renew_obs = args.renew_obs
            self.add_low_lang = args.add_low_lang

    def read_lists(self):
        tasks_list_path = self.dataset_path / '{}_list.pkl'.format(self.setd)
        if not tasks_list_path.is_file():
            self.task_list = {}
            self.variation_list =set()
            for path in self.dataset_path.rglob('low_dim_obs*'):#PosixPath('/home/zp_3c/liuchang/vlmbench/data/train/open_drawer/variation2/episodes/episode0/low_dim_obs.pkl')
                path = path.relative_to(self.dataset_path)#PosixPath('open_drawer/variation2/episodes/episode0/low_dim_obs.pkl')

                episode = (os.path.split(path.parents[0]))[-1]
                list_number = re.findall(r"\d+",episode)
                if int(list_number[0]) < 300:
                    continue

                task_name = str(path.parents[3]) #open_drawer
                if task_name not in self.task_list:
                    self.task_list[task_name]={'success':[], 'fail':[]}
                self.variation_list.add(path.parents[2])#PosixPath('open_drawer/variation2')
                if 'fail_cases' in str(path):
                    self.fail_cases_list.append(path.parent)
                    self.task_list[task_name]['fail'].append(path.parent)
                else:
                    self.episode_list.append(path.parent) #PosixPath('open_drawer/variation2/episodes/episode0')
                    self.task_list[task_name]['success'].append(path.parent)
            self.variation_list = list(self.variation_list)
            with open(tasks_list_path,'wb') as f:
                pickle.dump({'task_list': self.task_list, 
                            'episode_list': self.episode_list,
                            'fail_cases_list': self.fail_cases_list,
                            'variation_list': self.variation_list}, f)
        else:
            with open(tasks_list_path,'rb') as f:
                # {'task_list': {'open_drawer': {...}}, 'episode_list': [PosixPath('open_drawer/variation2/episodes/episode0'), PosixPath('open_drawer/variation7/episodes/episode0'), PosixPath('open_drawer/variation0/episodes/episode0'), PosixPath('open_drawer/variation5/episodes/episode0'), PosixPath('open_drawer/variation1/episodes/episode0'), PosixPath('open_drawer/variation8/episodes/episode0'), PosixPath('open_drawer/variation11/episodes/episode0'), PosixPath('open_drawer/variation4/episodes/episode0'), PosixPath('open_drawer/variation10/episodes/episode0'), ...], 'fail_cases_list': [], 'variation_list': [PosixPath('open_drawer/variation4'), PosixPath('open_drawer/variation7'), PosixPath('open_drawer/variation2'), PosixPath('open_drawer/variation10'), PosixPath('open_drawer/variation1'), PosixPath('open_drawer/variation3'), PosixPath('open_drawer/variation9'), PosixPath('open_drawer/variation8'), PosixPath('open_drawer/variation5'), ...]}
                info_dict = pickle.load(f)
                self.task_list = info_dict['task_list']
                self.episode_list = info_dict['episode_list']
                self.variation_list = info_dict['variation_list']
                self.fail_cases_list = info_dict['fail_cases_list']
    
    def __getitem__(self, index):
        episode = self.episode_list[index]
        self.get_episode(episode)

    def get_episode(self,episode):
        variation_path = episode.parents[1]
        task_name = episode.parents[2]

        low_dim_obs = self.dataset_path/episode/"low_dim_obs.pkl"
        with open(low_dim_obs, 'rb') as f:
            demo_temple = pickle.load(f)
        
        sequence_length = len(demo_temple._observations)
        obs_select_inds = np.arange(sequence_length)
        lang = demo_temple.high_level_instructions[0]
        
        if self.sample_numbers:
            if self.random_sample:
                obs_select_inds = np.sort(np.random.choice(obs_select_inds, self.sample_numbers, replace=False))
            else:
                obs_select_inds = obs_select_inds[0:self.sample_numbers]
        split_by_waypoint = True
        # 根据watpoints切分
        # obs_select_inds：选择出每个waypoint的开始index
        if split_by_waypoint:
            # lang = demo_temple.high_level_instructions[0]
            obs_select_inds = [0]
            previous_waypoint="waypoint0"
            # all_way_points只存了分割点的waypoint
            self.all_waypoints = [previous_waypoint]
            for i, obs in enumerate(demo_temple._observations):
                if obs.current_waypoint_name == previous_waypoint:
                    continue
                else:
                    previous_waypoint = obs.current_waypoint_name
                    self.all_waypoints.append(previous_waypoint)
                    obs_select_inds.append(i)
                    # lang+=str(f" Step {num2words(obs_select_inds.index(i))}:"+obs.low_level_description)

        # episode_number = int(episode.name.replace('episode',''))
        episode_name = episode.name
        variation_number = int(variation_path.name.replace('variation',''))

        # key_frames = obs_select_inds
        if self.mode == 'waypoint':
            key_frames = obs_select_inds
        else:
            key_frames = keypoint_discovery(demo_temple._observations)

        select_frames=[]

        if 0 not in key_frames:
            select_frames.append(0)
        for frame in key_frames:
            if frame not in select_frames:
                select_frames.append(frame)
        # add end obs to max_traj_len
        if (sequence_length-1) not in select_frames:
            select_frames.append(sequence_length-1)

        demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
                                    task_name, self.obs_config , episode_name,selected_frame=select_frames)   

        rgbs = torch.Tensor([])
        pcds = torch.Tensor([])
        states = torch.Tensor([])
        for frame in select_frames:
            rgb = torch.Tensor([])
            pcd = torch.Tensor([])
            ac = torch.Tensor([])

            obs=demos[0]._observations[frame]
            ac = torch.tensor(np.append(obs.gripper_pose,obs.gripper_open))
            rgb = torch.cat([torch.Tensor(obs.left_shoulder_rgb).unsqueeze(0), torch.Tensor(obs.right_shoulder_rgb).unsqueeze(0), torch.Tensor(obs.wrist_rgb).unsqueeze(0)])
            pcd = torch.cat([torch.Tensor(obs.left_shoulder_point_cloud).unsqueeze(0), torch.Tensor(obs.right_shoulder_point_cloud).unsqueeze(0), torch.Tensor(obs.wrist_point_cloud).unsqueeze(0)])
            
            rgbs = torch.cat([rgbs, rgb.unsqueeze(0)])
            pcds = torch.cat([pcds, pcd.unsqueeze(0)])
            states = torch.cat([states, ac.unsqueeze(0)])

        action = states[1:]         # except for the start index action
        rgbs = rgbs[:-1]            # except for the end index rgb   T, N, H, W, C
        pcds = pcds[:-1]            # except for the end index pcd
        gripper = states[:-1]       # except for the end index action

        rgbs = rgbs.permute(0,1,4,2,3)
        pcds = pcds.permute(0,1,4,2,3)
        # normalise to [-1, 1]
        rgbs = rgbs / 255.0
        rgbs = 2 * (rgbs - 0.5)

        use_frames = select_frames[:-1]
        attn_indices = [{cam: obs_to_attn(demos[0]._observations[f], cam) for cam in self.cameras} for f in use_frames]

        frame_ids = list(range(len(select_frames) - 1))

        device = torch.device("cpu")
        langs = []
        langs.append(lang)
        lang_feat = get_language_feat(langs, "clip", args.num_words, device).float()
        lang_feat = lang_feat.squeeze()
        
        # 存储带有 waypoint 的相关信息
        state_dict: List = [[] for _ in range(7)]
        state_dict[0].extend(frame_ids)
        state_dict[1].extend(rgbs)              # torch.Size([4, 3, 3, 128, 128])
        state_dict[2].extend(pcds)
        state_dict[3].extend(action)
        state_dict[4].extend(gripper)
        state_dict[5].extend(attn_indices)
        state_dict[6].append(lang_feat)

        taskvar_dir = self.output / task_name / f"variation{variation_number}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)
        np.save(taskvar_dir / f"{episode_name}.npy", state_dict)


    
    @staticmethod
    def depth2normal(d_im):
        d_im = d_im.astype("float32")
        # zy, zx = np.gradient(d_im)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=3)
        zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=3)
        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # offset and rescale values to be in 0-1
        normal += 1
        normal /= 2
        return normal

    @staticmethod
    def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        return np.array([x1, y1, x2, y2])

    def __len__(self):
        return len(self.episode_list)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args: Arguments):
        # load RLBench environment
        self.env = RLBenchEnv(
            data_path=args.data_dir,
            apply_rgb=True,
            apply_pc=True,
            apply_cameras=args.cameras,
        )

        with open("episodes.json") as fid:
            episodes = json.load(fid)
        self.max_eps_dict = episodes["max_episode_length"]
        self.variable_lengths = set(episodes["variable_length"])

        for task_str in args.tasks:
            if task_str in self.max_eps_dict:
                continue
            _, state_ls, _ = get_observation(task_str, args.offset, 0, self.env)
            self.max_eps_dict[task_str] = len(state_ls) - 1
            raise ValueError(
                f"Guessing that the size of {task_str} is {len(state_ls) - 1}"
            )

        broken = set(episodes["broken"])
        tasks = [t for t in args.tasks if t not in broken]
        variations = range(args.offset, args.max_variations)
        self.items = []
        for task_str, variation in itertools.product(tasks, variations):
            episodes_dir = args.data_dir / task_str / f"variation{variation}" / "episodes"
            episodes = [
                (task_str, variation, int(ep.stem[7:]))
                for ep in episodes_dir.glob("episode*")
            ]
            self.items += episodes

        self.num_items = len(self.items)

    def __len__(self) -> int:
        return self.num_items

    def __getitem__(self, index: int) -> None:
        task, variation, episode = self.items[index]
        taskvar_dir = args.output / f"{task}+{variation}"
        taskvar_dir.mkdir(parents=True, exist_ok=True)

        try:
            demo, state_ls, action_ls = get_observation(
                task, variation, episode, self.env
            )
        except (FileNotFoundError, RuntimeError, IndexError) as e:
            print(e)
            return
        #调整维度
        state_ls = einops.rearrange(
            state_ls,
            "t 1 (m n ch) h w -> t n m ch h w",
            ch=3,
            n=len(args.cameras),
            m=2,
        )

        frame_ids = list(range(len(state_ls) - 1))
        num_frames = len(frame_ids)
        attn_indices = get_attn_indices_from_demo(task, demo, args.cameras) # {'left_shoulder': (97, 5), 'right_shoulder': (35, 6), 'wrist': (66, 112)}

        if (task in self.variable_lengths and num_frames > self.max_eps_dict[task]) or (
            task not in self.variable_lengths and num_frames != self.max_eps_dict[task]
        ):
            print(f"ERROR ({task}, {variation}, {episode})")
            print(f"\t {len(frame_ids)} != {self.max_eps_dict[task]}")
            return

        state_dict: List = [[] for _ in range(5)]
        print("Demo {}".format(episode))
        state_dict[0].extend(frame_ids)     #[0,1,2,3]                                      frame_ids
        state_dict[1].extend(state_ls[:-1]) # except for the end index img                  states
        state_dict[2].extend(action_ls[1:]) # except for the start index action             action
        state_dict[3].extend(attn_indices)  #                                               cameras
        state_dict[4].extend(action_ls[:-1]) # except for the end index action              gripper

        np.save(taskvar_dir / f"ep{episode}.npy", state_dict)  # type: ignore


if __name__ == "__main__":
    args = Arguments().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if 'all' in list(args.tasks):
        args.train_tasks =  [
            'drop_pen_color', 'drop_pen_relative', 'drop_pen_size',
            'wipe_table_color', 'wipe_table_relative', 'wipe_table_shape', 'wipe_table_size', 'wipe_table_direction',
            'pour_demo_color', 'pour_demo_relative', 'pour_demo_size',
            'pick_cube_color', 'pick_cube_relative', 'pick_cube_shape', 'pick_cube_size',
            'stack_cubes_color', 'stack_cubes_size', 'stack_cubes_relative', 'stack_cubes_shape',
            'place_into_shape_sorter_color', 'place_into_shape_sorter_shape', 'place_into_shape_sorter_relative',
            'open_drawer',
            'open_door_complex'
            ]
    else:
        args.train_tasks = list(args.tasks)

    args.output = args.output/args.mode
    dataset = Hive_dataset(
            args.train_dir, 
            'train_single_variation',
            img_size=args.img_size,
            unused_camera_list = args.unused_camera_list, 
            preprocess = args.preprocess, 
            use_fail_cases = args.use_fail_cases, 
            sample_numbers = args.sample_numbers, 
            train_tasks=args.train_tasks,
            args=args
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=args.workers,
        collate_fn=lambda x: x,
        pin_memory=True,
        persistent_workers=args.persistent_workers,
    )

    for _ in tqdm(dataloader):
        continue

