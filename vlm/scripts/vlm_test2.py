from vlm_test import hiveformerAgent,set_seed,set_obs_config,set_env,load_test_config
import argparse
from distutils.util import strtobool
from amsolver.backend.utils import task_file_to_task_class
import os
from pathlib import Path
import numpy as np
from num2words import num2words
import pickle
from vlm.scripts.utils import keypoint_discovery
from amsolver.utils import get_stored_demos
import torch
from hiverformer.utils import RLBenchEnv
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
import cv2
import argparse
import os
import random
from distutils.util import strtobool
from pathlib import Path

import cv2
import numpy as np
import torch
from num2words import num2words
from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor
from pytorch_transformers import BertTokenizer
from scipy.spatial.transform import Rotation as R
from torch.autograd import Variable
# from train_vlmbench import load
from hiverformer.process_instructions import get_language_feat
from amsolver.action_modes import ActionMode, ArmActionMode
from amsolver.backend.utils import task_file_to_task_class
from amsolver.environment import Environment
from amsolver.observation_config import ObservationConfig
# from amsolver.utils import get_stored_demos
from cliport.agent import (BlindLangAgent_6Dof, ImgDepthAgent_6dof,
                           TwoStreamClipLingUNetLatTransporterAgent)
from hiverformer.network import Hiveformer
from hiverformer.utils import obs_to_attn,RLBenchEnv,Mover

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

def add_argments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task', type=str, default="drawer")
    parser.add_argument('--setd', type=str, default="seen")
    parser.add_argument('--mode', type=str, default="test")
    parser.add_argument('--ignore_collision', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--recorder', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--data_folder', type=str, default="/home/liuchang/DATA/rlbench_data/single_test")
    # parser.add_argument('--setd', type=str, default="seen")
    parser.add_argument("--load", type=str, default="/home/liuchang/projects/VLMbench/VLMbench/xp/hiveformer/open_keyframe_version0/model.epoch=15000-value=0.pth", help='path of the trained model')
    parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=False)
    args = parser.parse_args()
    return args

if __name__=="__main__":
    set_seed(0)
    args = add_argments()
    # set_env
    obs_config = set_obs_config([128,128])
    task_files,env = set_env(args,obs_config)
    env.launch()    
    agent = hiveformerAgent(args)
    recorder = Recorder() if args.recorder else None

    # set_output
    output_file_name = f"/home/liuchang/projects/VLMbench/VLMbench/results_hiveformerAgent/hiveformerAgent_{args.task}_{args.setd}"
    output_file_name += ".txt"
    file = open(output_file_name, "w")

    # just to use its function
    tmp = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=True
        )

    model = Hiveformer(
                    depth=4,
                    dim_feedforward=64,
                    hidden_dim=64,
                    instr_size=512, #768
                    mask_obs_prob=0.0,
                    max_episode_length=3,#100
                    num_words=75,
                    num_layers=1,
                    num_tasks = 24 #106
                    ).cuda()
    model_dict = torch.load(args.load, map_location="cpu")
    model.load_state_dict(model_dict["weight"])
    print("loading model from "+ str(args.load))

    # load_env_data
    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    if args.mode == "valid":
        data_folder = Path(args.data_folder)
    else:
        data_folder = Path(os.path.join(args.data_folder, args.setd))
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        success_times,grasp_success_times,all_time = 0,0,0
        task = env.get_task(task_to_train)
        # move = Mover(task, max_tries=10)
        for num, e in enumerate(e_path):
            if "episode16" not in str(e):
                continue
            task_base = str(e/"task_base.ttm")
            waypoint_sets = str(e/"waypoint_sets.ttm")
            config = str(e/"configs.pkl")
            descriptions, obs = task.load_config(task_base, waypoint_sets, config)
            waypoints_info = {name: obj for name, obj in obs.object_informations.items() if "waypoint" in name}
            all_time+=1
            high_descriptions = descriptions[0]     
            step=0
            if args.add_low_lang:
                for waypoint in waypoints_info:
                    if "low_level_descriptions" in waypoints_info[waypoint]:
                        high_descriptions = high_descriptions+f" Step {num2words(step)}:"
                        high_descriptions += waypoints_info[waypoint]["low_level_descriptions"]
                        step+=1
            print(high_descriptions)
            target_grasp_obj_name = None
            try:
                if len(waypoints_info['waypoint1']['target_obj_name'])!=0:
                    target_grasp_obj_name = waypoints_info['waypoint1']['target_obj_name']
                    grasp_pose = waypoints_info['waypoint1']['pose'][0]
                else:
                    grasp_pose = waypoints_info['waypoint1']['pose'][0]
                    target_name = None
                    distance = np.inf
                    for g_obj in task._task.get_graspable_objects():
                        obj_name = g_obj.get_name()
                        obj_pos = g_obj.get_position()
                        c_distance = np.linalg.norm(obj_pos-grasp_pose[:3])
                        if c_distance < distance:
                            target_name = obj_name
                            distance = c_distance
                    if distance < 0.2:
                        target_grasp_obj_name = target_name
            except:
                print(f"need re-generate: {e}")
                continue
            
            # get_key_frame_data if valid mode
            if args.mode == "valid":
                low_dim_obs = str(e/"low_dim_obs.pkl")
                with open(low_dim_obs, 'rb') as f:
                    demo_temple = pickle.load(f)
                    key_frames = keypoint_discovery(demo_temple._observations)
                if 0 not in key_frames:
                    key_frames.insert(0,0)
                if (len(demo_temple)-1) not in key_frames:
                    key_frames.insert((len(demo_temple)-1),-1)
                demos = get_stored_demos(1, False, e, variation_number, 
                        task_name, obs_config , episode_name,selected_frame=key_frames)

            variation_path = e.parents[1]
            variation_number = int(variation_path.name.replace('variation',''))
            task_name = e.parents[2]
            episode_name = e.name

            rgbs = torch.Tensor([])
            pcds = torch.Tensor([])
            grippers = torch.Tensor([])
            preds=[]

            for frame in range(3):
                rgb,pcd,gripper = tmp.get_rgb_pcd_gripper_from_obs(obs)
                rgbs = torch.cat([rgbs , rgb.unsqueeze(1)], dim=1)
                pcds = torch.cat([pcds , pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers , gripper.unsqueeze(1)], dim=1)
                padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool().cuda()
                lang = []
                lang.append(high_descriptions)
                language = get_language_feat(lang,"clip",75,device=padding_mask.device)
                pred = model(
                    rgbs.cuda(),
                    pcds.cuda(),
                    padding_mask,
                    language.cuda(),
                    grippers.cuda(),
                    )
                action = model.compute_action(pred).detach().cpu().numpy()
                action_np = action[-1]
                obs, reward, terminate = task.step(action_np, None , recorder = recorder, need_grasp_obj = target_grasp_obj_name)


            for frame in key_frames:
                rgb = torch.Tensor([])
                pcd = torch.Tensor([])
                gripper = torch.Tensor([])

                obs=demos[0]._observations[frame]
                rgb,pcd,gripper = tmp.get_rgb_pcd_gripper_from_obs(obs)
                
                if frame != key_frames[-1]:
                    rgbs = torch.cat([rgbs , rgb.unsqueeze(1)], dim=1)
                    pcds = torch.cat([pcds , pcd.unsqueeze(1)], dim=1)
                grippers = torch.cat([grippers , gripper.unsqueeze(1)], dim=1)
                padding_mask = torch.ones_like(rgbs[:, :, 0, 0, 0, 0]).bool().cuda()
                if frame != key_frames[-1]:


                    pred = model(
                    rgbs.cuda(),
                    pcds.cuda(),
                    padding_mask,
                    language.cuda(),
                    grippers.cuda(),
                    )
                    action = model.compute_action(pred).detach().cpu().numpy()
                    preds.append(action)
            print(grippers)
            print(preds)


