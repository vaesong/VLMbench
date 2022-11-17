import os
import numpy as np
from torch.utils.data import Dataset
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

class VLM_dataset(Dataset):
    def __init__(self, root, setd, img_size=(256, 256), 
                    unused_camera_list = ['left_shoulder', 'right_shoulder', 'overhead','wrist'], preprocess = True, 
                    use_fail_cases = True, sample_numbers = None, train_tasks = None, random_sample = False, args=None):
        self.root = root
        self.setd = setd
        self.dataset_path = Path(os.path.join(self.root, self.setd))
        self.task_list = {}
        self.variation_list = []
        self.episode_list = []   
        self.fail_cases_list = []
        self.read_lists()
        self.cameras = args.cameras
        self.use_fail_cases = use_fail_cases
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
        self.tokenizer = BertTokenizer.from_pretrained('/home/liuchang/projects/VLMbench/VLMbench/vlm/scripts/base-no-labels/ep_67_588997')

    def read_lists(self):
        tasks_list_path = self.dataset_path / '{}_list.pkl'.format(self.setd)
        if not tasks_list_path.is_file():
            self.task_list = {}
            self.variation_list =set()
            for path in self.dataset_path.rglob('low_dim_obs*'):#PosixPath('/home/zp_3c/liuchang/vlmbench/data/train/open_drawer/variation2/episodes/episode0/low_dim_obs.pkl')
                path = path.relative_to(self.dataset_path)#PosixPath('open_drawer/variation2/episodes/episode0/low_dim_obs.pkl')
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
        output_dict = self.get_episode(episode,fake=False)
        
        return output_dict

    def get_episode(self,episode,fake):
        variation_path = episode.parents[1]
        task_name = episode.parents[2]
        fail_cases = 'fail_cases' in str(episode)

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
                    lang+=str(f" Step {num2words(obs_select_inds.index(i))}:"+obs.low_level_description)
        if self.preprocess:
            preprocess_data_folder = self.dataset_path/episode/'preprocess_data'

            need_rebuild = False
            if not preprocess_data_folder.is_dir():
                preprocess_data_folder.mkdir()
                need_rebuild = True
            if not hasattr(demo_temple, 'observation_config'):
                need_rebuild = True
            else:
                need_rebuild = not (demo_temple.observation_config == self.obs_config)
            obs_list = os.listdir(preprocess_data_folder)
            if len(obs_list)<len(obs_select_inds):
                need_rebuild=True
            elif len(obs_list)>0:
                obs = []
                for i in obs_select_inds:
                    ob_path = preprocess_data_folder/(str(i)+'_preprocess.pkl')
                    if ob_path.is_file():
                        try:
                            with open(ob_path, 'rb') as f:
                                obs.append(pickle.load(f))
                        except:
                            need_rebuild = True
                            break
                    else:
                        need_rebuild = True
                        break
            if need_rebuild:
                episode_name = episode.name
                variation_number = int(variation_path.name.replace('variation',''))
                demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
                                    task_name, self.obs_config, episode_name, fail_cases)
                data = demos[0]
                obs = data._observations
                for i in obs_select_inds:
                    file_name = preprocess_data_folder/ "{}_preprocess.pkl".format(i)
                    with open(file_name, 'wb') as f:
                        pickle.dump(obs[i], f)
                print('Finish {} preprocess!'.format(episode))
                demo_temple.observation_config = self.obs_config
                with open(low_dim_obs, 'wb') as f:
                    pickle.dump(demo_temple, f)
                obs = [obs[i] for i in obs_select_inds]
        else:
            # episode_number = int(episode.name.replace('episode',''))
            episode_name = episode.name
            variation_number = int(variation_path.name.replace('variation',''))

            if self.mode == 'waypoint':
                key_frames = obs_select_inds
            else:
                key_frames = keypoint_discovery(demo_temple._observations)

            select_frames=[]

            # 获取抽样轨迹  
            max_traj_len= self.args.maxAction
            if 0 not in key_frames:
                select_frames.append(0)
            for frame in key_frames:
                if frame not in select_frames:
                    select_frames.append(frame)
            # add end obs to max_traj_len
            if (sequence_length-1) not in select_frames:
                select_frames.append(sequence_length-1)
            
            if self.mode == 'keyframe':
                sperate_index = max_sperate_index(select_frames)
                range1 = select_frames[sperate_index-1]
                range2 = select_frames[sperate_index]

                random_index1 = random.randint(range1+1,int(range1+(range2-range1)/2))
                random_index2 = random.randint(random_index1+1,range2-1)
                select_frames.insert(sperate_index,random_index2)
                select_frames.insert(sperate_index,random_index1)

            valid_action_length = len(select_frames) - 1

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

            pad_len = max_traj_len - valid_action_length
            padding_mask = torch.tensor([True] * valid_action_length + [False] * pad_len)

            # padding
            img_pad_vec = [0, 0] * rgbs.dim()
            img_pad_vec[-1] = pad_len
            rgbs = F.pad(rgbs, img_pad_vec, value=0)
            pcds = F.pad(pcds, img_pad_vec, value=0)

            action_pad_vec = [0, 0] * action.dim()
            action_pad_vec[-1] = pad_len
            action = F.pad(action, action_pad_vec, value=0)
            gripper = F.pad(gripper, action_pad_vec, value=0)

            tframe_ids = torch.tensor(np.array(range(valid_action_length)))
            tframe_ids = F.pad(tframe_ids, (0, pad_len), value=-1)

            use_frames = select_frames[:-1]
            attn_indices = [{cam: obs_to_attn(demos[0]._observations[f], cam) for cam in self.cameras} for f in use_frames]

            attns = torch.Tensor([])
            for i in range(0, len(use_frames)):
                attn_cams = torch.Tensor([])
                for cam in self.cameras:
                    u, v = attn_indices[i][cam]
                    attn = torch.zeros((1, 1, 128, 128))
                    if not (u < 0 or u > 127 or v < 0 or v > 127):
                        attn[0, 0, v, u] = 1
                    attn_cams = torch.cat([attn_cams, attn])
                attns = torch.cat([attns, attn_cams.unsqueeze(0)])  # num 3(cameras) 1 360 360
            pad_vec = [0] * (2 * attns.dim())
            pad_vec[-1] = pad_len
            attns = F.pad(attns, pad_vec)
            rgbs = torch.cat([rgbs, attns], 2)

            # data augmentation
            modals = self._transform(rgbs=rgbs, pcds=pcds)
            rgbs = modals["rgbs"]
            pcds = modals["pcds"]

            output_dict = {
                # instruction
                "language": str(lang), 
                # img and pcd
                "rgbs": rgbs,
                "pcds": pcds,
                # "attns": attns,
                # state
                "action": action,
                "gripper": gripper,
                # others
                "padding_mask": padding_mask,
                "valid_length": valid_action_length,
                "task": str(task_name)
            }
            return output_dict
    
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