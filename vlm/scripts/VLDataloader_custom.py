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

class VLM_dataset(Dataset):
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
        self.use_fail_cases = use_fail_cases
        if train_tasks is None:
            train_tasks =  [
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

        self.obs_config = ObservationConfig()
        self.obs_config.set_all(True)
        # self.obs_config.set_depth(False)
        self.obs_config.right_shoulder_camera.image_size = self.img_size
        self.obs_config.left_shoulder_camera.image_size = self.img_size
        self.obs_config.overhead_camera.image_size = self.img_size
        self.obs_config.wrist_camera.image_size = self.img_size
        self.obs_config.front_camera.image_size = self.img_size

        self.views = list(set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']) - set(unused_camera_list))

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
        output_dict = self.get_episode(episode)
        # itm data 
        fake_episode = random.choice(self.episode_list)
        while str(episode.parents[2]).split("_")[:2] == str(fake_episode.parents[2]).split("_")[:2]: # do it untill their tasks are totally different
            fake_episode = random.choice(self.episode_list)
        prob_itm = np.random.random()  #随机概率，random_episode = episode or fake_episode
        if  prob_itm <= 0.5:
                ismatch = 1
                # random_episode = episode
                output_dict["random_traj"] = output_dict["traj"]  
                output_dict["ismatch"] = ismatch
        elif prob_itm <= 1:
                ismatch = 0
                output_dict["random_traj"] = self.get_episode(fake_episode)["traj"]
                output_dict["ismatch"] = ismatch           
        return output_dict

    def get_episode(self,episode):
        variation_path = episode.parents[1]
        task_name = episode.parents[2]
        fail_cases = 'fail_cases' in str(episode)

        low_dim_obs = self.dataset_path/episode/"low_dim_obs.pkl"
        with open(low_dim_obs, 'rb') as f:
            demo_temple = pickle.load(f)
        
        sequence_length = len(demo_temple._observations)
        obs_select_inds = np.arange(sequence_length)
        if self.sample_numbers:
            if self.random_sample:
                obs_select_inds = np.sort(np.random.choice(obs_select_inds, self.sample_numbers, replace=False))
            else:
                obs_select_inds = obs_select_inds[0:self.sample_numbers]
        split_by_waypoint = True
        # 根据watpoints切分
        # obs_select_inds：选择出每个waypoint的开始index
        if split_by_waypoint:
            lang = demo_temple.high_level_instructions[0]
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
            # for i in range(len(obs_select_inds)):
            #     if i+1<len(obs_select_inds):
            #         random_i = np.random.randint(obs_select_inds[i], obs_select_inds[i+1])
            #     else:
            #         random_i = np.random.randint(obs_select_inds[i], sequence_length)
            #     obs_select_inds[i] = random_i
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
            # demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
            #                         task_name, self.obs_config, episode_name, fail_cases, obs_select_inds)
            # demos = get_stored_demos_nodepth(1, False, self.dataset_path, variation_number, 
            #                         task_name, self.obs_config , episode_name)            
            # data = demos[0]
            # obs = data._observations
            #拿到waypoint切割点obs_select_inds的observation
            # obs = [obs[i] for i in obs_select_inds]
            key_frames = keypoint_discovery(demo_temple._observations)
            action=[]
            traj=[]
            select_frames=[]
            # 用于策略训练
            # while i < sequence_length-sperate:
            #     frame_img=[]
            #     frame_state=[]
            #     for spe in range(sperate): # 间隔5帧取图片
            #         frame_img.append(obs[i+spe].front_rgb)#torch.from_numpy(np.array(i[t])).permute(0,3,2,1).float()
            #         frame_state.append(obs[i+spe].gripper_pose)
            #     i=i+sperate 
            #     if i < sequence_length:
            #         target.append(np.append((obs[i].gripper_pose-obs[i-sperate].gripper_pose),(obs[i].gripper_open)))
            #     else:
            #         target.append(target[-1])
            #     img.append(frame_img)
            #     state.append(frame_state)
                
            # 获取抽样轨迹  
            max_traj_len= self.args.maxAction
            if 0 not in key_frames:
                select_frames.append(0)
            for frame in key_frames:
                if frame not in select_frames:
                    select_frames.append(frame)
            # add end obs to max_traj_len
            if (sequence_length-1) not in select_frames:
                select_frames.append()(sequence_length-1)
            
            sperate_index = max_sperate_index(select_frames)
            range1 = select_frames[sperate_index-1]
            range2 = select_frames[sperate_index]

            random_index1 = random.randint(range1+1,int(range1+(range2-range1)/2))
            random_index2 = random.randint(random_index1+1,range2-1)
            select_frames.insert(sperate_index,random_index2)
            select_frames.insert(sperate_index,random_index1)

            valid_action_length = len(select_frames)

            demos = get_stored_demos_nodepth(1, False, self.dataset_path, variation_number, 
                                     task_name, self.obs_config , episode_name,selected_frame=select_frames)   

            for frame in select_frames:
                obs=demos[0]._observations[frame]
                action.append((np.append(obs.gripper_pose,obs.gripper_open)))
                traj.append(obs.front_rgb)
            while len(traj) < max_traj_len: # padding to max_traj_len
                action.append(action[-1])
                traj.append(traj[-1])

                  
            random_history_index=random.randint(1,valid_action_length-1)
            action_label = action[random_history_index]
            # history_traj = copy.deepcopy(traj)[:]
            # action = action[random_history_index]

            
            maxlength=self.args.maxInput
            # original tokens
            instr_tokens = self.tokenizer.tokenize(lang)
            temp_instr_tokens = ['[CLS]'] + instr_tokens + ['[SEP]']
            instr_tokens = temp_instr_tokens + ['[PAD]'] * (maxlength-len(temp_instr_tokens))
            instr_tokens=self.tokenizer.convert_tokens_to_ids(instr_tokens)
            
            # mask tokens
            mask_instr_tokens,mlm_label=mask_tokens(torch.LongTensor(instr_tokens),self.tokenizer)
            output_dict = {
                # "img":img,
                "traj": np.array(traj),
                # "target": target,
                "language": np.array(instr_tokens), 
                "mask_language":np.array(mask_instr_tokens),
                "mlm_label":np.array(mlm_label),
                "random_history_index":random_history_index,
                "valid_length":valid_action_length,
                "action":np.array(action),
                "action_label":action_label
                # "state":state,
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