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


# from param import args
# from pyvirtualdisplay import Display
# disp = Display().start()
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

class ReplayAgent(object):

     def act(self, step_list, step_id, obs, lang, use_gt_xy=False,use_gt_z=False, use_gt_theta=False, use_gt_roll_pitch=False):
        current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step_list[step_id]
        action = np.zeros(8)
        action[:7] = gt_pose
        action[7] = gripper_control
        return action, waypoint_type

class CliportAgent(object):
    def __init__(self, model_name, device_id=0, z_roll_pitch=True, checkpoint=None, args=None) -> None:
        cfg = {
            'train':{
                'attn_stream_fusion_type': 'add',
                'trans_stream_fusion_type': 'conv',
                'lang_fusion_type': 'mult',
                'n_rotations':36,
                'batchnorm':False
            }
        }
        device = torch.device(device_id)
        if model_name=="cliport_6dof":
            self.agent = TwoStreamClipLingUNetLatTransporterAgent(name='agent', device=device, cfg=cfg, z_roll_pitch=z_roll_pitch).to(device)
        elif model_name == "imgdepth_6dof":
            self.agent = ImgDepthAgent_6dof(name='agent',device=device, cfg=cfg).to(device)
        elif model_name == 'blindlang_6dof':
            self.agent = BlindLangAgent_6Dof(name='agent',device=device, cfg=cfg).to(device)
        self.model_name = model_name
        if checkpoint is not None:
            state_dict = torch.load(checkpoint,device)
            self.agent.load_state_dict(state_dict['state_dict'])
        self.agent.eval()
        self.args = args
    @staticmethod
    def generate_action_list(waypoints_info, args):
        all_waypoints = []
        i=0
        while True:
            waypoint_name = f"waypoint{i}"
            i+=1
            if waypoint_name in waypoints_info:
                all_waypoints.append(waypoint_name)
            else:
                break
        step_list, point_list = [], []
        attention_id = waypoints_info["waypoint0"]["target_obj"]
        gripper_control = 1
        for i, wp in enumerate(all_waypoints):
            waypoint_info = waypoints_info[wp]
            waypoint_type = waypoint_info['waypoint_type']
            # if "pre" in waypoint_type:
            #     focus_wp = all_waypoints[i+1]
            # elif "post" in waypoint_type:
            #     focus_wp = all_waypoints[i-1]
            # else:
            focus_wp = wp
            if focus_wp not in point_list:
                focus_waypoint_info = waypoints_info[focus_wp]
                if "grasp" in waypoint_type:
                    attention_id = waypoint_info["target_obj"]
                    related_rotation = False
                else:
                    related_rotation = args.relative
                if focus_waypoint_info["gripper_control"] is not None:
                    gripper_control = focus_waypoint_info["gripper_control"][1]
                gt_pose = focus_waypoint_info['pose'][0]
                point_list.append(focus_wp)
                step_list.append([focus_wp, focus_waypoint_info['low_level_descriptions'], attention_id, gripper_control, focus_waypoint_info["waypoint_type"], related_rotation, gt_pose])
        return step_list
    
    def act(self, step_list, step_id, obs, lang, use_gt_xy=False,use_gt_z=False, use_gt_theta=False, use_gt_roll_pitch=False):
        current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step_list[step_id]
        with torch.no_grad():
            z_max = 1.2
            if 'door' in self.args.task or 'drawer' in self.args.task:
                z_max = 1.8
            inp_img, lang_goal, p0, output_dict = self.agent.act(obs, [lang], bounds = np.array([[-0.05,0.67],[-0.45, 0.45], [0.7, z_max]]), pixel_size=5.625e-3)
        action = np.zeros(8)
        action[:7] = gt_pose
        if not use_gt_xy:
            action[:2] = output_dict['place_xy']
        if not use_gt_z:
            action[2] = output_dict['place_z']
        action[7] = gripper_control
        if related_rotation:
            prev_pose = R.from_quat(self.prev_pose[3:])
            current_pose = R.from_quat(gt_pose[3:])
            rotation = (prev_pose.inv()*current_pose).as_euler('zyx')
            # rotation[rotation<0]+=2*np.pi
            if not use_gt_theta:
                rotation[0] = output_dict['place_theta']
            if not use_gt_roll_pitch:
                rotation[1] = output_dict['pitch']
                rotation[2] = output_dict['roll']
            related_rot = R.from_euler("zyx",rotation)
            action[3:7] = (prev_pose*related_rot).as_quat()
        else:
            rotation = R.from_quat(gt_pose[3:]).as_euler('zyx')
            # rotation[rotation<0]+=2*np.pi
            if not use_gt_theta:
                rotation[0] = output_dict['place_theta']
            if not use_gt_roll_pitch:
                rotation[1] = output_dict['pitch']
                rotation[2] = output_dict['roll']
            action[3:7] = R.from_euler("zyx",rotation).as_quat()
        self.prev_pose = action[:7]
        return action, waypoint_type

class hiveformerAgent():
    def __init__(self,args=None):
        self.args = args
        self.rgbs = torch.Tensor([])
        self.pcds = torch.Tensor([])
        self.grippers = torch.Tensor([])
        self.tok = BertTokenizer.from_pretrained('/home/liuchang/projects/VLMbench/VLMbench/vlm/scripts/base-no-labels/ep_67_588997')
        self.env = RLBenchEnv(
        data_path="",
        apply_rgb=True,
        apply_pc=True,
        apply_cameras=("left_shoulder", "right_shoulder", "wrist"),
        headless=True
        )
        self.model = Hiveformer(
        depth=4,
        dim_feedforward=64,
        hidden_dim=64,
        instr_size=512, #768
        mask_obs_prob=0.0,
        max_episode_length=args.maxAction,#100
        num_words=75,
        num_layers=1,
        num_tasks = 24 #106
        ).cuda()
        if args.load is not None:
            model_dict = torch.load(args.load, map_location="cpu")
            self.model.load_state_dict(model_dict["weight"])
            print("loading model from "+ str(args.load))
    def clear (self):
        self.rgbs = torch.Tensor([])
        self.pcds = torch.Tensor([])
        self.grippers = torch.Tensor([])
    
    def act(self,obs,language,action_feat,step,step_id):
        # current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step
        with torch.no_grad():
            ob = obs # get current obs
            rgb,pcd,gripper = self.env.get_rgb_pcd_gripper_from_obs(ob)
            self.rgbs = torch.cat([self.rgbs , rgb.unsqueeze(1)], dim=1)
            self.pcds = torch.cat([self.pcds , pcd.unsqueeze(1)], dim=1)
            self.grippers = torch.cat([self.grippers , gripper.unsqueeze(1)], dim=1)
            padding_mask = torch.ones_like(self.rgbs[:, :, 0, 0, 0, 0]).bool().cuda()

            # lang_tokens = self.tok.tokenize(language)
            # language = ['[CLS]'] + language + ['[SEP]']
            # instr_tokens = temp_instr_tokens + ['[PAD]'] * (80-len(temp_instr_tokens))
            # language = torch.from_numpy(np.array(self.tok.convert_tokens_to_ids(instr_tokens))).unsqueeze(0)
            # print(language)
            lang = []
            lang.append(language)
            language = get_language_feat(lang,"clip",75,device=padding_mask.device)

            # self.model.eval()
            
            pred = self.model(
                self.rgbs.cuda(),
                self.pcds.cuda(),
                padding_mask,
                language.cuda(),
                self.grippers.cuda(),
            )
            action = self.model.compute_action(pred)  # type: ignore

            # output["attention"] = pred["attention"]
            # action[:,0] = torch.clamp(action[:,0],-0.274,0.774)
            # action[:,1] = torch.clamp(action[:,1],-0.654,0.654)
            # action[:,2] = torch.clamp(action[:,2],0.713,1.751)

            action_np = action[-1].detach().cpu().numpy()

            # if waypoint_type =="grasp":
            #     if  np.linalg.norm(action_np[:3]-(gt_pose[:3])) < 0.1:
            #         action_np[:7] = gt_pose[:]

        return action_np

def load_test_config(data_folder: Path, task_name):
    episode_list = []
    for path in data_folder.rglob('configs*'):
        t_name = path.parents[3].name
        v_name = path.parents[2].name
        if t_name == task_name :
            episode_list.append(path.parent)
    return episode_list

def set_seed(seed, torch=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    if torch:
        import torch
        torch.manual_seed(seed)

def set_obs_config(img_size):
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    
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
    return obs_config

def set_env(args,obs_config):
    need_post_grap = True
    need_pre_move = False
    if args.task == 'drop':
        task_files = ['drop_pen_color', 'drop_pen_relative', 'drop_pen_size']
    elif args.task == 'pick':
        # task_files = ['pick_cube_shape', 'pick_cube_relative', 'pick_cube_color', 'pick_cube_size']
        task_files = ['pick_cube_color']
    elif args.task == 'stack':
        # task_files = ['stack_cubes_color', 'stack_cubes_relative', 'stack_cubes_shape', 'stack_cubes_size']
        task_files = ['stack_cubes_color']
    elif args.task == 'shape_sorter':
        need_pre_move = True
        args.ignore_collision = True
        task_files = ['place_into_shape_sorter_color', 'place_into_shape_sorter_relative', 'place_into_shape_sorter_shape']
    elif args.task == 'wipe':
        args.ignore_collision = True
        task_files = ['wipe_table_shape', 'wipe_table_color', 'wipe_table_relative', 'wipe_table_size', 'wipe_table_direction']
    elif args.task == 'pour':
        task_files = ['pour_demo_color', 'pour_demo_relative', 'pour_demo_size']
    elif args.task == 'drawer':
        args.ignore_collision = True
        args.renew_obs = False
        need_post_grap=False
        task_files = ['open_drawer']
    elif args.task == 'door':
        args.ignore_collision = True
        need_post_grap=False
        task_files = ['open_door']
    elif args.task == 'door_complex':
        args.ignore_collision = True
        need_post_grap=False
        task_files = ['open_door_complex']
    else:
        task_files = [args.task]
    if args.ignore_collision:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME)
    else:
        action_mode = ActionMode(ArmActionMode.ABS_EE_POSE_PLAN_WORLD_FRAME_WITH_COLLISION_CHECK)
    env = Environment(action_mode, obs_config=obs_config, headless=True) # set headless=False, if user want to visualize the simulator
    return task_files,env

def add_argments():
    parser = argparse.ArgumentParser(description='')
    #dataset
    parser.add_argument('--data_folder', type=str, default="/home/liuchang/DATA/rlbench_data/single_test")
    parser.add_argument('--setd', type=str, default="seen")
    parser.add_argument("--load", type=str, default="/home/liuchang/projects/VLMbench/VLMbench/xp/hiveformer/open_keyframe_version0/model.epoch=15000-value=0.pth", help='path of the trained model')
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--model_name', type=str, default="cliport_6dof")
    parser.add_argument('--maxAction', type=int, default=3, help='Max Action sequence')
    parser.add_argument('--img_size',nargs='+', type=int, default=[128,128])
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--language_padding', type=int, default=80)
    parser.add_argument('--task', type=str, default="drawer")
    parser.add_argument('--replay', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--recorder', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--relative', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--renew_obs', type=lambda x:bool(strtobool(x)), default=True)
    parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--ignore_collision', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--goal_conditioned', type=lambda x:bool(strtobool(x)), default=False)
    parser.add_argument('--wandb_entity', type=str, default=None, help="visualize the test results. Account Name")
    parser.add_argument('--agent', type=str, default="hiveformerAgent", help="test agent")
    parser.add_argument('--wandb_project', type=str, default=None,  help="visualize the test results. Project Name")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = add_argments()
    if args.wandb_entity is not None:
        import wandb
    set_seed(0)
    obs_config = set_obs_config(args.img_size)
    # recorder = Recorder()
    need_test_numbers = 20
    replay_test = args.replay
    
    renew_obs = args.renew_obs
    task_files,env = set_env(args,obs_config)

    env.launch()

    if args.recorder:
        recorder = Recorder()
    else:
        recorder = None

    train_tasks = [task_file_to_task_class(t, parent_folder = 'vlm') for t in task_files]
    data_folder = Path(os.path.join(args.data_folder, args.setd))


    # if not replay_test:
    #     checkpoint = args.checkpoints
    #     agent = CliportAgent(args.model_name, device_id=args.gpu,z_roll_pitch=True, checkpoint=checkpoint, args=args)
    # else:
    #     agent = ReplayAgent()
    if args.agent == "CliportAgent":
        checkpoint = args.checkpoints
        agent = CliportAgent(args.model_name, device_id=args.gpu,z_roll_pitch=True, checkpoint=checkpoint, args=args)
    elif args.agent =="hiveformerAgent":
        agent = hiveformerAgent(args)
    else:
        agent = ReplayAgent()

    output_file_name = f"/home/liuchang/projects/VLMbench/VLMbench/results_{args.agent}/{args.agent}_{args.task}_{args.setd}"
    if args.goal_conditioned:
        output_file_name += "_goalcondition"
    if args.ignore_collision:
        output_file_name += "_ignore"
    output_file_name += ".txt"
    file = open(output_file_name, "w")
    for i, task_to_train in enumerate(train_tasks):
        e_path = load_test_config(data_folder, task_files[i])
        success_times,grasp_success_times,all_time = 0,0,0
        task = env.get_task(task_to_train)
        # move = Mover(task, max_tries=10)
        for num, e in enumerate(e_path):
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
            step_list = CliportAgent.generate_action_list(waypoints_info, args)
            history_img = [obs]
            history_action = [(np.append(obs.gripper_pose,obs.gripper_open))]
            hidden_states,rewards,actions = [],[],[]
            use_gt_xy,use_gt_z,use_gt_theta,use_gt_roll_pitch = False ,False,True,True
            grasped = False
            for i in range(args.maxAction):
                print(i)
                # x_loss,y_loss,z_loss=0,0,0
                # step = step_list[i]
                if args.agent =="hiveformerAgent" and i==0:
                    agent.clear()     
                action = agent.act(obs,high_descriptions,history_action,step,i)         
                print("action:")
                print(action)
                # current_waypoint,_, attention_id, gripper_control, waypoint_type, related_rotation, gt_pose  = step
                # print("gt_pose:")
                # print(gt_pose)
                # if waypoint_type == "grasp":
                #     collision_checking = False 
                #     pre_action = action.copy()
                #     pose = R.from_quat(action[3:7]).as_matrix()
                #     pre_action[:3] -= 0.08*pose[:, 2]
                #     pre_action[7] = 1
                #     # action_list+=[pre_action, action]
                # else: 
                #     collision_checking = True
                
                # x_loss=(action[0]-gt_pose[0])
                # y_loss=(action[1]-gt_pose[1])
                # z_loss=(action[2]-gt_pose[2])
                # if args.goal_conditioned:
                #     action[:7] = gt_pose
                #     action[7] = gripper_control
                # action[3:7]=gt_pose[3:7]
                # if i == len(step_list)-1:
                #     action[7] = 1
                # file.write(f"{task.get_name()}:x_loss: {x_loss}, y_loss: {y_loss}, z_loss {z_loss} %!\n")
                # print(f"{task.get_name()}:x_loss: {x_loss}, y_loss: {y_loss}, z_loss {z_loss} %!\n")
                try:
                    # if waypoint_type == "grasp":
                    # obs, reward, terminate = task.step(pre_action, None, recorder = recorder, need_grasp_obj = target_grasp_obj_name)
                    # if i == 1:
                    #     collision_checking = False  
                    # else :
                    #     collision_checking = True
                    # if i == 1:
                    #     print("grasp_pose:")
                    #     print(grasp_pose)
                    #     action[:7] = grasp_pose
                    #     action[7] = 0 
                    obs, reward, terminate = task.step(action, None , recorder = recorder, need_grasp_obj = target_grasp_obj_name)
                    history_img.append(obs)
                    history_action.append((np.append(obs.gripper_pose,obs.gripper_open)))
                except:
                    reward = 0 
                    break
                # rewards.append(reward)
                if reward == 0.5:
                    grasped = True
                    grasp_success_times+=1
                elif reward == 1:
                    success_times+=1
                    successed = True
                    break
            # if reward == 1 or grasped == True:
            recorder.save(f"./records_{args.agent}/{task.get_name()}/{num+1}.avi")
            recorder.del_snap()
            print(f"{task.get_name()}: success {success_times} times in {all_time} steps! success rate {round(success_times/all_time * 100, 2)}%!")
            print(f"{task.get_name()}: grasp success {grasp_success_times} times in {all_time} steps! grasp success rate {round(grasp_success_times/all_time * 100, 2)}%!")
            file.write(f"{task.get_name()}:grasp success: {grasp_success_times}, success: {success_times}, toal {all_time} steps, success rate: {round(success_times/all_time * 100, 2)}%!\n\n")   
    file.close()
    env.shutdown()




        






