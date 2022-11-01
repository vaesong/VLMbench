import json
import os
import sys
import numpy as np
import random
import math
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
# from env import R2RBatch
#import utils
#from utils import padding_idx, print_progress
import model_PREVALENT
import os
from pickle import NONE
import time
from tools.TimeAverageMeter import AverageMeter, sec_to_str
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from torch._C import device
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import argparse
import warnings
from distutils.util import strtobool
import sys
from os.path import join, dirname, abspath, isfile
import torch.nn.functional as F
from vlm.scripts.VLDataloader_custom import VLM_dataset
from pytorch_transformers import (BertConfig, BertTokenizer)
import utils
from tensorboardX import SummaryWriter
from pytorch_transformers import BertConfig
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
from pytorch_transformers import (WEIGHTS_NAME, AdamW, WarmupLinearSchedule,
                                  BertConfig, BertForMaskedLM, BertTokenizer)
from param import args
# def collate_fn(batch):
# #     output_batch = []
#     lens = [len(dat["img"]) for dat in batch]
#     maxlen=max(lens)
#     for dat in batch:
#         if len(dat["img"]) < maxlen:
#                 for i in range(maxlen-len(dat["img"])):
#                         dat["img"].append (np.zeros_like(dat["img"][0]))
#                         dat["state"].append (np.zeros_like(dat["state"][0]))        
#                         dat["language"].append ("") 
#                         dat["target"].append(np.full(7,fill_value=-100))         
#     return batch
def save(epoch, path,vln_bert,optim):
        ''' Snapshot models '''
        the_dir, _ = os.path.split(path)
        os.makedirs(the_dir, exist_ok=True)
        states = {}
        def create_state(name, model, optimizer):
                states[name] = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                }
        all_tuple = [("vln_bert", vln_bert, optim)]
                        # ("critic", self.critic, self.critic_optimizer)]
        for param in all_tuple:
                create_state(*param)
        torch.save(states, path)

def load(path,vln_bert,optim):
        ''' Loads parameters (but not training state) '''
        states = torch.load(path)

        def recover_state(name, model, optimizer):
                state = model.module.state_dict()
                model_keys = set(state.keys())
                load_keys = set(states[name]['state_dict'].keys())
                if model_keys != load_keys:
                        print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                        state.update(states[name]['state_dict'])
                        model.module.load_state_dict(state)
                        if args.loadOptim:
                                optimizer.load_state_dict(states[name]['optimizer'])
        all_tuple = [("vln_bert", vln_bert, optim)]
        for param in all_tuple:
                recover_state(*param)
        return states['vln_bert']['epoch'] - 1,vln_bert,optim

class NextActionPrediction(nn.Module):
    """
    2-class classification model : is_next, is_not_next
    """

    def __init__(self, hidden, actionspace):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, actionspace)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x)) 

def main(gpu, ngpus_per_node, args):

        args.gpu = gpu + args.gpu_start
        if args.gpu is not None:
                print("Use GPU: {} for training".format(args.gpu))
        if args.distributed:
                if args.dist_url == "env://" and args.rank == -1:
                        args.rank = int(os.environ["RANK"])
                
                args.rank = args.rank * ngpus_per_node + gpu
                dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                        world_size=args.world_size, rank=args.rank)


        train_dataset = VLM_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, args=args)
        # 设置采样
        if args.distributed:
                train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        else:
                train_sampler = None

        train_loader = torch.utils.data.DataLoader(  
                train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                num_workers=args.workers, pin_memory=True, sampler=train_sampler, 
                drop_last=True,persistent_workers=True) #,persistent_workers=True
                
        log_dir = 'snap/%s' % args.name
        if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        writer = SummaryWriter(log_dir=log_dir)
        argsDict = args.__dict__
        with open(log_dir+'/setting.txt', 'w') as f:
                f.writelines('------------------ start ------------------' + '\n')
                for eachArg, value in argsDict.items():
                        f.writelines(eachArg + ' : ' + str(value) + '\n')
                # f.writelines('------------------- end -------------------')

        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        criterion2 = nn.L1Loss()
         
        start_iter = 0
        
        config = BertConfig.from_pretrained("/home/liuchang/projects/VLMbench/VLMbench/vlm/scripts/base-no-labels/ep_67_588997")
        config.img_feature_dim = args.vision_size
        config.img_feature_type = ""
        config.update_lang_bert = args.update
        config.update_add_layer = args.update_add_layer
        config.vl_layers = args.vl_layers
        config.la_layers = args.la_layers
        config.action_space = args.action_space

        mlmhead = BertOnlyMLMHead(config).cuda(args.gpu)
        is_match = NextActionPrediction(config.hidden_size, 2).cuda(args.gpu)

        base_vocab = ['<PAD>', '<UNK>', '<EOS>']
        padding_idx = base_vocab.index('<PAD>')

        vln_bert = model_PREVALENT.VLNBERT()
        # 如果采用 分布式训练
        if args.distributed:
                if args.gpu is not None:
                        torch.cuda.set_device(args.gpu)
                        vln_bert.cuda(args.gpu)
                        
                        # args.batch_size = int(args.batch_size / ngpus_per_node)
                        # args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
                        vln_bert = torch.nn.parallel.DistributedDataParallel(vln_bert, device_ids=[args.gpu], find_unused_parameters=True)
                else:
                        vln_bert.cuda()
                        vln_bert = torch.nn.parallel.DistributedDataParallel(vln_bert)
        else:
                vln_bert.cuda(args.gpu)

        # critic = model_PREVALENT.Critic().cuda(args.gpu)
        optimizer = torch.optim.Adam(vln_bert.parameters(),args.lr)

        
        if args.load is not None:
                start_iter,vln_bert,optimizer = load(args.load,vln_bert,optimizer)
                print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))

        vln_bert.train()
        timer = {"batch_time":AverageMeter('Time', ':6.3f')}
        for iter in range(1, args.iters+1):
                # 在分布式模式下，需要在每个 epoch 开始时调用set_epoch()方法，然后再创建 DataLoader 迭代器，以使shuffle 操作能够在多个 epoch 中正常工作。 
                # 否则，dataloader迭代器产生的数据将始终使用相同的顺序，使得每个epoch在每个GPU上分割的数据集都是一样的
                if args.distributed:
                        train_sampler.set_epoch(iter)
                total_acc_mlm,total_acc_itm, total_action_loss= 0,0,0
                avg_acc_mlm,avg_acc_itm,avg_action_loss = 0,0,0
                step_mlm,step_itm,step_action = 0,0,0
                optimizer.zero_grad()

                # 设置时间
                batch_time = timer["batch_time"]
                end = time.time()
                for batch_step, batch_data in enumerate(train_loader):
                        if len(batch_data)==0:
                                continue                
                        prob_itm = np.random.random()
                        if prob_itm <= 0.25:
                                task = "mlm"
                                language = batch_data["mask_language"]
                                img = batch_data["traj"].cuda(args.gpu)
                        elif prob_itm <= 0.5:
                                task = "itm"
                                language = batch_data["language"]
                                img = batch_data["random_traj"].cuda(args.gpu)
                        else :
                                task = "action"
                                language = batch_data["language"]
                                img = batch_data["traj"].cuda(args.gpu)
                        language_attention_mask = (language != padding_idx).long().cuda(args.gpu)
                        token_type_ids = torch.zeros_like(language_attention_mask).long().cuda(args.gpu)                                               
                        # initial
                        language_inputs = {'mode':'language',
                        'sentence':       Variable(language, requires_grad=False).long().cuda(args.gpu),
                        'attention_mask': language_attention_mask,
                        'lang_mask':         language_attention_mask,  
                        'token_type_ids': token_type_ids}
                        # vln_bert = model_OSCAR.VLNBERT().cuda(args.gpu)

                        h_t, language_features = vln_bert(**language_inputs)
                        language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1) 

                        if task == "action":                       
                                visual_temp_mask=(utils.length2mask(batch_data["random_history_index"].tolist(),args.maxAction) == 0).long().cuda(args.gpu)
                        else :
                                visual_temp_mask=(utils.length2mask(batch_data["valid_length"].tolist(),args.maxAction) == 0).long().cuda(args.gpu)
                        visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1).cuda(args.gpu)

                        action =  batch_data["action"].cuda(args.gpu)
                        # vln_bert.module.vln_bert.config.directions = args.maxAction
                        visual_inputs = {'mode':      'visual',
                                'sentence':           language_features,
                                'attention_mask':     visual_attention_mask,
                                'lang_mask':          language_attention_mask,
                                'vis_mask':           visual_temp_mask,
                                'token_type_ids':     token_type_ids,
                                # 'action_feats':       input_a_t,
                                # 'pano_feats':         f_t,
                                'img':                img,
                                "action_feats":       action        
                                }
                        state_proj,attended_language,attended_visual,lang_output,visn_output,lang_output_pooler,visn_output_poller,action= vln_bert(**visual_inputs)      
                        
                        if task == 'mlm': 
                                prediction_scores = mlmhead(lang_output)                                
                                mask_loss = criterion(prediction_scores.view(-1,config.vocab_size), batch_data["mlm_label"].view(-1).cuda(args.gpu))
                                loss = mask_loss
                                bool_label = batch_data["mlm_label"] > 0
                                pred = prediction_scores[bool_label, :].argmax(1)
                                valid_labels = batch_data["mlm_label"][bool_label].cuda(args.gpu)
                                acc_mlm = (pred == valid_labels).type(torch.float).mean() * 100.
                                total_acc_mlm += acc_mlm
                                step_mlm = step_mlm + 1
                                avg_acc_mlm = total_acc_mlm /step_mlm
                        elif task == 'itm':
                                cls_part = lang_output_pooler * visn_output_poller
                                match_scores = is_match(cls_part)
                                match_loss = criterion(match_scores,batch_data["ismatch"].cuda(args.gpu)) * 5
                                loss = match_loss
                                correct = match_scores.argmax(dim=-1).eq(batch_data["ismatch"].cuda(args.gpu)).sum().item()
                                acc_itm = correct / batch_data["ismatch"].nelement() *100
                                total_acc_itm += acc_itm
                                step_itm = step_itm + 1
                                avg_acc_itm = total_acc_itm /(step_itm)
                        elif task == "action":
                                action_loss = criterion2(action,batch_data["action_label"].cuda(args.gpu)) * 1000
                                loss = action_loss
                                total_action_loss += action_loss
                                step_action +=1
                                avg_action_loss = total_action_loss/step_action
                        
                        loss.backward()
                        if args.distributed:
                                torch.nn.utils.clip_grad_norm(vln_bert.module.parameters(), 40.)
                        else:
                                torch.nn.utils.clip_grad_norm(vln_bert.parameters(), 40.)
                        optimizer.step()

                        # 计算时间
                        batch_time.update(time.time() - end)
                        end = time.time()
                        time_per_epoch = batch_time.avg * len(train_loader)
                        epochs_left = args.iters - iter - 1
                        batches_left = len(train_loader) - batch_step - 1

                        time_elapsed = sec_to_str(batch_time.sum)
                        time_left = sec_to_str(batches_left * batch_time.avg + epochs_left * time_per_epoch)
                        time_estimate = sec_to_str(args.iters * time_per_epoch)
                        # 打印一些东西
                        tmp_str = 'Epoch: [{}/{}] Batch: [{}/{}]  ' \
                                'Elapsed: {}  ' \
                                'ETA: {} / {}  '\
                                'acc_mlm: {}  ' \
                                'acc_itm: {} ' \
                                'action_loss:  {}'.format(iter + 1, args.iters, batch_step, len(train_loader), 
                        time_elapsed, time_left, time_estimate,acc_mlm, acc_itm, action_loss)

                        print(tmp_str)
                        
                # print("-------------------------------------------")
                # print("iter: "+str(iter)+" done!")        
                if iter % args.log_every ==0:
                        if args.distributed:
                                save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)),vln_bert.module,optimizer)
                        else:
                                save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)),vln_bert,optimizer)
                writer.add_scalar("avg_acc_mlm", avg_acc_mlm, iter)
                writer.add_scalar("avg_acc_itm", avg_acc_itm, iter)
                writer.add_scalar("avg_action_loss", avg_action_loss, iter)
                # writer.flush()
                
if __name__=="__main__":

        if args.dist_url == "env://" and args.world_size == -1:
                args.world_size = int(os.environ["WORLD_SIZE"])

        #如果没有设置数量，就自动检测
        ngpus_per_node = torch.cuda.device_count() if args.gpu_number==0 else args.gpu_number
        args.ngpus_per_node = ngpus_per_node
        # ngpus_per_node = 5
        if args.distributed:
                args.world_size = ngpus_per_node * args.world_size
                mp.spawn(main, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
                main(args.gpu, ngpus_per_node, args)     

