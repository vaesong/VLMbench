import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
# from env import R2RBatch
#import utils
#from utils import padding_idx, print_progress
# import model_PREVALENT
import os
from pickle import NONE
import numpy as np
import torch
# from vlm.scripts.VLDataloader_custom import VLM_dataset
from pytorch_transformers import (BertConfig)
import utils
from tensorboardX import SummaryWriter
from pytorch_transformers import BertConfig
from pytorch_transformers.modeling_bert import BertOnlyMLMHead
# from param import args
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
                state = model.state_dict()
                model_keys = set(state.keys())
                load_keys = set(states[name]['state_dict'].keys())
                if model_keys != load_keys:
                        print("NOTICE: DIFFERENT KEYS IN THE LISTEREN")
                        state.update(states[name]['state_dict'])
                        model.load_state_dict(state)
                        # if args.loadOptim:
                        #         optimizer.load_state_dict(states[name]['optimizer'])
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

def main(args):
        train_dataset = VLM_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_numbers = args.sample_numbers, args=args)
        train_loader = torch.utils.data.DataLoader(  
                train_dataset, batch_size=args.batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True, sampler=None, 
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
        criterion2 = nn.SmoothL1Loss()
         
        start_iter = 0
        
        config = BertConfig.from_pretrained("/home/liuchang/projects/VLMbench/VLMbench/vlm/scripts/base-no-labels/ep_67_588997")
        config.img_feature_dim = args.vision_size
        config.img_feature_type = ""
        config.update_lang_bert = args.update
        config.update_add_layer = args.update_add_layer
        config.vl_layers = args.vl_layers
        config.la_layers = args.la_layers
        config.action_space = args.action_space

        mlmhead = BertOnlyMLMHead(config).cuda()
        is_match = NextActionPrediction(config.hidden_size, 2).cuda()

        base_vocab = ['<PAD>', '<UNK>', '<EOS>']
        padding_idx = base_vocab.index('<PAD>')

        vln_bert = model_PREVALENT.VLNBERT().cuda()
        # critic = model_PREVALENT.Critic().cuda()
        optimizer = torch.optim.Adam(vln_bert.parameters(),args.lr)

        
        if args.load is not None:
                start_iter,vln_bert,optimizer = load(args.load,vln_bert,optimizer)
                print("\nLOAD the model from {}, iteration ".format(args.load, start_iter))

        vln_bert.train()

        for iter in range(1, args.iters+1):
                total_acc_mlm,total_acc_itm, total_action_loss= 0,0,0
                avg_acc_mlm,avg_acc_itm,avg_action_loss = 0,0,0
                step_mlm,step_itm,step_action = 0,0,0
                optimizer.zero_grad()
                for batch_step, batch_data in enumerate(train_loader):
                        if len(batch_data)==0:
                                continue                
                        prob_itm = np.random.random()
                        if prob_itm <= 0.25:
                                task = "mlm"
                                language = batch_data["mask_language"]
                                img = batch_data["traj"]
                        elif prob_itm <= 0.5:
                                task = "itm"
                                language = batch_data["language"]
                                img = batch_data["random_traj"]
                        else :
                                task = "action"
                                language = batch_data["language"]
                                img = batch_data["traj"]
                        language_attention_mask = (language != padding_idx).long().cuda()
                        token_type_ids = torch.zeros_like(language_attention_mask).long().cuda()                                                  
                        # initial
                        language_inputs = {'mode':'language',
                        'sentence':       Variable(language, requires_grad=False).long().cuda(),
                        'attention_mask': language_attention_mask,
                        'lang_mask':         language_attention_mask,  
                        'token_type_ids': token_type_ids}
                        # vln_bert = model_OSCAR.VLNBERT().cuda()    

                        h_t, language_features = vln_bert(**language_inputs)
                        language_features = torch.cat((h_t.unsqueeze(1), language_features[:,1:,:]), dim=1) 

                        if task == "action":                       
                                visual_temp_mask=(utils.length2mask(batch_data["random_history_index"].tolist(),args.maxAction) == 0).long()
                        else :
                                visual_temp_mask=(utils.length2mask(batch_data["valid_length"].tolist(),args.maxAction) == 0).long()
                        visual_attention_mask = torch.cat((language_attention_mask, visual_temp_mask), dim=-1).cuda()

                        action =  batch_data["action"]
                        vln_bert.vln_bert.config.directions = args.maxAction
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
                                mask_loss = criterion(prediction_scores.view(-1,config.vocab_size), batch_data["mlm_label"].view(-1).cuda())
                                loss = mask_loss
                                bool_label = batch_data["mlm_label"] > 0
                                pred = prediction_scores[bool_label, :].argmax(1)
                                valid_labels = batch_data["mlm_label"][bool_label].cuda()   
                                acc_mlm = (pred == valid_labels).type(torch.float).mean() * 100.
                                total_acc_mlm += acc_mlm
                                step_mlm = step_mlm + 1
                                avg_acc_mlm = total_acc_mlm /step_mlm
                        elif task == 'itm':
                                cls_part = lang_output_pooler * visn_output_poller
                                match_scores = is_match(cls_part)
                                match_loss = criterion(match_scores,batch_data["ismatch"].cuda()) * 5
                                loss = match_loss
                                correct = match_scores.argmax(dim=-1).eq(batch_data["ismatch"].cuda()).sum().item()
                                acc_itm = correct / batch_data["ismatch"].nelement() *100
                                total_acc_itm += acc_itm
                                step_itm = step_itm + 1
                                avg_acc_itm = total_acc_itm /(step_itm)
                        elif task == "action":
                                action_loss = criterion2(action,batch_data["action_label"].cuda()) * 100
                                loss = action_loss
                                total_action_loss += action_loss
                                step_action +=1
                                avg_action_loss = total_action_loss/step_action
                        print("batch_step "+str(batch_step)+" done!")
                        loss.backward()
                        torch.nn.utils.clip_grad_norm(vln_bert.parameters(), 40.)
                        optimizer.step()  
                        
                if iter % args.log_every ==0:
                        save(iter, os.path.join("snap", args.name, "state_dict", "LAST_iter%d" % (iter)),vln_bert,optimizer)
                writer.add_scalar("avg_acc_mlm", avg_acc_mlm, iter)
                writer.add_scalar("avg_acc_itm", avg_acc_itm, iter)
                writer.add_scalar("avg_action_loss", avg_action_loss, iter)
                # writer.flush()
                
if __name__=="__main__":
        main(args)        

