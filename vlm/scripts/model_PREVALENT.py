# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from param import args
import math
from vlnbert.vlnbert_init import get_vlnbert_models
import einops
import torchvision.models as models
import numpy as np

class VLNBERT(nn.Module):
    def __init__(self, feature_size=2048+128):
        super(VLNBERT, self).__init__()
        print('\nInitalizing the VLN-BERT model ...')

        self.vln_bert = get_vlnbert_models(args, config=None)  # initialize the VLN-BERT
        self.vln_bert.config.directions = 4  # a preset random number

        hidden_size = self.vln_bert.config.hidden_size
        layer_norm_eps = self.vln_bert.config.layer_norm_eps

        self.action_state_project = nn.Sequential(
            nn.Linear(hidden_size+args.angle_feat_size, hidden_size), nn.Tanh())
        self.action_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.bert_size = 768
        self.action_size = 7
        self.fc = nn.Sequential(
            nn.Linear(768, 128), 
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32,8),
            )
        
        # self.mlm_fc = nn.Sequential(
        #     nn.Linear(768, 128), 
        #     nn.ReLU(),
        #     nn.Linear(128,32),
        #     nn.ReLU(),
        #     nn.Linear(32,8),
        #     )
        
        self.drop_env = nn.Dropout(p=args.featdropout)
        self.img_projection = nn.Linear(feature_size, hidden_size, bias=True)
        self.cand_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)

        self.vis_lang_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.state_proj = nn.Linear(hidden_size*2, hidden_size, bias=True)
        self.state_LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        
        self.vision_encoder = VisionEncoder(self.vln_bert.config.img_feature_dim, self.vln_bert.config)
                # resnet 152
        resnet152 = models.resnet152(pretrained=True)
        resnet152.fc = nn.Linear(2048,2048)
        for parm in resnet152.parameters():
                parm.requires_grad = False 
        self.resnet152=resnet152.cuda()


    def forward(self, mode, sentence, token_type_ids=None,
                attention_mask=None, lang_mask=None, vis_mask=None,
                position_ids=None, action_feats=None, pano_feats=None, img=None,task=None):

        if mode == 'language':
            init_state, encoded_sentence = self.vln_bert(mode, sentence, attention_mask=attention_mask, lang_mask=lang_mask)
            return init_state, encoded_sentence

        elif mode == 'visual':
            

            # state_action_embed = torch.cat((sentence[:,0,:], action_feats), 1)
            # state_action_embed =sentence[:,0,:]
            # state_with_action = self.action_state_project(state_action_embed)
            # state_with_action = self.action_LayerNorm(state_with_action)
            # state_feats = torch.cat((state_with_action.unsqueeze(1), sentence[:,1:,:]), dim=1)
            state_feats = sentence
            candidate_feat=[]
            for i in range(args.batch_size):
                    rgb = img[i]
                    rgb = rgb.permute(0,3,1,2).float() 
                    img_feat = self.resnet152(rgb.cuda()).cpu().data.numpy()
                    action_feat = np.repeat(action_feats[i], 16, axis=1).numpy()
                    img_feat = np.concatenate((img_feat,action_feat),axis=1) #repeat 8 dim action 16times to 128
                    candidate_feat.append(img_feat)      
            candidate_feat = torch.from_numpy(np.array(candidate_feat)).cuda() 
            img_feats = self.vision_encoder(candidate_feat.float())
            # if task == "action":
            #     input_tensor = einops.rearrange(x, "b t k -> (b t) k")
            #     ctx_tensor = einops.rearrange(x, "b t k -> (b t) k")
            #     ctx_tensor = einops.repeat(ctx_tensor, "(b t) k -> (b t) (tp k)", tp=T, t=T)
            #     state_feats = torch.cat([state_feats, img_feats], 1).cuda()
            #     lang_mask = attention_mask
            #     attention_mask = torch.cat((lang_mask, vis_mask), dim=-1).cuda()
            # cand_feats[..., :-args.angle_feat_size] = self.drop_env(cand_feats[..., :-args.angle_feat_size])
            # logit is the attention scores over the candidate features
            lang_output_pooler,visn_output_poller, logit, attended_language, attended_visual,lang_output,visn_output = self.vln_bert(mode, state_feats,
                attention_mask=attention_mask, lang_mask=lang_mask, vis_mask=vis_mask, img_feats=img_feats)

            # update agent's state, unify history, language and vision by elementwise product
            vis_lang_feat = self.vis_lang_LayerNorm(attended_language * attended_visual)
            state_output = torch.cat((lang_output_pooler, vis_lang_feat), dim=-1)
            state_proj = self.state_proj(state_output)
            state_proj = self.state_LayerNorm(state_proj)
            action = self.fc(state_proj)
             
            return state_proj,attended_language,attended_visual,lang_output,visn_output,lang_output_pooler,visn_output_poller,action
            # return state_proj, logit

        else:
            ModuleNotFoundError

class VisionEncoder(nn.Module):
    def __init__(self, vision_size, config):
        super().__init__()
        feat_dim = vision_size

        # Object feature encoding
        self.visn_fc = nn.Linear(feat_dim, config.hidden_size)
        self.visn_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, visn_input):
        feats = visn_input

        x = self.visn_fc(feats)
        x = self.visn_layer_norm(x)

        output = self.dropout(x)
        return output

class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()









