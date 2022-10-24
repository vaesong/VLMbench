import numpy as np
import torch

# Identify way-point in each RLBench Demo
def _is_stopped(demo, i, obs, stopped_buffer):
    next_is_not_final = i == (len(demo) - 2)
    gripper_state_no_change = i < (len(demo) - 2) and (
        obs.gripper_open == demo[i + 1].gripper_open
        and obs.gripper_open == demo[i - 1].gripper_open
        and demo[i - 2].gripper_open == demo[i - 1].gripper_open
    )
    small_delta = np.allclose(obs.joint_velocities, 0, atol=0.1)
    stopped = (
        stopped_buffer <= 0
        and small_delta
        and (not next_is_not_final)
        and gripper_state_no_change
    )
    return stopped

def keypoint_discovery(demo):
    episode_keypoints = []
    prev_gripper_open = demo[0].gripper_open
    stopped_buffer = 0
    for i, obs in enumerate(demo):
        stopped = _is_stopped(demo, i, obs, stopped_buffer)
        stopped_buffer = 4 if stopped else stopped_buffer - 1
        # If change in gripper, or end of episode.
        last = i == (len(demo) - 1)
        if i != 0 and (obs.gripper_open != prev_gripper_open or last or stopped):
            episode_keypoints.append(i)
        prev_gripper_open = obs.gripper_open
    if (
        len(episode_keypoints) > 1
        and (episode_keypoints[-1] - 1) == episode_keypoints[-2]
    ):
        episode_keypoints.pop(-2)
    # HACK for tower3 task
    return episode_keypoints

def max_sperate_index(inputs):

    max = 0
    for i in range(len(inputs)):
        if i > 0:# 保证有两个数
            if inputs[i] - inputs[i-1] > max:
                max = inputs[i] - inputs[i-1]
                max_index = i
    return max_index

def mask_tokens(inputs, tokenizer):

    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    labels = inputs.clone()

    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    #probability_matrix = torch.full(labels.shape, args.mlm_probability)
    probability_matrix = torch.full(labels.shape, 0.15)    
    special_tokens_mask = [val in tokenizer.all_special_ids for val in labels.tolist()]#特殊字符
    # att_mask = [val == tokenizer.pad_token_id for val in labels.tolist()]#是否是att_mask
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)#probabilitu_matrix中是special_tokens_mask的位置进行填充
    #masked_indices = torch.bernoulli(torch.full(labels.shape, args.mlm_probability)).type(torch.ByteTensor)
    masked_indices = torch.bernoulli(probability_matrix).type(torch.ByteTensor)#选出需要mask的下标

    # attention_mask = torch.full(labels.shape, 1).masked_fill_(torch.tensor(att_mask, dtype=torch.bool), value=0)
    #先用1填充然后把是padding的位置全部变为0
    labels[1-masked_indices] = -1  # We only compute loss on masked tokens


    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).type(torch.ByteTensor) & masked_indices

    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)#%15里面的%80被mask掉



    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).type(torch.ByteTensor) & masked_indices & ~indices_replaced
    #剩下的位置被随机

    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)

    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def length2mask(length, size=None):
    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
                > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
    return mask



