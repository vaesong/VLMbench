import argparse
import os
import torch

class Param:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="")

        # General
        self.parser.add_argument('--iters', type=int, default=500, help='training iterations')
        self.parser.add_argument('--name', type=str, default='default', help='experiment id')
        self.parser.add_argument('--vlnbert', type=str, default='prevalent', help='oscar or prevalent')

        # Data preparation
        self.parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 8)')
        self.parser.add_argument('--workers', type=int, default=8) #default=32
        self.parser.add_argument('--maxInput', type=int, default=80, help="max input instruction")
        self.parser.add_argument('--maxAction', type=int, default=12, help='Max Action sequence')
        self.parser.add_argument('--ignoreid', type=int, default=-100)
        self.parser.add_argument("--loadOptim",action="store_const", default=False, const=True)

        # Load the model from
        self.parser.add_argument("--load", default=None, help='path of the trained model')

        # Dropout Param
        self.parser.add_argument('--dropout', type=float, default=0.5)
        self.parser.add_argument('--featdropout', type=float, default=0.3)

        # Training Configurations
        self.parser.add_argument('--optim', type=str, default='rms')    # rms, adam
        self.parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")

        # Model hyper params:
        self.parser.add_argument("--action_repeat", type=int, default=16)
        self.parser.add_argument("--angle_feat_size", type=int, default=128)

        # A2C
        self.parser.add_argument("--gamma", default=0.9, type=float)
        self.parser.add_argument("--normalize", dest="normalize_loss", default="total", type=str, help='batch or total')

        # vlmbench
        self.parser.add_argument('--data_dir',default="/home/liuchang/DATA/rlbench_data" ,type=str)
        self.parser.add_argument('--setd', type=str, default='train')
        self.parser.add_argument('--img_size',nargs='+', type=int, default=[360, 360])

        self.parser.add_argument('--preprocess', action='store_true', 
                help="whether preprocess the data. Next time can directly use. Add if you don't want it.")
        self.parser.add_argument('--unused_camera_list', nargs='+',default=['left_shoulder', 'right_shoulder', 'front','wrist'])
        # default=['left_shoulder', 'right_shoulder', 'overhead','wrist','front']
        self.parser.add_argument('--use_fail_cases', action='store_true', help="add if use the fail cases")
        self.parser.add_argument('--sample_numbers', type=int, default=0, help="downsample from total demonstrations")
        self.parser.add_argument('--pin_memory', action='store_true', help="do not use if the RAM is small")
        self.parser.add_argument('--train_tasks', nargs='+', type=str, default =None)
        self.parser.add_argument('--relative', type=lambda x:bool(strtobool(x)), default=False)
        self.parser.add_argument('--renew_obs', type=lambda x:bool(strtobool(x)), default=False)
        self.parser.add_argument('--add_low_lang', type=lambda x:bool(strtobool(x)), default=True)
        #traning
        self.parser.add_argument('--start_epoch', default=0, type=int)
        self.parser.add_argument('--log_every', default=25, type=int,
                                help='Print log message at this many iterations (default: 10)')
        self.parser.add_argument('--log-freq', default=1, type=int,
                                help='Print log message at this many iterations (default: 1)')
        self.parser.add_argument('--gpu', default=1, type=int,
                                help='GPU id to use.')
        self.parser.add_argument('--checkpoint_path', default='/home/zp_3c/liuchang/vlmbench/weights', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
        self.parser.add_argument('--resume', default= None, type=str,
                        help='resume training from checkpoint-path/model-best.pth')
        self.parser.add_argument('--baseline_mode', type=str, default='cliport_6dof')
        self.parser.add_argument('--wandb_entity', type=str, default=None, help="visualize the training. Account Name")
        self.parser.add_argument('--wandb_project', type=str, default=None,  help="visualize the training. Project Name")

        #distributed training
        self.parser.add_argument('--world-size', default=1, type=int,
                help='number of nodes for distributed training')
        self.parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
        self.parser.add_argument('--dist-url', default='tcp://127.0.0.1:23459', type=str,
                        help='url used to set up distributed training')
        self.parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
        self.parser.add_argument('--distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                                'N processes per node, which has N GPUs. This is the '
                                'fastest way to use PyTorch for either single node or '
                                'multi node data parallel training')
        self.parser.add_argument('--gpu_number', type=int, default=2)
        self.parser.add_argument('--gpu_start', type=int, default=0)

        # HOP
        self.parser.add_argument("--model_type", default="bert", type=str,
                                help="The model architecture to be fine-tuned.")
        self.parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                                help="The model checkpoint for weights initialization.")
        self.parser.add_argument("--mlm", action='store_true',
                                help="Train with masked-language modeling loss instead of language modeling.")
        self.parser.add_argument("--mlm_probability", type=float, default=0.15,
                                help="Ratio of tokens to mask for masked language modeling loss")
        self.parser.add_argument("--config_name", default="", type=str,
                                help="Optional pretrained config name or path if not the same as model_name_or_path")
        self.parser.add_argument("--tokenizer_name", default="", type=str,
                                help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
        self.parser.add_argument("--cache_dir", default="", type=str,
                                help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
        self.parser.add_argument("--block_size", default=-1, type=int,
                                help="Optional input sequence length after tokenization."
                                "The training dataset will be truncated in block of this size for training."
                                "Default to the model max input length for single sentence inputs (take into account special tokens).")
        self.parser.add_argument("--do_train", action='store_true',
                                help="Whether to run training.")
        self.parser.add_argument("--do_eval", action='store_true',
                                help="Whether to run eval on the dev set.")
        self.parser.add_argument("--do_trainval", action='store_true',
                                help="Whether to run eval when training.")
        self.parser.add_argument("--evaluate_during_training", action='store_true',
                                help="Run evaluation during training at each logging step.")
        self.parser.add_argument("--do_lower_case", action='store_true',
                                help="Set this flag if you are using an uncased model.")
        self.parser.add_argument("--per_gpu_train_batch_size", default=1, type=int,
                                help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=1, type=int,
                                help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                                help="Number of updates steps to accumulate before performing a backward/update pass.")
        self.parser.add_argument("--learning_rate", default=5e-5, type=float,
                                help="The initial learning rate for Adam.")
        self.parser.add_argument("--weight_decay", default=0.0, type=float,
                                help="Weight deay if we apply some.")
        self.parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                                help="Epsilon for Adam optimizer.")
        self.parser.add_argument("--max_grad_norm", default=1.0, type=float,
                                help="Max gradient norm.")
        self.parser.add_argument("--num_train_epochs", default=1.0, type=float,
                                help="Total number of training epochs to perform.")
        self.parser.add_argument("--max_steps", default=-1, type=int,
                                help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
        self.parser.add_argument("--warmup_steps", default=0, type=int,
                                help="Linear warmup over warmup_steps.")
        self.parser.add_argument('--logging_steps', type=int, default=50,
                                help="Log every X updates steps.")
        self.parser.add_argument('--save_steps', type=int, default=50,
                                help="Save checkpoint every X updates steps.")
        self.parser.add_argument("--eval_all_checkpoints", action='store_true',
                                help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
        self.parser.add_argument("--no_cuda", action='store_true',
                                help="Avoid using CUDA when available")
        self.parser.add_argument('--overwrite_output_dir', action='store_true',
                                help="Overwrite the content of the output directory")
        self.parser.add_argument('--overwrite_cache', action='store_true',
                                help="Overwrite the cached training and evaluation sets")
        self.parser.add_argument('--seed', type=int, default=42,
                                help="random seed for initialization")
        self.parser.add_argument('--fp16', action='store_true',
                                help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
        self.parser.add_argument('--fp16_opt_level', type=str, default='O1',
                                help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                                "See details at https://nvidia.github.io/apex/amp.html")
        self.parser.add_argument("--local_rank", type=int, default=-1,  #-1
                                help="For distributed training: local_rank")
        self.parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
        self.parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
        self.parser.add_argument("--vision_size", type=int, default=2176,help="imgaction size")
        self.parser.add_argument("--action_space", type=int, default=36,help="action space")
        self.parser.add_argument("--vl_layers", type=int, default=4,help="how many fusion layers")
        self.parser.add_argument("--la_layers", type=int, default=9,help="how many lang layers")
        self.parser.add_argument('--update', type=bool, default=True, help='update lang Bert')
        self.parser.add_argument('--update_add_layer', type=bool, default=True, help='update add layer')
        self.parser.add_argument('--include_next', type=bool, default=True, help='do action classification')
        self.parser.add_argument('--result_dir', type=str, default='tasks/R2R/results/', help='path to the result_dir file')
        self.parser.add_argument('--plot_dir', type=str, default='tasks/R2R/plots/', help='path to the plot_dir file')
        self.parser.add_argument('--snapshot_dir', type=str, default='tasks/R2R/snapshots/', help='path to the snapshot_dir file')
        self.parser.add_argument('--philly', action='store_true', help='program runs on Philly, used to redirect `write_model_path`')
        self.parser.add_argument("--resume_path", default=None, type=str,
                                help="The model checkpoint for weights initialization.") 
        self.parser.add_argument('-r', '--run_name', type=str, help="name for wandb run")
        self.parser.add_argument("--prevalent_only", type=bool, default=False)

        self.args = self.parser.parse_args()

        if self.args.optim == 'rms':
            print("Optimizer: Using RMSProp")
            self.args.optimizer = torch.optim.RMSprop
        elif self.args.optim == 'adam':
            print("Optimizer: Using Adam")
            self.args.optimizer = torch.optim.Adam
        elif self.args.optim == 'adamW':
            print("Optimizer: Using AdamW")
            self.args.optimizer = torch.optim.AdamW
        elif self.args.optim == 'sgd':
            print("Optimizer: sgd")
            self.args.optimizer = torch.optim.SGD
        else:
            assert False

param = Param()
args = param.args

args.description = args.name
args.IMAGENET_FEATURES = 'img_features/ResNet-152-imagenet.tsv'
args.log_dir = 'snap/%s' % args.name

if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
DEBUG_FILE = open(os.path.join('snap', args.name, "debug.log"), 'w')
