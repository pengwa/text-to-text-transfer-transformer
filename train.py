import os
from filelock import Timeout, FileLock
from os import path
import socket



# host_name = socket.gethostname()
# lock = FileLock(host_name + ".pkg_installed.txt.lock")
# lock.acquire()

# if path.exists(host_name + ".pkg_installed.txt"):
#     print("already installed ")
# else:
#     cwd = os.system("pwd")
#     print("start install python pkg from ", cwd)
#     os.system("pip uninstall -y onnxruntime_gpu")
#     os.system("pip install " + str(os.environ['T5_MODEL_PATH']) + "/ort/onnxruntime_gpu-1.4.0-cp37-cp37m-linux_x86_64.whl --target /workspace/python_pkgs")
#     os.system("pip install . --target /workspace/python_pkgs")
#     os.system("echo abc > " + host_name + ".pkg_installed.txt")
#     os.system("pip show onnxruntime_gpu")
#     os.system("pip show t5")
#     os.system("pip show transformers")
# lock.release()

import functools
import t5
import torch
import transformers
import tensorflow.compat.v1 as tf
import csv
import os
import sys
import time
import argparse
import random
import logging
from t5.models.ort_supplement import *

def parse_arguments():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--max_input_seq_length",
                        default=512,
                        type=int,
                        help="")
    parser.add_argument("--max_output_seq_length",
                        default=512,
                        type=int,
                        help="")
    parser.add_argument("--train_batch_size",
                        default=1,
                        type=int,
                        help="Per GPU batch size for training.")
    parser.add_argument("--gradient_accumulation_steps",
                        default=1,
                        type=int,
                        help=".")
    parser.add_argument("--horizontal_parallel_size",
                        default=1,
                        type=int,
                        help=".")
    parser.add_argument("--data_parallel_size",
                        default=1,
                        type=int,
                        help=".")
    parser.add_argument("--pipeline_parallel_size",
                        default=1,
                        type=int,
                        help=".")
    parser.add_argument("--learning_rate",
                        default=1e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps",
                        default=10,
                        type=float,
                        help="Total number of training steps to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.2843,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--allreduce_post_accumulation',
                        default=False,
                        action='store_true',
                        help="--use_nccl.")
    parser.add_argument('--use_ib',
                        default=False,
                        action='store_true',
                        help="Whether to use infiniband on Azure ML submission.")
    parser.add_argument('--deepspeed_zero_stage',
                        type=int,
                        default=0,
                        help="Whether ORT will partition optimizer.")
    parser.add_argument("--gpu_memory_limit_gb",
                        type=int,
                        default=16,
                        help="GPU memory limit in GBs")
    parser.add_argument('--schedule',
                        default='warmup_poly',
                        type=str)
    parser.add_argument('--model_name',
                        default='t5-base',
                        type=str)
    parser.add_argument('--tensorboard_dir',
                        default='./outputs',
                        type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(1)
    tf.logging.set_verbosity(tf.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    logging.getLogger('tensorflow').setLevel(logging.FATAL)
    user_folder_path = "/TEE2-lustre/pengwa" # #'/home/pengwa'
    os.environ['T5_MODEL_PATH'] = user_folder_path
    os.environ['T5_MODEL_NAME'] = args.model_name

    tf.config.set_visible_devices([], 'GPU') 

    print("=========================use model name: ", args.model_name)
    # tf.get_default_session()
    #  config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     K.set_session(tf.Session(config = config))
    config = transformers.T5Config.from_json_file(json_file=args.model_name + "-config.json")
    device = torch.device("cpu") # don't need use GPU for pytorch, in case its weight are stored on GPU

    model = t5.models.HfPyTorchModel(config, "/tmp/hft5_" + str(os.environ['OMPI_COMM_WORLD_RANK']) + "/", device, args)
    # Evaluate the pre-trained checkpoint, before further fine-tuning
    #model.eval(
    #    "glue_cola_v002",
    #    sequence_length={"inputs": 64, "targets": 4},
    #    batch_size=128,
    #)

    print("After building PyTroch model")
    # Run 1000 steps of fine-tuning
    model.train(
        mixture_or_task_name="glue_cola_v002",
        steps=args.max_steps,
        save_steps=32,
        sequence_length={"inputs": args.max_input_seq_length, "targets": args.max_output_seq_length},
        split="train",
        #batch_size=32,
        batch_size=1,
        optimizer=functools.partial(transformers.AdamW, lr=1e-4),
        model_name=args.model_name,
    )


if __name__ == "__main__":
    print("======================in t5 train.py==================")
    main()