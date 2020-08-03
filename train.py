import functools
import t5
import torch
import transformers
import tensorflow.compat.v1 as tf

class Args:
    def __init__(self):
        self.gradient_accumulation_steps = 1
        self.fp16 = True
        self.train_batch_size = 1 #32

        # aligned with https://github.com/google-research/text-to-text-transfer-transformer/blob/75fe4d160137d06ebfd7b08a1164468ad15e0251/t5/models/mesh_transformer_main.py#L80
        self.input_sequence_length = 512
        self.target_sequence_length = 512

        self.gpu_memory_limit_gb = 16
        self.deepspeed_zero_stage = 0 #1
        self.learning_rate = 1e-4

        self.max_steps = 8
        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1
        self.schedule='warmup_poly'
        self.warmup_proportion = 0.2843
        self.allreduce_post_accumulation = True
        self.horizontal_parallel_size = 2
        self.data_parallel_size = 1
        self.pipeline_parallel_size = 1

tf.config.set_visible_devices([], 'GPU') 
model_name="t5-base"
model_name="t5-tiny"
model_name="t5-small"

#model_name="t5-11b"
args = Args()

from t5.models.ort_supplement import *

# tf.get_default_session()
#  config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     K.set_session(tf.Session(config = config))
config = transformers.T5Config.from_json_file(json_file=model_name + "-config.json")
device = torch.device("cpu") # don't need use GPU for pytorch, in case its weight are stored on GPU
model = t5.models.HfPyTorchModel(config, "/tmp/hft5_" + str(os.environ['OMPI_COMM_WORLD_RANK']) + "/", device, args)
# Evaluate the pre-trained checkpoint, before further fine-tuning
#model.eval(
#    "glue_cola_v002",
#    sequence_length={"inputs": 64, "targets": 4},
#    batch_size=128,
#)


# Run 1000 steps of fine-tuning
model.train(
    mixture_or_task_name="glue_cola_v002",
    steps=64,
    save_steps=32,
    sequence_length={"inputs": 64, "targets": 4},
    split="train",
    #batch_size=32,
    batch_size=1,
    optimizer=functools.partial(transformers.AdamW, lr=1e-4),
    model_name=model_name,
)

