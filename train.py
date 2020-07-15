import functools
import t5
import torch
import transformers
if False and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Args:
    def __init__(self):
        self.gradient_accumulation_steps = 1
        self.fp16 = True
        self.train_batch_size = 32
        self.max_seq_length = 64
        self.output_seq_length = 4
        self.gpu_memory_limit_gb = 12
        self.partition_optimizer = False
        self.allreduce_post_accumulation = False
        self.learning_rate = 6e-3
        self.max_steps = 100
        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1
        self.schedule='warmup_poly'
        self.loss_scale = 0.0
        self.warmup_proportion = 0.2843


model_name="t5-base"
#model_name="t5-tiny"

args = Args()

print(args.schedule)
from t5.models.ort_supplement import *
device = setup_onnxruntime_with_mpi(args)

config = transformers.T5Config.from_json_file(json_file=model_name + "-config.json")
model = t5.models.HfPyTorchModel(config, "/tmp/hft5/", device, args)
# Evaluate the pre-trained checkpoint, before further fine-tuning
#model.eval(
#    "glue_cola_v002",
#    sequence_length={"inputs": 64, "targets": 4},
#    batch_size=128,
#)


# Run 1000 steps of fine-tuning
model.train(
    mixture_or_task_name="glue_cola_v002",
    steps=1000,
    save_steps=100,
    sequence_length={"inputs": 64, "targets": 4},
    split="train",
    batch_size=32,
    optimizer=functools.partial(transformers.AdamW, lr=1e-4),
    model_name=model_name,
)

