import functools
import t5
import torch
import transformers
if False and torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model_name="t5-base"
model_name="t5-tiny"
args = {
    "gradient_accumulation_steps" : 1,
    "fp16" : False,
    "train_batch_size" : 32,
    "max_seq_length" : 16,
    "output_seq_length" : 4,
    "gpu_memory_limit_gb" : 12,
    "partition_optimizer" : False,
    "allreduce_post_accumulation" : False,
    "learning_rate" : 6e-3,
    'max_steps' : 100
}
device = ort_supplement.setup_onnxruntime_with_mpi(args)

config = transformers.T5Config.from_json_file(json_file=model_name + "-config.json")
model = t5.models.HfPyTorchModel(config, "/tmp/hft5/", device)
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

