from datasets import load_dataset

ds = load_dataset("imagenet-1k")
train_ds = ds["train"]
