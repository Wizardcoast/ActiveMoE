## Unitests

Prepare data.
A checkpoint trained for 5 iterations test_ckpt. 
A checkpoint dir converted by huggingface_adaptor contains both hf and megatron models
find them in oss://inf-alpha/home/jiaran/temp/test

pytest checkpoint/test_resume.py
pytest checkpoint/test_convert_ckpt.py

## Test Scripts
### Resume training

A checkpoint contains model weights, optimizer states, and learning-rate scheduler.
There are several cases of resuming training including:
* Pretrain: training from scratch on the same dataset.
* Finetune: loading model weights and finetuning on new dataset.
* Further-train: loading model weight, optimizer states, and learning-rate scheduler, then training on new dataset.

Here are several testing scripts in `tests/unit_tests/resume`, to illustrate how to do pretraining, finetuning,
and further-pretrain.

#### Pretrain
```shell
# cd /path/to/tests/unit_tests/resume
sh pretrain.sh 2>&1 | tee pretrain.log

# Wait until the checkpoint saved, and continue running ~10 iterations,
# then ctrl-c to interrupt.
sh pretrain.sh 2>&1 | tee pretrain_resume.log

# You can compare the loss, learning-rate, iteration between the two
# logs, the expect values are identical in the next iteration after 
# loading checkpoint.
```

#### Finetune
```shell
# Add `--finetune --no-load-optim`, and set `--load`
# The iteration starts from 0
sh finetune.sh 2>&1 | tee finetune.log

# After the script interrupted, do resuming
# In the resume scripts, just delete `--finetune --no-load-optim`,
# and set the `--load` option to the directory of checkpoints.
sh finetune_resume.sh 2>&1 | tee finetune_resume.log
```

#### Further-train
```shell
# Add `--reset-sample-stat`, and set `--load`
# The iteration starts from 0, and lr-scheduler will be scheduled after
# loading checkpoint.
# TODO: you can use `--no-load-lr-scheduler` to update learning-rate schedule.
sh further_train.sh 2>&1 | tee further_train.log

# After script interrupted, resuming from checkpoints.
sh further_train_resume.sh 2>&1 | tee further_train_resume.log
```
