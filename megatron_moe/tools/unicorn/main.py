import argparse
from unicorn import (
    convert_checkpoint_from_megatron_to_transformers,
    convert_checkpoint_from_transformers_to_megatron
)
from unicorn import (
    add_transformers_checkpoint_args,
    add_megatron_checkpoint_args,
    add_checkpointing_args
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_checkpointing_args(parser)
    parser = add_transformers_checkpoint_args(parser)
    parser = add_megatron_checkpoint_args(parser)
    args = parser.parse_args()

    if args.convert_checkpoint_from_megatron_to_transformers:
        print("=> convert megatron to transformers ...")
        convert_checkpoint_from_megatron_to_transformers(args)
    else:
        print("=> convert transformers to megatron ...")
        convert_checkpoint_from_transformers_to_megatron(args)
