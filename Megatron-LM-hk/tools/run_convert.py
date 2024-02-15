import subprocess

# Paths configuration
megatron_save_root = "/workspace/megatron/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb512_gas2/exp2.1/checkpoint/"
hf_save_root = "/workspace/hf_ckpt/amber"

# Base iteration and multiplier range
base_iter = 48000
multipliers = range(1, 9)  # From 1 to 8

# Loop through each multiplier
for multiplier in multipliers:
    iter_val = base_iter * multiplier
    # Construct the MEGATRON_PATH dynamically based on ITER
    formatted_iter = "{:06}".format(iter_val)
    megatron_path = f"{megatron_save_root}/iter_0{formatted_iter}"

    # Command to execute your script
    command = f"""
    MEGATRON_SAVE_ROOT={megatron_save_root}
    HF_SAVE_ROOT={hf_save_root}
    ITER={iter_val}
    MEGATRON_PATH={megatron_path}
    bash convert_llama.sh {formatted_iter} {hf_save_root} {megatron_path}
    """
    print(command)

    # Execute the command
    process = subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    print(f"Completed iteration: {iter_val}")
