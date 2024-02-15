import torch
rank0_checkpoint_path = "/workspace/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb256_gas1/exp1.1/checkpoint/iter_0022000/mp_rank_00/model_optim_rng.pt"
rank0_checkpoint_path= "/workspace/checkpoints/hkg_7b_nl_tp1_pp1_mb1_gb512_gas2/exp2.1/checkpoint/iter_0074000/mp_rank_00/model_optim_rng.pt"
state_dict = torch.load(rank0_checkpoint_path, map_location="cpu")
# print(state_dict)
# print([dict.keys() for dict in state_dict])
for k ,v in state_dict.items():
    print(k)
    print(v)
    # print(dict[0])
    # print(dict.value())