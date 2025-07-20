import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl', init_method='env://')

def check_communication():
    tensor = torch.ones(1).cuda()
    dist.all_reduce(tensor)
    print(f"Rank {dist.get_rank()} has tensor value: {tensor[0].item()}")

if __name__ == "__main__":
    check_communication()