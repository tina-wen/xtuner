from typing import List, Tuple
import os
import torch


lib = torch.library.Library("fsdp", "FRAGMENT")


@torch.library.impl(lib, "chunk_cat", "PrivateUse1")
def chunk_cat(
    tensors: List[torch.Tensor],
    dim: int,
    num_chunks: int,
    out: torch.Tensor,
) -> None:
    tensors = [tensor.contiguous() for tensor in tensors]
    out = out.contiguous()
    torch._chunk_cat(tensors, dim, num_chunks, out=out)


@torch.library.impl(lib, "all_gather_copy_in", "PrivateUse1")
def all_gather_copy_in_npu(
    all_gather_inputs: List[torch.Tensor],
    inp_split_sizes: List[int],
    all_gather_input_numel: int,
    world_size: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    all_gather_output = torch.empty(
        (all_gather_input_numel * world_size,), dtype=dtype, device=device
    )
    all_gather_input = all_gather_output.narrow(
        0, all_gather_input_numel * rank, all_gather_input_numel
    )
    foreach_copy_dsts = torch.split(all_gather_input, inp_split_sizes)
    with torch.no_grad():
        if foreach_copy_dsts[0].device == all_gather_inputs[0].device:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs, non_blocking=True)
        else:
            torch._foreach_copy_(foreach_copy_dsts, all_gather_inputs)
    return all_gather_input, all_gather_output

@torch.library.impl(lib, "split_with_sizes_copy", "PrivateUse1")
def split_with_sizes_copy(
    all_gather_output: torch.Tensor,
    all_gather_input_split_sizes: List[int],
    dim: int,
    out: List[torch.Tensor],
    num_expert: int = 128,
    hidden_size: int = 4096,
    moe_intermediate_size: int = 1536,
) -> None:
    # 当且仅当满足如下条件，才启用gmm_nz优化
    # 1. 打开GROUPMM_NZ_TRANSPOSE开关
    # 2. all_gather_input_split_sizes长度大于1
    # 3. 切分后的最后一个权重用于GMM down_proj
    enable_gmm_nz = int(os.getenv("GROUPMM_NZ_TRANSPOSE","0")) \
            and len(all_gather_input_split_sizes) > 1 \
            and out[-1].shape[0] * out[-1].shape[1] == num_expert * hidden_size * moe_intermediate_size
    
    if enable_gmm_nz:
        from special_op import npu_special_slice
        num_rank = out[0].shape[0]
        total_size = sum(all_gather_input_split_sizes)

        # 切分后最后两个权重用于GMM up_proj和down_proj
        up_size = out[-1].shape[1]
        down_size = out[-2].shape[1]

        up_start = total_size - up_size
        down_start = up_start - down_size

        out[-1].resize_(num_expert,moe_intermediate_size,hidden_size)
        out[-2].resize_(num_expert,hidden_size,moe_intermediate_size*2)

        # GMM权重切分和转NZ使用融合算子
        npu_special_slice(all_gather_output, dim, up_start, total_size, out[-1])
        npu_special_slice(all_gather_output, dim, down_start, up_start, out[-2])

        other_tensors = all_gather_output[:, :down_start].view(num_rank, -1)
        torch.split_with_sizes_copy(
                other_tensors, all_gather_input_split_sizes[:-2], dim=dim, out=out[:-2]
        )
        
        return

    torch.split_with_sizes_copy(
        all_gather_output, all_gather_input_split_sizes, dim=dim, out=out
    )
