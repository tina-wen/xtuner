import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from xtuner.v1.float8.config import ScalingGranularity
from xtuner.v1.float8.float8_gmm_tile_wise import TileWiseFloat8GroupedLinear
from xtuner.v1.ops import group_gemm

from torch.autograd import Function
import torch_npu
import os

class GroupedLinear(nn.Module):
    # TODO:Missng example docs
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_routed_experts: int,
        moe_bias: bool = False,
        ep_mesh: DeviceMesh | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_routed_experts = num_routed_experts
        # 添加训练NZ开关，注意该开关只在npu上有性能增益
        if int(os.getenv("GROUPMM_NZ_TRANSPOSE","0")):
            weight = torch.empty(num_routed_experts * in_features, out_features)
        else:
            weight = torch.empty(num_routed_experts * out_features, in_features)

        self.ep_mesh = ep_mesh
        if self.ep_mesh is not None and self.ep_mesh.size() > 1:
            self.weight = nn.Parameter(distribute_tensor(weight, ep_mesh, [Shard(0)]))
        else:
            self.weight = nn.Parameter(weight)

        self.moe_bias = moe_bias
        if self.moe_bias:
            bias = torch.zeros(num_routed_experts, out_features)
            if self.ep_mesh is not None and self.ep_mesh.size() > 1:
                self.bias = nn.Parameter(distribute_tensor(bias, ep_mesh, [Shard(0)]))
            else:
                self.bias = nn.Parameter(torch.zeros(num_routed_experts, out_features))

    def forward(self, x: torch.Tensor, tokens_per_expert: torch.Tensor, decoding: bool = False):
        weight = self.weight.to_local() if isinstance(self.weight, DTensor) else self.weight
        
        if int(os.getenv("GROUPMM_NZ_TRANSPOSE","0")):
            weight = weight.view(-1, self.in_features, self.out_features)
            out = NpuGMMOp.apply(weight, x, tokens_per_expert)
        else:
            weight = weight.view(-1, self.out_features, self.in_features)
            out = group_gemm(x, weight, tokens_per_expert)

        if self.moe_bias:
            bias = self.bias.to_local() if isinstance(self.bias, DTensor) else self.bias
            out = out + bias.repeat_interleave(tokens_per_expert, dim=0)  # TODO: 无法 compile
        return out


class NpuGMMOp(Function):
    @staticmethod
    def forward(ctx, weight, x, tokens_per_expert):
        if not int(os.getenv("GROUPMM_NZ_TRANSPOSE","0")):
            weight = torch.transpose(weight, 1, 2)
        ctx.save_for_backward(weight, x, tokens_per_expert)
        outs = torch_npu.npu_grouped_matmul([x], [weight], group_list = tokens_per_expert, group_type = 0, group_list_type = 1, split_item = 2)
        return outs[0]


    @staticmethod
    def backward(ctx, grad_output):
        tensors = ctx.saved_tensors
        weight = tensors[0]
        input_tensor = tensors[1]
        tokens_per_expert = tensors[2]
        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul([grad_output], [weight], group_list = tokens_per_expert, 
                                                  group_type = 0, group_list_type = 1, split_item=2)[0] 
        grad_weight = torch_npu.npu_grouped_matmul([input_tensor.T], [grad_output], bias=None, group_list = tokens_per_expert,
                                                   split_item=3, group_type=2, group_list_type=1)[0]
        if not int(os.getenv("GROUPMM_NZ_TRANSPOSE","0")):
            grad_weight = torch.transpose(grad_weight, 1, 2)
        return grad_weight, grad_input, None



def build_grouped_linear(
    in_features: int,
    out_features: int,
    num_routed_experts: int,
    moe_bias: bool = False,
    ep_mesh: DeviceMesh | None = None,
    float8_cfg=None,
):
    """Build a grouped linear layer with optional float8 support."""
    if float8_cfg is None:
        return GroupedLinear(in_features, out_features, num_routed_experts, moe_bias=moe_bias, ep_mesh=ep_mesh)
    elif float8_cfg.scaling_granularity_grouped_gemm == ScalingGranularity.TILEWISE:
        return TileWiseFloat8GroupedLinear(
            in_features, out_features, num_routed_experts, moe_bias=moe_bias, ep_mesh=ep_mesh
        )
    else:
        raise NotImplementedError(f"Unsupported float8 scaling granularity: {float8_cfg.scaling_granularity_gemm}")
