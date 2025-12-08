from typing import (
    Callable,
    List,
    Optional,
    Union,
    Tuple
)

import torch
import torch.nn as nn

from torch.distributed._composable import contract
from torch.distributed._composable_state import _insert_module_state

from torch.distributed.tensor import DeviceMesh, Shard
from torch.distributed.utils import _get_root_modules
from torch.distributed.device_mesh import _get_device_handle

from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo, HSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_init import (
    _get_device_from_mesh,
    _get_managed_modules,
    _get_managed_states,
    _get_post_forward_mesh_info,
    _init_default_fully_shard_mesh,
    _move_states_to_device,
)
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.fsdp._fully_shard._fully_shard import (
    cls_to_fsdp_cls,
    _unimplemented_deepcopy,
    FSDPModule,
)

@contract(state_cls=FSDPState)  # type: ignore[operator]
def fully_shard(
    module: Union[nn.Module, List[nn.Module]],
    *,
    hook_module: nn.Module = None,
    mesh: Optional[DeviceMesh] = None,
    reshard_after_forward: Union[bool, int] = True,
    shard_placement_fn: Optional[Callable[[nn.Parameter], Optional[Shard]]] = None,
    mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
    offload_policy: OffloadPolicy = OffloadPolicy(),
):
    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        raise ValueError(
            f"fully_shard does not support containers that do not implement forward: {module}"
        )
    mesh = mesh or _init_default_fully_shard_mesh()
    if mesh.ndim not in (1, 2):
        raise ValueError(f"fully_shard expects a 1D or 2D DeviceMesh but got {mesh}")
    elif mesh.ndim == 1:
        mesh_info = FSDPMeshInfo(mesh, shard_mesh_dim=0)
    else:
        if mesh.mesh_dim_names is None:
            raise AssertionError(
                "Please init the 2D mesh for HSDP with mesh_dim_names specified"
            )
        mesh_info = HSDPMeshInfo(mesh, shard_mesh_dim=1, replicate_mesh_dim=0)
    device = _get_device_from_mesh(mesh)
    post_forward_mesh_info = _get_post_forward_mesh_info(
        reshard_after_forward, mesh_info
    )

    arg_module = module
    modules = (
        (module,) if isinstance(module, nn.Module) else tuple(_get_root_modules(module))
    )
    state = fully_shard.state(modules[0])
    state.init(modules, device, mp_policy, hook_module=hook_module)

    managed_modules = _get_managed_modules(modules)
    params, buffers = _get_managed_states(managed_modules)
    _move_states_to_device(params, buffers, device)
    if params:
        state._fsdp_param_group = FSDPParamGroup(
            params,
            modules,
            mesh_info,
            post_forward_mesh_info,
            device,
            shard_placement_fn,
            mp_policy,
            offload_policy,
        )

    # For Dynamo
    for managed_module in managed_modules:
        managed_module._is_fsdp_managed_module = True  # type: ignore[assignment]
        managed_module._fsdp_use_orig_params = True  # type: ignore[assignment]

    # Place FSDP leftmost for highest priority in the method resolution order
    for module in modules:
        cls = module.__class__
        new_cls = cls_to_fsdp_cls.get(cls, None)
        if not new_cls:
            dct = {"__deepcopy__": _unimplemented_deepcopy}
            new_cls = type(f"FSDP{cls.__name__}", (FSDPModule, cls), dct)
            cls_to_fsdp_cls[cls] = new_cls
        module.__class__ = new_cls
    return arg_module


def fsdp_state_init(
    self,
    modules: Tuple[nn.Module, ...],
    device: torch.device,
    mp_policy: MixedPrecisionPolicy,
    hook_module: nn.Module = None,
) -> None:
    for module in modules:
        _insert_module_state(module, self)
    self._modules = modules
    self._device = device
    self._device_handle = _get_device_handle(device.type)
    self._mp_policy = mp_policy


    if len(modules) == 1:
        self._pre_forward_hook_handle = modules[0].register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = modules[0].register_forward_hook(
            self._post_forward, prepend=False
        )
    elif hook_module is not None:
        self._pre_forward_hook_handle = hook_module.register_forward_pre_hook(
            self._pre_forward, prepend=True, with_kwargs=True
        )
        self._post_forward_hook_handle = hook_module.register_forward_hook(
            self._post_forward, prepend=False
        )
    else:
        hook_handle = _register_group_forward_hooks(
            modules,
            self._pre_forward,
            self._post_forward,
            self._modules_to_run_forward,
        )
        self._pre_forward_hook_handle = hook_handle
        self._post_forward_hook_handle = hook_handle

def apply_fsdp_partial_shard_patch():
    torch.distributed.fsdp.fully_shard = fully_shard
    torch.distributed.fsdp._fully_shard._fsdp_state.FSDPState.init = fsdp_state_init