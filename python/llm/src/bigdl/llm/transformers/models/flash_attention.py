from typing import Optional, TypeVar, Union, overload
from bigdl.llm.utils.common import invalidInputError

import torch
import torch.nn.functional as F
from torch import Tensor, device, dtype, nn

T = TypeVar("T", bound="torch.nn.Module")

import bigdl.llm.ggml.model.llama.llama_cpp as ggml

import torch
import ctypes


def ggml_flash_attn(q: torch.Tensor, # [b, h, sq, d]
                              k: torch.Tensor, # [b, h, sk, d]
                              v: torch.Tensor, # [b, h, d, sk]
                              masked: bool,
                              ):

    q_ptr = q.data_ptr()
    q_ne = tuple(reversed(q.shape))
    q_nb = tuple([s * 4 for s in reversed(q.stride())])
    k_ptr = k.data_ptr()
    k_ne = tuple(reversed(k.shape))
    k_nb = tuple([s * 4 for s in reversed(k.stride())])
    v_prt = v.data_ptr()
    v_ne = tuple(reversed(v.shape))
    v_nb = tuple([s * 4 for s in reversed(v.stride())])

    output_shape = (q.shape[0], q.shape[1], q.shape[2], v.shape[2])

    output = torch.empty(output_shape, dtype=torch.float32)
    output_ptr = output.data_ptr()
    output_ne = tuple(reversed(output.shape))

    # ctx_p = ctx.context
    q_ne = (ctypes.c_int64 * 4)(*q_ne)
    q_nb = (ctypes.c_int64 * 4)(*q_nb)
    q_data = ctypes.c_void_p(q_ptr)
    k_ne = (ctypes.c_int64 * 4)(*k_ne)
    k_nb = (ctypes.c_int64 * 4)(*k_nb)
    k_data = ctypes.c_void_p(k_ptr)
    v_ne = (ctypes.c_int64 * 4)(*v_ne)
    v_nb = (ctypes.c_int64 * 4)(*v_nb)
    v_data = ctypes.c_void_p(v_prt)
    output_ptr = ctypes.c_void_p(output_ptr)
    output_ne = (ctypes.c_int64 * 4)(*output_ne)

    ggml.ggml_compute_forward_flash_attention(
        q_ne=q_ne,
        q_nb=q_nb,
        q_data=q_data,
        k_ne=k_ne,
        k_nb=k_nb,
        k_data=k_data,
        v_ne=v_ne,
        v_nb=v_nb,
        v_data=v_data,
        masked=masked,
        output_ne=output_ne,
        output_data=output_ptr,
    )

    return output