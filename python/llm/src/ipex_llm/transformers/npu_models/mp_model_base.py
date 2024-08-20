#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import os
import torch
import time
import argparse

from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from intel_npu_acceleration_library.backend.factory import NNFactory
from typing import Optional, Sequence, List, Union, Any, Tuple
import numpy as np
import math
from intel_npu_acceleration_library.backend.runtime import set_contiguous, record_function
from intel_npu_acceleration_library.backend.runtime import adapt_output_tensor, _model_cache
from collections import deque
from transformers.cache_utils import Cache
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib
import ctypes
from ipex_llm.utils.common import invalidInputError
from typing import Optional, List, Generator
import uuid
from functools import partial
import torch.nn.functional as F
import torch.nn.parallel
import torch.distributed as dist
from filelock import FileLock

from transformers.utils import logging

logger = logging.get_logger(__name__)
import gc
from colorama import Fore, Back, Style
import torch.multiprocessing as mp
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast


@torch.no_grad()
def run_model(
    x: Union[torch.Tensor, List[torch.Tensor]],
    weights: List[torch.Tensor],
    backend_cls: Any,
    op_id: str,
    replica: int = 1,
) -> torch.Tensor:
    global _model_cache
    import time

    t0 = time.perf_counter()

    # Use or not op_id depending on the class used
    op_kwargs = {"op_id": op_id} if op_id else {}

    if not isinstance(x, (list, tuple)):
        x = [x]

    # Reshape input
    input_dtype = x[0].dtype
    x_np = [set_contiguous(elem).to(torch.float16).numpy() for elem in x]
    op_args = []
    op_args_flatten = []
    for w in weights:
        if isinstance(w, tuple):  # from QuantizedLinear
            op_args.append((set_contiguous(w[0]).numpy(), set_contiguous(w[1]).numpy()))
            op_args_flatten.append(op_args[-1][0])
            op_args_flatten.append(op_args[-1][1])
        else:
            op_args.append(set_contiguous(w).to(torch.float16).numpy())
            op_args_flatten.append(op_args[-1])

    shape_dtype_signature = "_".join(
        ["_".join(str(dim) for dim in t.shape) + f"_{t.dtype}" for t in x_np + op_args_flatten]
    )
    key = f"{backend_cls.func.__name__}_{shape_dtype_signature}"
    models = _model_cache.get(key, None)

    input_shapes = [elem.shape for elem in x_np]
    if models is None:
        _model_cache[key] = deque([backend_cls(*input_shapes) for i in range(replica)])
    elif len(models) < 1:
        _model_cache[key].append(backend_cls(*input_shapes))
    else:
        _model_cache[key].rotate(1)

    # Get the model
    model = _model_cache[key][0]

    with record_function(f"npu_factory_mul_{key}"):
        ret = model.run(x_np, *op_args, **op_kwargs)

    if isinstance(ret, list):
        results = [adapt_output_tensor(r, r.shape, input_dtype) for r in ret]
    else:
        results = adapt_output_tensor(ret, ret.shape, input_dtype)

    return results


class LowBitBaseMultiDecoderGraph(NNFactory):
    def __init__(
        self,
        hidden_shape: Sequence[int],
        *shapes,
        num_heads: int,
        num_key_value_heads: int,
        num_layers: int,
        cached_cos,
        cached_sin,
        input_layernorm_weights=None,
        post_attn_layernorm_weights=None,
        mode: str = "prefill",
        dtype: np.dtype = np.int8,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        profile: bool = False,
        device: str = "NPU",
        rms_norm_eps,
        intermediate_size,
    ):
        super().__init__(profile, device)
        self.max_seq_len = max_seq_len
        self.intermediate_size = intermediate_size
        self.dtype = dtype
        self.cached_cos = cached_cos
        self.cached_sin = cached_sin
        self.batch_size, self.seq_len, self.hidden_size = hidden_shape
        self.mode = mode
        self.rms_norm_eps = rms_norm_eps
        self.transpose_value = transpose_value

        cos = self.constant(self.cached_cos)
        self.cos = self.unsqueeze(cos, axis=0)

        sin = self.constant(self.cached_sin)
        self.sin = self.unsqueeze(sin, axis=0)

        if mode == "decode":
            self.kv_seq_len = self.max_seq_len + 1
        else:
            self.kv_seq_len = self.seq_len

        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # define input, the order self.parameter matters
        input = self.parameter((self.batch_size, self.seq_len, self.hidden_size))

        # Self Attention
        if mode == "decode":
            attention_mask = self.parameter((self.batch_size, 1, 1, self.max_seq_len + 1))
        else:
            attention_mask = self.parameter((self.batch_size, 1, self.seq_len, self.seq_len))

        position_ids = self.parameter((self.batch_size, self.seq_len))
        past_keys = []
        past_values = []
        if mode == "decode":
            for i in range(num_layers):
                past_key = self.parameter(
                    (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
                )
                if transpose_value:
                    past_value = self.parameter(
                        (self.batch_size, self.num_key_value_heads, self.head_dim, self.max_seq_len)
                    )
                else:
                    past_value = self.parameter(
                        (self.batch_size, self.num_key_value_heads, self.max_seq_len, self.head_dim)
                    )
                past_keys.append(past_key)
                past_values.append(past_value)
        else:
            past_keys = [None] * num_layers
            past_values = [None] * num_layers

        if input_layernorm_weights is None:
            input_layernorm_weights = []
            post_attn_layernorm_weights = []
            for i in range(num_layers):
                input_layernorm_weights.append(
                    self.parameter(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
                post_attn_layernorm_weights.append(
                    self.parameter(
                        (
                            1,
                            self.hidden_size,
                        )
                    )
                )
        else:
            input_layernorm_weights = [self.constant(w) for w in input_layernorm_weights]
            post_attn_layernorm_weights = [self.constant(w) for w in post_attn_layernorm_weights]

        hidden_states = input

        curr_key_values = []
        for i in range(num_layers):
            hidden_states, new_key_states, new_value_states = self.build_decoder(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                input_layernorm_weight=input_layernorm_weights[i],
                post_attention_layernorm_weight=post_attn_layernorm_weights[i],
                past_key=past_keys[i],
                past_value=past_values[i],
            )
            curr_key_values.append((new_key_states, new_value_states))

        # define outputs
        hidden_states = self.convert_to_fp16(hidden_states)

        for i in range(num_layers):
            new_key_states = self.convert_to_fp16(curr_key_values[i][0])
            new_value_states = self.convert_to_fp16(curr_key_values[i][1])

        print("start compiling")
        self.compile()

    def repeat_kv(self, hidden_states, n_rep, transpose=False):
        if n_rep == 1:
            return hidden_states
        if not transpose:
            hidden_states = self.reshape(
                hidden_states,
                [self.batch_size, self.num_key_value_heads, 1, self.kv_seq_len, self.head_dim],
            )
            hidden_states = self.broadcast(
                hidden_states,
                [self.batch_size, self.num_key_value_heads, n_rep, self.kv_seq_len, self.head_dim],
            )
            hidden_states = self.reshape(
                hidden_states,
                [self.batch_size, n_rep * self.num_key_value_heads, self.kv_seq_len, self.head_dim],
            )
        else:
            hidden_states = self.reshape(
                hidden_states,
                [self.batch_size, self.num_key_value_heads, 1, self.head_dim, self.kv_seq_len],
            )
            hidden_states = self.broadcast(
                hidden_states,
                [self.batch_size, self.num_key_value_heads, n_rep, self.head_dim, self.kv_seq_len],
            )
            hidden_states = self.reshape(
                hidden_states,
                [self.batch_size, n_rep * self.num_key_value_heads, self.head_dim, self.kv_seq_len],
            )
        return hidden_states

    def build_decoder(
        self,
        hidden_states,
        attention_mask,
        position_ids,
        input_layernorm_weight,
        post_attention_layernorm_weight,
        past_key=None,
        past_value=None,
    ):

        residual = hidden_states
        input_2d = self.reshape(hidden_states, (self.batch_size * self.seq_len, self.hidden_size))
        input_2d = self.layer_norm(input_2d, input_layernorm_weight)
        attn_output, new_key_states, new_value_states = self.attention(
            input_2d, position_ids, attention_mask, past_key, past_value
        )
        hidden_states = self.eltwise_add(residual, attn_output)
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states, post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.eltwise_add(residual, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)

        return hidden_states, new_key_states, new_value_states

    def attention(self, hidden_states, position_ids, attention_mask, past_key, past_value):
        query_states = self.linear(
            hidden_states,
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        key_states = self.linear(
            hidden_states,
            self.num_key_value_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )
        value_states = self.linear(
            hidden_states,
            self.num_key_value_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            wt_dtype=self.dtype,
        )

        query_states = self.reshape(
            query_states, [self.batch_size, self.seq_len, self.num_heads, self.head_dim]
        )
        key_states = self.reshape(
            key_states, [self.batch_size, self.seq_len, self.num_key_value_heads, self.head_dim]
        )
        value_states = self.reshape(
            value_states, [self.batch_size, self.seq_len, self.num_key_value_heads, self.head_dim]
        )

        query_states = self.transpose(query_states, [0, 2, 1, 3])
        key_states = self.transpose(key_states, [0, 2, 1, 3])
        if self.transpose_value:
            value_states = self.transpose(value_states, [0, 2, 3, 1])
        else:
            value_states = self.transpose(value_states, [0, 2, 1, 3])

        query_states, key_states = self.apply_rotary_pos_emb(
            query_states, key_states, self.cos, self.sin, position_ids
        )
        new_key_states = key_states
        new_value_states = value_states

        if self.mode == "decode":
            key_states = self.concat(past_key, key_states, axis=-2)
            if self.transpose_value:
                value_states = self.concat(past_value, value_states, axis=-1)
            else:
                value_states = self.concat(past_value, value_states, axis=-2)

        key_states = self.repeat_kv(key_states, self.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.num_key_value_groups, self.transpose_value)

        attn_weight = self.matmul(query_states, key_states, False, True) / (
            math.sqrt(self.head_dim)
        )
        attn_weight = self.eltwise_add(attn_weight, attention_mask)
        attn_weight = self.convert_to_fp32(attn_weight)
        attn_weight = self.softmax(attn_weight, -1)
        attn_weight = self.convert_to_fp16(attn_weight)
        attn_output = self.matmul(attn_weight, value_states, False, self.transpose_value)

        attn_output = self.transpose(attn_output, [0, 2, 1, 3])
        attn_output = self.reshape(attn_output, [self.batch_size, self.seq_len, self.hidden_size])

        attn_output = self.linear(
            attn_output, self.hidden_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )

        return attn_output, new_key_states, new_value_states

    def mlp(self, hidden_states):
        mm1 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )
        mm2 = self.linear(
            hidden_states, self.intermediate_size, self.hidden_size, bias=False, wt_dtype=self.dtype
        )  # type: ignore[attr-defined]
        mm1 = self.eltwise_mul(self.swish(mm1), mm2)  # type: ignore[attr-defined]
        hidden_states = self.linear(
            mm1, self.hidden_size, self.intermediate_size, bias=False, wt_dtype=self.dtype
        )
        return hidden_states

    def layer_norm(self, hidden_states, layernorm_weight):
        hidden_states = self.convert_to_fp32(hidden_states)
        variance = self.reduce_mean(
            self.power(hidden_states, self.constant(np.array([[2]], dtype=np.float32))),
            -1,
            keep_dims=True,
        )
        eps = self.constant(self.rms_norm_eps)
        hidden_states = self.eltwise_div(hidden_states, self.sqrt(self.eltwise_add(variance, eps)))
        layernorm_weight = self.convert_to_fp32(layernorm_weight)
        hidden_states = self.eltwise_mul(layernorm_weight, hidden_states)
        hidden_states = self.convert_to_fp16(hidden_states)
        return hidden_states

    def rotate_half(self, x):
        x1 = self.slice(
            x,
            [0, 0, 0, 0],
            [self.batch_size, self.num_heads, self.seq_len, self.head_dim // 2],
        )
        x2 = self.slice(
            x,
            [0, 0, 0, self.head_dim // 2],
            [self.batch_size, self.num_heads, self.seq_len, self.head_dim],
        )
        return self.concat(self.negative(x2), x1, axis=-1)

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        position_ids = self.squeeze(position_ids)
        cos = self.gather(cos, self.convert_to_int32(position_ids), self.constant(1), 0)
        sin = self.gather(sin, self.convert_to_int32(position_ids), self.constant(1), 0)
        cos = self.unsqueeze(cos, [1])
        sin = self.unsqueeze(sin, [1])

        q_embed = self.eltwise_add(
            self.eltwise_mul(q, cos), self.eltwise_mul(self.rotate_half(q), sin)
        )
        k_embed = self.eltwise_add(
            self.eltwise_mul(k, cos), self.eltwise_mul(self.rotate_half(k), sin)
        )

        return q_embed, k_embed


class BaseMultiDecoderlayerDecode(torch.nn.Module):

    def __init__(
        self,
        npu_backend_graph_cls: LowBitBaseMultiDecoderGraph,
        parameters: List[Tuple[torch.Tensor]],
        input_laynorm_weights: List[torch.Tensor],
        post_attn_layernorm_weights: List[torch.Tensor],
        layer_indexes: List[int],
        intra_stages: int,
        cached_cos: torch.Tensor,
        cached_sin: torch.Tensor,
        num_heads: int,
        head_dim: int,
        num_key_value_heads: int,
        rms_norm_eps,
        intermediate_size,
        max_seq_len: int = 1024,
        transpose_value: bool = False,
        do_print: bool = False,
    ):
        super().__init__()

        self.head_dim = head_dim
        self.num_heads = num_heads

        self.do_print = do_print
        self.npu_backend_graph_cls = npu_backend_graph_cls

        op_parameters = []
        for w in parameters:
            if isinstance(w, tuple):  # from QuantizedLinear
                op_parameters.append((w[0].numpy(), w[1].numpy()))
            else:
                op_parameters.append(w.to(torch.float16).numpy())
        self.op_parameters = op_parameters
        self.op_id = str(uuid.uuid4())
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        if isinstance(parameters[0], tuple):
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        self.intra_stages = intra_stages
        self.layer_indexes = layer_indexes
        self.num_layers_1 = len(self.layer_indexes) // 2
        self.num_layers_0 = len(self.layer_indexes) - self.num_layers_1
        num_layers = len(self.layer_indexes) // intra_stages
        self.layer_ranges = []
        for i in range(intra_stages):
            if i == intra_stages - 1:
                self.layer_ranges.append((i * num_layers, len(self.layer_indexes)))
            else:
                self.layer_ranges.append((i * num_layers, (i + 1) * num_layers))

        self.backend_decoders = []

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            lm_0 = input_laynorm_weights[start:end]
            lm_1 = post_attn_layernorm_weights[start:end]
            decoder = self.npu_backend_graph_cls(
                [1, 1, num_heads * head_dim],
                input_layernorm_weights=lm_0,
                post_attn_layernorm_weights=lm_1,
                cached_cos=cached_cos,
                cached_sin=cached_sin,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                num_layers=end - start,
                max_seq_len=max_seq_len,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
                mode="decode",
                transpose_value=self.transpose_value,
                dtype=np_dtype,
            )
            self.backend_decoders.append(decoder)

        for i in range(intra_stages):
            start, end = self.layer_ranges[i]
            num_intra_layers = end - start
            with FileLock(f"decoder_run.lock"):
                self.backend_decoders[i].setWeights(
                    3 + (num_intra_layers) * 2, self.op_id, *op_parameters[start * 7 : end * 7]
                )
                backend_lib.run(self.backend_decoders[i]._mm)

        self.kv_cache_c_parameter_handel = []
        self.kv_cache_parameters = []
        self.kv_cache_prefetched = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:

        inputs = (
            hidden_states.to(torch.float16),
            attention_mask,
            position_ids,
        )

        if len(self.kv_cache_parameters) > 0:
            # the case kv cache changed
            cached_prt = self.kv_cache_parameters[0].storage().data_ptr()
            current_ptr = past_key_value.key_cache[self.layer_indexes[0]].storage().data_ptr()
            if cached_prt != current_ptr:
                self.kv_cache_parameters = []
                self.kv_cache_c_parameter_handel = []
                self.kv_cache_prefetched = False

        if len(self.kv_cache_parameters) == 0:
            for idx in self.layer_indexes:
                past_key = past_key_value.key_cache[idx]
                past_value = past_key_value.value_cache[idx]

                invalidInputError(
                    past_key.dtype == torch.float16, f"past_key dtype is {past_key.dtype}"
                )

                new_size = (past_key.size(0), past_key.size(1), self.max_seq_len, past_key.size(3))
                past_key = past_key.as_strided(new_size, past_key.stride(), storage_offset=0)
                invalidInputError(past_key.is_contiguous(), "past_key is not contiguous")
                past_value = past_value.as_strided(new_size, past_value.stride(), storage_offset=0)
                if self.transpose_value:
                    past_value = past_value.transpose(-1, -2)
                invalidInputError(past_value.is_contiguous(), "past_value is not contiguous")

                self.kv_cache_parameters.append(past_key)
                self.kv_cache_parameters.append(past_value)

            for i in range(self.intra_stages):
                start, end = self.layer_ranges[i]
                layer_kv_cache = self.kv_cache_parameters[start * 2 : end * 2]
                layer_kv_cache = [p.numpy() for p in layer_kv_cache]
                handle = self.backend_decoders[i].create_parameters(layer_kv_cache)
                self.kv_cache_c_parameter_handel.append(handle)

        x_np = [elem.to(torch.float16).numpy() for elem in inputs]

        with record_function(f"npu_factory"):
            if not self.kv_cache_prefetched:
                for i in range(self.intra_stages):
                    self.backend_decoders[i].load_wt_fn(
                        len(inputs),
                        self.backend_decoders[i]._mm,
                        self.kv_cache_c_parameter_handel[i],
                    )

            array_type = ctypes.POINTER(ctypes.c_char) * self.intra_stages
            models_ptr = array_type(
                *[self.backend_decoders[i]._mm for i in range(self.intra_stages)]
            )
            inputs_ptr = (ctypes.c_void_p * 3)(
                x_np[0].ctypes.data_as(ctypes.c_void_p),
                x_np[1].ctypes.data_as(ctypes.c_void_p),
                x_np[2].ctypes.data_as(ctypes.c_void_p),
            )
            t0 = time.perf_counter()
            backend_lib.run_decoders(models_ptr, inputs_ptr, 2, 3)
            t1 = time.perf_counter()

        hidden_states = self.backend_decoders[-1].torch_out[0]

        if self.do_print:
            print("outputs:", hidden_states)

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs, t1 - t0

    def post_forward(self, past_key_value, cache_position):
        key_value_states = []
        for i in range(self.intra_stages):
            for j in range(1, len(self.backend_decoders[i].torch_out)):
                key_value_states.append(self.backend_decoders[i].torch_out[j])

        cache_kwargs = {
            "cache_position": cache_position,
            "max_seq_len": self.max_seq_len,
            "transpose": self.transpose_value,
        }
        for i in range(len(self.layer_indexes)):
            key_states, value_states = past_key_value.update(
                key_value_states[2 * i],
                key_value_states[2 * i + 1],
                self.layer_indexes[i],
                cache_kwargs,
            )

        for i in range(self.intra_stages):
            self.backend_decoders[i].load_wt_fn(
                3, self.backend_decoders[i]._mm, self.kv_cache_c_parameter_handel[i]
            )
        self.kv_cache_prefetched = True


class BaseSingleDecoderlayerPrefill(torch.nn.Module):

    def __init__(
        self,
        npu_backend_graph_cls: LowBitBaseMultiDecoderGraph,
        parameters: List[torch.Tensor],
        cached_cos,
        cached_sin,
        layer_norm_0,
        layer_norm_1,
        num_heads: int,
        num_key_value_heads: int,
        layer_idx: int,
        rms_norm_eps,
        intermediate_size,
        max_seq_len: int = 128,
        transpose_value: bool = False,
    ):
        super().__init__()
        self.npu_backend_graph_cls = npu_backend_graph_cls
        self.op_parameters = parameters
        self.op_id = str(uuid.uuid4())
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len
        self.transpose_value = transpose_value
        # self.rotary_emb = rotary_emb
        if isinstance(parameters[0], tuple):  # weight, scale from QuantizedLinear
            np_dtype = np.int8 if parameters[0][0].dtype == torch.int8 else np.uint8
        else:  # FP16 Linear
            np_dtype = np.float16

        self.backend_cls_prefill = partial(
            self.npu_backend_graph_cls,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            num_layers=1,
            cached_cos=cached_cos,
            cached_sin=cached_sin,
            input_layernorm_weights=None,
            post_attn_layernorm_weights=None,
            max_seq_len=max_seq_len,
            rms_norm_eps=rms_norm_eps,
            intermediate_size=intermediate_size,
            mode="prefill",
            transpose_value=self.transpose_value,
            dtype=np_dtype,
        )
        self.layer_norm_0 = layer_norm_0
        self.layer_norm_1 = layer_norm_1

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Torch module forward method.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: result
        """

        seq_len = hidden_states.shape[1]

        backend_cls = self.backend_cls_prefill
        inputs = (hidden_states.to(torch.float16), attention_mask, position_ids)
        inputs += (self.layer_norm_0, self.layer_norm_1)
        hidden_states, past_key, past_value = run_model(
            inputs, self.op_parameters, backend_cls, self.op_id, replica=2
        )
        cache_kwargs = {
            "cache_position": cache_position,
            "max_seq_len": self.max_seq_len,
            "transpose": self.transpose_value,
        }
        key_states, value_states = past_key_value.update(
            past_key, past_value, self.layer_idx, cache_kwargs
        )

        outputs = (hidden_states,)
        outputs += (past_key_value,)
        return outputs
