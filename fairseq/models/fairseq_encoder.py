from typing import Dict, List, NamedTuple, Optional

import torch
import torch.nn as nn
from torch import Tensor



#@Callback: @Type{NamedTuple}使得@Type{Tuple}的每个字段都有一个明确的名称
EncoderOut = NamedTuple(
    "EncoderOut",
    [
        ("encoder_out", Tensor),  # T x B x C
        ("encoder_padding_mask", Optional[Tensor]),  # B x T
        ("encoder_embedding", Optional[Tensor]),  # B x T x C
        ("encoder_states", Optional[List[Tensor]]),  # List[T x B x C]
        ("src_tokens", Optional[Tensor]),  # B x T
        ("src_lengths", Optional[Tensor]),  # B x 1
    ],
)



"""
@Desc: Base class for encoders.
"""
class FairseqEncoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    #@Param{src_tokens}: @Type{LongTensor}, tokens in the source language of @Shape{[batch, src_len]}
    #@Param{src_lengths}: @Type{LongTensor}, lengths of each source sentence of @Shape{[batch]}
    def forward(self, src_tokens, src_lengths=None, **kwargs):
        raise NotImplementedError

    #@Desc: A TorchScript-compatible version of forward.
    #Encoders which use additional arguments may want to override this method for TorchScript compatibility.
    def forward_torchscript(self, net_input: Dict[str, Tensor]):
        if torch.jit.is_scripting():
            return self.forward(
                src_tokens=net_input["src_tokens"],
                src_lengths=net_input["src_lengths"],
            )
        else:
            return self.forward_non_torchscript(net_input)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    #@Desc: Reorder encoder output according to @Param{new_order}.
    #@Param{encoder_out}: output from the @Func{forward}.
    #@Param{new_order}: @Type{LongTensor}, desired order.
    #@Return: @Param{encoder_out} rearranged according to @Param{new_order}. 
    def reorder_encoder_out(self, encoder_out, new_order):
        raise NotImplementedError

    #@Desc: Maximum input length supported by the encoder.
    def max_positions(self):
        return 1e6  # an arbitrary large number

    #@Desc: Upgrade old state dicts to work with newer code.
    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    #@Desc: State from trainer to pass along to model at every update.
    #@Question: 什么意思呢？
    def set_num_updates(self, num_updates):
        def _apply(m):
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

        self.apply(_apply)
