from typing import Dict, List, Optional, Tuple

import torch.nn as nn
from fairseq import utils
from torch import Tensor



"""
@Desc: Base class for decoders.
"""
class FairseqDecoder(nn.Module):
    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.adaptive_softmax = None

    #@Param{prev_output_tokens}: @Type{LongTensor}, shifted output tokens of @Shape{[batch, tgt_len]}, for teacher forcing.
    #@Param{encoder_out}: @Type{Dict, Optional}, output from the encoder, used for encoder-side attention
    #@Return: @Tuple(
    #           The decoder's output of @Shape{[batch, tgt_len, vocab]}, 
    #           A dictionary with any model-specific outputs.
    #         )
    def forward(self, prev_output_tokens, encoder_out=None, **kwargs):
        x, extra = self.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        x = self.output_layer(x)
        return x, extra

    #@Return: (
    #           The decoder's features of @Shape{[batch, tgt_len, embed_dim]}, 
    #           A dictionary with any model-specific outputs.
    #         )
    def extract_features(self, prev_output_tokens, encoder_out=None, **kwargs):
        raise NotImplementedError

    #@Desc: Project features to the default output size, e.g., vocabulary size.
    #@Param{features}: @Shape{Tensor}, features returned by @Func{extract_features}.
    def output_layer(self, features, **kwargs):
        raise NotImplementedError

    #@Desc: Get normalized probabilities (or log probs) from a net's output.
    #@TODO: @Param explanation
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    #@Attention:TorchScript doesn't support super() method so that the scriptable Subclass can't access the base class model in Torchscript.
    #           Current workaround is to add a helper function with different name and call the helper function from scriptable Subclass.
    #@Desc: Get normalized probabilities (or log probs) from a net's output.
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, "adaptive_softmax") and self.adaptive_softmax is not None:
            if sample is not None:
                assert "target" in sample
                target = sample["target"]
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    #@Desc: Maximum input length supported by the decoder.
    def max_positions(self):
        return 1e6  # an arbitrary large number

    #@Desc: Upgrade old state dicts to work with newer code.
    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True
