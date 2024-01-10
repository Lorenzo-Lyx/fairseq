"""
Base classes for various fairseq models.
"""

import logging
from argparse import Namespace
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import Dictionary
from fairseq.dataclass.utils import (
    convert_namespace_to_omegaconf,
    gen_parser_from_dataclass,
)
from fairseq.models import FairseqDecoder, FairseqEncoder
#@Question: What's oemgaconf.DictConfig?
from omegaconf import DictConfig
from torch import Tensor


logger = logging.getLogger(__name__)


def check_type(module, expected_type):
    if hasattr(module, "unwrapped_module"):
        assert isinstance(
            module.unwrapped_module, expected_type
        ), f"{type(module.unwrapped_module)} != {expected_type}"
    else:
        assert isinstance(module, expected_type), f"{type(module)} != {expected_type}"



#@Desc: Base class for fairseq models.
class BaseFairseqModel(nn.Module):

    def __init__(self):
        super().__init__()
        self._is_generation_fast = False

    #@Desc: Add model-specific arguments to the parser.
    #@Return: None, But @Param{parser}会被构建，构建之后的@Param{parser}, field中的default key会被移除
    @classmethod
    def add_args(cls, parser):
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            #@Explain: do not set defaults so that settings defaults from various architectures still works
            gen_parser_from_dataclass(parser, dc(), delete_default=True)

    #@Desc: Factory function, Build a new model instance.
    @classmethod
    def build_model(cls, args, task):
        raise NotImplementedError("Model must implement the build_model method")

    #@Desc: Get targets from either the sample or the net's output.
    def get_targets(self, sample, net_output):
        return sample["target"]

    #@Desc: Get normalized probabilities (or log probs) from a net's output.
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
    #@Desc: Scriptable helper function for get_normalized_probs in @Class{BaseModel}
    def get_normalized_probs_scriptable(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Scriptable helper function for get_normalized_probs in ~BaseFairseqModel"""
        if hasattr(self, "decoder"):
            return self.decoder.get_normalized_probs(net_output, log_probs, sample)
        #@Explain: syntactic sugar for simple models which don't have a decoder (e.g., the classification tutorial)
        elif torch.is_tensor(net_output):
            logits = net_output.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    #@Desc: Similar to *forward* but only return features.
    def extract_features(self, *args, **kwargs):
        """Similar to *forward* but only return features."""
        return self(*args, **kwargs)

    #@Desc: Maximum length supported by the model.
    def max_positions(self):
        """Maximum length supported by the model."""
        return None

    #@Desc: Copies parameters and buffers from @Param{state_dict} into this module and its descendants.
    #       Overrides the method in @Class{nn.Module}. 
    #       Compared with that method this additionally "upgrades" @Param{state_dicts} from old checkpoints.
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg: Optional[DictConfig] = None,
        args: Optional[Namespace] = None,
    ):
        if model_cfg is None and args is not None:
            logger.warn(
                "using 'args' is deprecated, please update your code to use dataclass config"
            )
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)

    #@Desc: Upgrade old state dicts to work with newer code.
    def upgrade_state_dict(self, state_dict):
        self.upgrade_state_dict_named(state_dict, "")

    #@Desc: Upgrade old state dicts to work with newer code.
    #@Param{state_dict}: @Type{dict}, state dictionary to upgrade, in place
    #@Param{name}: @Type{str}, the state dict key corresponding to the current module
    #@Return: 没太搞清楚这个函数的作用是什么，循环中递归中，似乎没有改变什么东西
    def upgrade_state_dict_named(self, state_dict, name):
        assert state_dict is not None

        def do_upgrade(m, prefix):
            if len(prefix) > 0:
                prefix += "."

            for n, c in m.named_children():
                name = prefix + n
                if hasattr(c, "upgrade_state_dict_named"):
                    c.upgrade_state_dict_named(state_dict, name)
                elif hasattr(c, "upgrade_state_dict"):
                    c.upgrade_state_dict(state_dict)
                do_upgrade(c, name)

        do_upgrade(self, name)

    #@Desc: State from trainer to pass along to model at every update.
    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        for m in self.modules():
            if hasattr(m, "set_num_updates") and m != self:
                m.set_num_updates(num_updates)

    def set_epoch(self, epoch):
        for m in self.modules():
            if hasattr(m, "set_epoch") and m != self:
                m.set_epoch(epoch)

    #@Desc: Prepare model for inference.
    def prepare_for_inference_(self, cfg: DictConfig):
        """Prepare model for inference."""
        kwargs = {}
        kwargs["beamable_mm_beam_size"] = (
            None
            if getattr(cfg.generation, "no_beamable_mm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["need_attn"] = getattr(cfg.generation, "print_alignment", False)
        if getattr(cfg.generation, "retain_dropout", False):
            kwargs["retain_dropout"] = cfg.generation.retain_dropout
            kwargs["retain_dropout_modules"] = cfg.generation.retain_dropout_modules
        self.make_generation_fast_(**kwargs)

    #@Desc: 递归式地使得model的所有modules进入快速生成模式
    #@Note: Legacy entry point to optimize model for faster generation.
    #       Prefer prepare_for_inference_.
    def make_generation_fast_(self, **kwargs):
        #@Explain: only apply once
        if self._is_generation_fast:
            return
        self._is_generation_fast = True

        #@Desc: remove weight norm from all modules in the network
        def apply_remove_weight_norm(module):
            try:
                nn.utils.remove_weight_norm(module)
            #@Explain: this module didn't have weight norm
            except (AttributeError, ValueError):
                return

        self.apply(apply_remove_weight_norm)

        def apply_make_generation_fast_(module, prefix):
            if len(prefix) > 0:
                prefix += "."

            base_func = BaseFairseqModel.make_generation_fast_
            for n, m in module.named_modules():
                if (
                    m != self
                    and hasattr(m, "make_generation_fast_")
                    #@Explain:  don't call this implementation again, 
                    #           e.g., if children modules also inherit from BaseModel
                    and m.make_generation_fast_.__func__ is not base_func
                ):
                    name = prefix + n
                    m.make_generation_fast_(name=name, **kwargs)

        apply_make_generation_fast_(self, "")

        def train(mode=True):
            if mode:
                raise RuntimeError("cannot train after make_generation_fast")

        #@Explain: this model should no longer be used for training
        self.eval()
        self.train = train

    #@Desc: Make model exportable via ONNX trace.
    def prepare_for_onnx_export_(self, **kwargs):
        seen = set()

        def apply_prepare_for_onnx_export_(module):
            if (
                module != self
                and hasattr(module, "prepare_for_onnx_export_")
                and module not in seen
            ):
                seen.add(module)
                module.prepare_for_onnx_export_(**kwargs)

        self.apply(apply_prepare_for_onnx_export_)

    #@Desc: Load a @Class{fairseq.models.BaseModel} from a pre-trained model file.
    #       Downloads and caches the pre-trained model file if needed.
    #       The base implementation returns a @Class{fairseq.hub_utils.GeneratorHubInterface}, 
    #       which can be used to generate translations or sample from language models.
    #       The underlying @Class{fairseq.models.BaseModel} can be accessed via the @Attribute{generator.models}
    #       Other models may override this to implement custom hub interfaces.
    #Param{model_name_or_path}: @Type{str}, either the name of a pre-trained model to load 
    #                           or a path/URL to a pre-trained model state dict.
    #@Param{checkpoint_file}:   @Type{str, optional}, colon-separated list of checkpoint files
    #                           in the model archive to ensemble (default: 'model.pt')
    #@Param{data_name_or_path}: @Type{str, optional}, point args.data to the archive at the given path/URL.
    #                           Can start with '.' or './' to reuse the model archive path.
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            **kwargs,
        )
        logger.info(x["args"])
        return hub_utils.GeneratorHubInterface(x["args"], x["task"], x["models"])

    @classmethod
    def hub_models(cls):
        return {}



"""
@Desc: Base class for encoder-decoder models.
"""
class FairseqEncoderDecoderModel(BaseFairseqModel):
    #@Param{encoder}, @Class{BaseEncoder}, the encoder
    #@Param{decoder}, @Class{BaseDecoder}, the decoder
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        check_type(self.encoder, FairseqEncoder)
        check_type(self.decoder, FairseqDecoder)

    """
    @Desc:  Run the forward pass for an encoder-decoder model.
            First feed a batch of source tokens through the encoder.
            Then, feed the encoder output and previous decoder outputs (i.e., teacher forcing)
            to the decoder to produce the next outputs:
                encoder_out = self.encoder(src_tokens, src_lengths)
                return self.decoder(prev_output_tokens, encoder_out)
    @Param{src_tokens}, @Type{LongTensor}, tokens in the source language of @Shape{[batch, src_len]}
    @Param{src_lengths}, @Type{LongTensor}, source sentence lengths of @Shape{[batch]}
    @Param{prev_output_tokens}, @Type{LongTensor}, previous decoder outputs of @Shape{[batch, tgt_len]}, for teacher forcing
    @Return: @Tuple{(
                the decoder's output of @Shape{[batch, tgt_len, vocab]},
                a dictionary with any model-specific outputs
            )}
    """
    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return decoder_out

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    #@Desc: Similar to @Func{forward} but only return features.
    #@Return:   @Tuple{(
    #               the decoder's features of @Shape{[batch, tgt_len, embed_dim]}, 
    #               a dictionary with any model-specific outputs
    #           )}
    def extract_features(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        features = self.decoder.extract_features(
            prev_output_tokens, encoder_out=encoder_out, **kwargs
        )
        return features

    #@Desc: Project features to the default output size (typically vocabulary size).
    def output_layer(self, features, **kwargs):
        return self.decoder.output_layer(features, **kwargs)

    #@Desc: Maximum length supported by the model.
    def max_positions(self):
        return (self.encoder.max_positions(), self.decoder.max_positions())

    #@Desc: Maximum length supported by the decoder.
    def max_decoder_positions(self):
        return self.decoder.max_positions()


class FairseqModel(FairseqEncoderDecoderModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        utils.deprecation_warning(
            "FairseqModel is deprecated, please use FairseqEncoderDecoderModel "
            "or BaseFairseqModel instead",
            stacklevel=4,
        )



"""
@Desc: Base class for combining multiple encoder-decoder models.
"""
class FairseqMultiModel(BaseFairseqModel):
    def __init__(self, encoders, decoders):
        super().__init__()
        assert encoders.keys() == decoders.keys()
        self.keys = list(encoders.keys())
        for key in self.keys:
            check_type(encoders[key], FairseqEncoder)
            check_type(decoders[key], FairseqDecoder)

        self.models = nn.ModuleDict(
            {
                key: FairseqEncoderDecoderModel(encoders[key], decoders[key])
                for key in self.keys
            }
        )

    """
    @Desc:  Helper function to build shared embeddings for a set of languages after
            checking that all dicts corresponding to those languages are equivalent.
    @Param{dicts}:  Dict of lang_id to its corresponding Dictionary.
    @Param{langs}:  languages that we want to share embeddings for
    @Param{embed_dim}:  embedding dimension
    @Param{build_embedding}:callable function to actually build the embedding
    @Param{pretrained_embed_path}:  Optional path to load pretrained embeddings
    """
    @staticmethod
    def build_shared_embeddings(
        dicts: Dict[str, Dictionary],
        langs: List[str],
        embed_dim: int,
        build_embedding: callable,
        pretrained_embed_path: Optional[str] = None,
    ):
        shared_dict = dicts[langs[0]]
        if any(dicts[lang] != shared_dict for lang in langs):
            raise ValueError(
                "--share-*-embeddings requires a joined dictionary: "
                "--share-encoder-embeddings requires a joined source "
                "dictionary, --share-decoder-embeddings requires a joined "
                "target dictionary, and --share-all-embeddings requires a "
                "joint source + target dictionary."
            )
        return build_embedding(shared_dict, embed_dim, pretrained_embed_path)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        raise NotImplementedError

    #@Desc: Maximum length supported by the model.
    def max_positions(self):
        return {
            key: (
                self.models[key].encoder.max_positions(),
                self.models[key].decoder.max_positions(),
            )
            for key in self.keys
        }

    #@Desc: Maximum length supported by the decoder.
    def max_decoder_positions(self):
        return min(model.decoder.max_positions() for model in self.models.values())

    #@Desc: 返回多模型的第一个encoder
    @property
    def encoder(self):
        return self.models[self.keys[0]].encoder

    #@Desc: 返回多模型的第一个decoder
    @property
    def decoder(self):
        return self.models[self.keys[0]].decoder

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    #@Desc: Copies parameters and buffers from @Param{state_dict} into this module and its descendants.
    #       Overrides the @Func{nn.Module}.
    #       Compared with that method this additionally "upgrades" @Param{state_dicts} from old checkpoints.
    def load_state_dict(
        self,
        state_dict,
        strict=True,
        model_cfg=None,
        args: Optional[Namespace] = None,
    ):
        """Copies parameters and buffers from *state_dict* into this module and
        its descendants.

        Overrides the method in :class:`nn.Module`. Compared with that method
        this additionally "upgrades" *state_dicts* from old checkpoints.
        """

        if model_cfg is None and args is not None:
            logger.warn(
                "using 'args' is deprecated, please update your code to use dataclass config"
            )
            model_cfg = convert_namespace_to_omegaconf(args).model

        self.upgrade_state_dict(state_dict)

        from fairseq.checkpoint_utils import prune_state_dict

        new_state_dict = prune_state_dict(state_dict, model_cfg)
        return super().load_state_dict(new_state_dict, strict)



"""
@Desc: Base class for [decoder-only] models.
"""
class FairseqLanguageModel(BaseFairseqModel):
    #@Param{decoder}: @Class{FairseqDecoder}, the decoder
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
        check_type(self.decoder, FairseqDecoder)

    """
    @Desc:  Run the forward pass for a decoder-only model. 
            Feeds a batch of tokens through the decoder to predict the next tokens.
    @Param{src_tokens}: @Type{LongTensor}, tokens on which to condition the decoder, @Shape{[batch, tgt_len]}
    @Param{src_lengths}:@Type{LongTensor}, source sentence lengths, @Shape{[batch]}
    @Return: @Tuple{
                the decoder's output, @Shape{[batch, seq_len, vocab]},
                a dictionary with any model-specific outputs
            }
    """
    def forward(self, src_tokens, **kwargs):
        return self.decoder(src_tokens, **kwargs)

    def forward_decoder(self, prev_output_tokens, **kwargs):
        return self.decoder(prev_output_tokens, **kwargs)

    #@Desc: Similar to @Func{forward} but only return features.
    #@Return: @Tuple{
    #           the decoder's features, @Shape{[batch, seq_len, embed_dim]}, 
    #           a dictionary with any model-specific outputs
    #       }
    def extract_features(self, src_tokens, **kwargs):
        return self.decoder.extract_features(src_tokens, **kwargs)
    
    #@Desc: Project features to the default output size (typically vocabulary size).
    def output_layer(self, features, **kwargs):
        """Project features to the default output size (typically vocabulary size)."""
        return self.decoder.output_layer(features, **kwargs)

    #@Desc: Maximum length supported by the model.
    def max_positions(self):
        return self.decoder.max_positions()

    #@Desc: Maximum length supported by the decoder.
    def max_decoder_positions(self):
        return self.decoder.max_positions()

    @property
    def supported_targets(self):
        return {"future"}


"""
@Desc: Base class for [encoder-only] models.
"""
class FairseqEncoderModel(BaseFairseqModel):
    #@Param{encoder}: @Type{BaseEncoder}, the encoder
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        check_type(self.encoder, FairseqEncoder)

    """
    @Desc:  Run the forward pass for a encoder-only model. 
            Feeds a batch of tokens through the encoder to generate features.
    @Param{src_tokens}: @Type{LongTensor}, input tokens, @Shape{[batch, src_len]}
    @Param{src_lengths}:@Type{LongTensor}, source sentence lengths, @Shape{[batch]}
    @Return: the encoder's output, typically of @Shape{[batch, src_len, features]}
    """
    def forward(self, src_tokens, src_lengths, **kwargs):
        return self.encoder(src_tokens, src_lengths, **kwargs)

    #@Desc: Get normalized probabilities (or log probs) from a net's output.
    def get_normalized_probs(self, net_output, log_probs, sample=None):
        encoder_out = net_output["encoder_out"]
        if torch.is_tensor(encoder_out):
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    #@Desc: Maximum length supported by the model.
    def max_positions(self):
        return self.encoder.max_positions()
