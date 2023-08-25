import json
import logging
from typing import Union, List, Dict, Any
import warnings

import torch
from torch.nn.modules import Dropout

import numpy
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import h5py
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.util import lazy_groups_of
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.modules.highway import Highway
from allennlp.modules.scalar_mix import ScalarMix
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.data import Batch
from allennlp.data import Token, Vocabulary, Instance
from allennlp.data.fields import TextField
from allennlp.modules.elmo import _ElmoCharacterEncoder
from contraction import *
import numpy as np
from holder import *


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



class ElmoBilmDebias(torch.nn.Module):
    """
    This is a customized version of the elmo bilm + debiasing at the first layer
    """
    def __init__(self,
                 options_file: str,
                 weight_file: str,
                 requires_grad: bool = False,
                 vocab_to_cache: List[str] = None) -> None:
        super(ElmoBilmDebias, self).__init__()

        self._token_embedder = _ElmoCharacterEncoder(options_file, weight_file, requires_grad=requires_grad)

        self._requires_grad = requires_grad
        if requires_grad and vocab_to_cache:
            logging.warning("You are fine tuning ELMo and caching char CNN word vectors. "
                            "This behaviour is not guaranteed to be well defined, particularly. "
                            "if not all of your inputs will occur in the vocabulary cache.")
        # This is an embedding, used to look up cached
        # word vectors built from character level cnn embeddings.
        self._word_embedding = None
        self._bos_embedding: torch.Tensor = None
        self._eos_embedding: torch.Tensor = None
        if vocab_to_cache:
            logging.info("Caching character cnn layers for words in vocabulary.")
            # This sets 3 attributes, _word_embedding, _bos_embedding and _eos_embedding.
            # They are set in the method so they can be accessed from outside the
            # constructor.
            self.create_cached_cnn_embeddings(vocab_to_cache)

        with open(cached_path(options_file), 'r') as fin:
            options = json.load(fin)
        if not options['lstm'].get('use_skip_connections'):
            raise ConfigurationError('We only support pretrained biLMs with residual connections')
        self._elmo_lstm = ElmoLstm(input_size=options['lstm']['projection_dim'],
                                   hidden_size=options['lstm']['projection_dim'],
                                   cell_size=options['lstm']['dim'],
                                   num_layers=options['lstm']['n_layers'],
                                   memory_cell_clip_value=options['lstm']['cell_clip'],
                                   state_projection_clip_value=options['lstm']['proj_clip'],
                                   requires_grad=requires_grad)
        self._elmo_lstm.load_weights(weight_file)
        # Number of representation layers including context independent layer
        self.num_layers = options['lstm']['n_layers'] + 1

    def get_output_dim(self):
        return 2 * self._token_embedder.get_output_dim()

    def forward(self,  # pylint: disable=arguments-differ
                inputs: torch.Tensor,
                bias: torch.Tensor = None,
                num_bias: int = 1,
                contraction: (torch.Tensor, torch.Tensor) = None,
                word_inputs: torch.Tensor = None) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
            Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape ``(batch_size, timesteps)``,
            which represent word ids which have been pre-cached.
        Returns
        -------
        Dict with keys:
        ``'activations'``: ``List[torch.Tensor]``
            A list of activations at each layer of the network, each of shape
            ``(batch_size, timesteps + 2, embedding_dim)``
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, timesteps + 2)`` long tensor with sequence mask.
        Note that the output tensors all include additional special begin and end of sequence
        markers.
        """
        if self._word_embedding is not None and word_inputs is not None:
            try:
                mask_without_bos_eos = (word_inputs > 0).long()
                # The character cnn part is cached - just look it up.
                embedded_inputs = self._word_embedding(word_inputs) # type: ignore
                # shape (batch_size, timesteps + 2, embedding_dim)
                type_representation, mask = add_sentence_boundary_token_ids(
                        embedded_inputs,
                        mask_without_bos_eos,
                        self._bos_embedding,
                        self._eos_embedding
                )
            except RuntimeError:
                # Back off to running the character convolutions,
                # as we might not have the words in the cache.
                token_embedding = self._token_embedder(inputs)
                mask = token_embedding['mask']
                type_representation = token_embedding['token_embedding']
        else:
            token_embedding = self._token_embedder(inputs)
            mask = token_embedding['mask']
            type_representation = token_embedding['token_embedding']
        
        # debiasing the input embeddings
        #   1. take out the boundaries, i.e. len - 2
        batch_l, seq_l, elmo_size = type_representation.shape
        l0 = type_representation[:, 1:-1, :]
        #   2. debiasing
        if bias is not None:
            if num_bias == 1:
                bias = bias.expand(batch_l, 1, elmo_size)
                proj = l0.bmm(bias.transpose(1,2))
                l0 = l0 - (proj * bias)
            elif num_bias == 2:
                bias = bias.expand(batch_l, 2, elmo_size)
                bias1 = bias[:, 0:1, :]
                bias2 = bias[:, 1:2, :]
                proj1 = l0.bmm(bias1.transpose(1,2))
                proj2 = l0.bmm(bias2.transpose(1,2))
                l0 = l0 - (proj1 * bias1) - (proj2 * bias2)
            else:
                raise Exception('unrecognized num_bias: {0}'.format(num_bias))
        #   3. contraction
        if contraction is not None:
            if not hasattr(self, 'contract_U'):
                v1 = contraction[0].view(-1, elmo_size).cpu().numpy()
                v2 = contraction[1].view(-1, elmo_size).cpu().numpy()
    
                v1, v2 = maxSpan(v1, v2)
                U = np.identity(elmo_size)
                U = gsConstrained(U, v1, basis(np.vstack((v1, v2))))

                self.contract_v1 = torch.from_numpy(v1).view(1, 1, elmo_size)
                self.contract_v2 = torch.from_numpy(v2).view(1, 1, elmo_size)
                self.contract_U = torch.from_numpy(U).view(1, elmo_size, elmo_size).float()
                gpuid = contraction[0].get_device()
                if gpuid != -1:
                    self.contract_v1 = self.contract_v1.cuda(gpuid)
                    self.contract_v2 = self.contract_v2.cuda(gpuid)
                    self.contract_U =  self.contract_U.cuda(gpuid)

            opt = Holder()
            opt.gpuid = contraction[0].get_device()
            l0 = correction(opt, self.contract_U, self.contract_v1, self.contract_v2, l0.contiguous())

        #   4. reconcat with boundaries
        type_representation = torch.cat([type_representation[:, 0:1, :], l0, type_representation[:, -1:, :]], 1)

        # continue the lm
        lstm_outputs = self._elmo_lstm(type_representation, mask)

        # Prepare the output.  The first layer is duplicated.
        # Because of minor differences in how masking is applied depending
        # on whether the char cnn layers are cached, we'll be defensive and
        # multiply by the mask here. It's not strictly necessary, as the
        # mask passed on is correct, but the values in the padded areas
        # of the char cnn representations can change.
        output_tensors = [
                torch.cat([type_representation, type_representation], dim=-1) * mask.float().unsqueeze(-1)
        ]
        for layer_activations in torch.chunk(lstm_outputs, lstm_outputs.size(0), dim=0):
            output_tensors.append(layer_activations.squeeze(0))

        return {
                'activations': output_tensors,
                'mask': mask,
        }