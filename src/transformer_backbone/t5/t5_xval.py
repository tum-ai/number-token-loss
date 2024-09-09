import inspect
from typing import Optional, Tuple, Union, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn.init as init
from transformers import T5ForConditionalGeneration, LogitsProcessorList, StoppingCriteriaList, GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateNonBeamOutput, GenerateEncoderDecoderOutput, GenerateDecoderOnlyOutput
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.utils import logging, is_torch_fx_proxy, is_accelerate_available, ModelOutput

from transformers.generation.beam_search import BeamScorer, BeamSearchScorer
from transformers.generation.utils import GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput, _split_model_inputs, _split_model_outputs, stack_model_outputs

from src.tokenizer.abstract_tokenizer import NumberEncodingTokenizer

if is_accelerate_available():
    from accelerate.hooks import AlignDevicesHook, add_hook_to_module

logger = logging.get_logger(__name__)

GenerateBeamOutput = Union[GenerateBeamDecoderOnlyOutput, GenerateBeamEncoderDecoderOutput]


# Set maximal value for normalization 
V_MAX = 3000000000


class T5RegressionModelXval(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer: NumberEncodingTokenizer, dim_feedforward=1024, numhead_bias=True):
        super().__init__(config)
        super()._resize_token_embeddings(config.vocab_size)

        self.tokenizer = tokenizer

        self.num_head = nn.Sequential(
            nn.Linear(config.d_model, dim_feedforward, bias=numhead_bias),
            nn.GELU(),
            nn.Linear(dim_feedforward, 1, bias=numhead_bias),
        )

    def initialize_num_head_weights(self):
        for layer in self.num_head:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_number_embeddings: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_number_embeddings: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            number_labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        """
        Overwrites forward method of parent class T5ForConditionalGeneration.
        Computes embeddings from input_ids and directly passes them to encoder and decoder.
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                raise ValueError("Not supported")

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            #######################
            # Customized code start
            #######################
            if inputs_embeds is not None:
                raise ValueError("inputs_embeds are not None")
            # Convert encoder inputs in embeddings if needed
            # Normalize embeddings
            number_locs = input_ids == self.tokenizer.num_token_id
            input_number_embeddings[number_locs] = torch.sign(input_number_embeddings[number_locs]) * torch.log10(torch.abs(input_number_embeddings[number_locs]) + 1) / (torch.log10(torch.tensor(V_MAX)) / 10)
            
            inputs_embeds = self.shared(input_ids) * input_number_embeddings.unsqueeze(-1)
            encoder_outputs = self.encoder(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            #######################
            # Customized code end
            #######################

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            #######################
            # Customized code start
            #######################
            decoder_input_ids, decoder_number_embeddings = self._shift_right(labels, number_labels)
            #######################
            # Customized code end
            #######################

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        #######################
        # Customized code start
        #######################
        if decoder_inputs_embeds is not None:
            raise ValueError("decoder_inputs_embeds are not None")
            # Convert encoder inputs in embeddings if needed
        # Normalize embeddings
        number_locs = decoder_input_ids == self.tokenizer.num_token_id
        if number_locs.any():
            decoder_number_embeddings[number_locs] = torch.sign(decoder_number_embeddings[number_locs]) * torch.log10(torch.abs(decoder_number_embeddings[number_locs]) + 1) / (torch.log10(torch.tensor(V_MAX)) / 10)
        
        decoder_inputs_embeds = self.shared(decoder_input_ids) * decoder_number_embeddings.unsqueeze(-1)

        # Decode
        decoder_outputs = self.decoder(
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        #######################
        # Customized code end
        #######################

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        outputs = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

        #######################
        # Customized code start
        #######################

        # Compute number predictions and loss

        num_preds = self.num_head(sequence_output)

        if number_labels is not None:
            num_mask = labels == self.tokenizer.convert_tokens_to_ids(self.tokenizer.num_token)
            loss_num = F.mse_loss(
                num_preds[num_mask],
                number_labels[num_mask].view(-1, 1),
                reduction="mean",
            ) * 0.001
            outputs["number_loss"] = loss_num
            outputs["token_loss"] = outputs.loss
            loss = loss + loss_num
            outputs.loss = loss

        outputs["number_predictions"] = num_preds

        return outputs

    #######################
    # Customized code end
    #######################

    def _shift_right(self, input_ids, input_number_embeddings):
        """
        Overwritten to also shift the input_number_embeddings.
        """
        decoder_start_token_id = self.config.decoder_start_token_id
        decoder_start_number_embedding = 1
        pad_token_id = self.config.pad_token_id

        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. "
                "See T5 docs for more information."
            )

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
           raise ValueError("Not supported")
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

            #######################
            # Customized code start
            #######################
            shifted_input_number_embeddings = input_number_embeddings.new_zeros(input_number_embeddings.shape)
            shifted_input_number_embeddings[..., 1:] = input_number_embeddings[..., :-1].clone()
            shifted_input_number_embeddings[..., 0] = decoder_start_number_embedding
            #######################
            # Customized code end
            #######################

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids, shifted_input_number_embeddings

    def _prepare_encoder_decoder_kwargs_for_generation(
            self,
            inputs_tensor: torch.Tensor,
            model_kwargs,
            model_input_name: Optional[str],
            generation_config: GenerationConfig,
    ) -> Dict[str, Any]:
        """
        Overwritten from T5ForConditionalGeneration to feed custom number embeddings to the encoder.
        """
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(self, "hf_device_map"):
            if hasattr(encoder, "_hf_hook"):
                encoder._hf_hook.io_same_device = True
            else:
                add_hook_to_module(encoder, AlignDevicesHook(io_same_device=True))

        # 2. Prepare encoder args and encoder kwargs from model kwargs and generation config.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {
                argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
            }
        encoder_kwargs["output_attentions"] = generation_config.output_attentions
        encoder_kwargs["output_hidden_states"] = generation_config.output_hidden_states

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        #######################
        # Customized code start
        #######################
        encoder_input_ids = encoder_kwargs.pop("input_ids", None)
        encoder_input_number_embeddings = model_kwargs.pop("input_number_embeddings", None)
        
        # Normalize embeddings
        number_locs = encoder_input_ids == self.tokenizer.num_token_id
        encoder_input_number_embeddings[number_locs] = torch.sign(encoder_input_number_embeddings[number_locs]) * torch.log10(torch.abs(encoder_input_number_embeddings[number_locs]) + 1) / (torch.log10(torch.tensor(V_MAX)) / 10)
        
        encoder_kwargs["inputs_embeds"] = self.shared(encoder_input_ids) * encoder_input_number_embeddings.unsqueeze(-1)
        #######################
        # Customized code end
        #######################

        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    def _prepare_decoder_input_ids_for_generation(
            self,
            batch_size: int,
            model_input_name: str,
            model_kwargs: Dict[str, torch.Tensor],
            decoder_start_token_id: torch.Tensor,
            device: torch.device = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """
        Overwritten to add decoder_number_embeddings as model input.
        """
        # 1. Check whether the user has defined `decoder_input_ids` manually. To facilitate in terms of input naming,
        # we also allow the user to pass it under `input_ids`, if the encoder does not use it as the main input.
        if model_kwargs is not None and "decoder_input_ids" in model_kwargs:
            decoder_input_ids = model_kwargs.pop("decoder_input_ids")
        elif "input_ids" in model_kwargs and model_input_name != "input_ids":
            decoder_input_ids = model_kwargs.pop("input_ids")
        else:
            decoder_input_ids = None

        # 2. `decoder_start_token_id` must have shape (batch_size, 1)
        if device is None:
            device = self.device
        if decoder_start_token_id.ndim == 1:
            if decoder_start_token_id.shape[0] != batch_size:
                raise ValueError(
                    f"`decoder_start_token_id` expected to have length {batch_size} but got {decoder_start_token_id.shape[0]}"
                )
            decoder_start_token_id = decoder_start_token_id.view(-1, 1)
        else:
            decoder_start_token_id = (
                    torch.ones((batch_size, 1), dtype=torch.long, device=device) * decoder_start_token_id
            )

        #######################
        # Customized code start
        #######################
        number_embeddings_start = torch.ones((batch_size, 1), dtype=torch.long, device=device)

        # 3. Encoder-decoder models expect the `decoder_input_ids` to start with a special token. Let's ensure that.
        # no user input -> use decoder_start_token_id as decoder_input_ids
        if decoder_input_ids is None:
            decoder_input_ids = decoder_start_token_id
            model_kwargs["decoder_number_embeddings"] = number_embeddings_start
        # user input but doesn't start with decoder_start_token_id -> prepend decoder_start_token_id (and adjust
        # decoder_attention_mask if provided)
        elif (decoder_input_ids[:, 0] != decoder_start_token_id[:, 0]).all().item():
            decoder_input_ids = torch.cat([decoder_start_token_id, decoder_input_ids], dim=-1)
            model_kwargs["decoder_number_embeddings"] = torch.cat([number_embeddings_start, model_kwargs["decoder_number_embeddings"]], dim=-1)
        #######################
        # Customized code end
        #######################
            if "decoder_attention_mask" in model_kwargs:
                decoder_attention_mask = model_kwargs["decoder_attention_mask"]
                decoder_attention_mask = torch.cat(
                    (torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask),
                    dim=-1,
                )
                model_kwargs["decoder_attention_mask"] = decoder_attention_mask

        return decoder_input_ids, model_kwargs

    def prepare_inputs_for_generation(
            self,
            input_ids,
            decoder_number_embeddings=None,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs,
    ):
        """
        Overwritten from T5ForConditionalGeneration to add decoder_number_embeddings as model input.
        """

        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            decoder_number_embeddings = decoder_number_embeddings[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "decoder_number_embeddings": decoder_number_embeddings,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def _sample(
            self,
            input_ids: torch.LongTensor,
            logits_processor: LogitsProcessorList,
            stopping_criteria: StoppingCriteriaList,
            generation_config: GenerationConfig,
            synced_gpus: bool,
            streamer: Optional[BaseStreamer],
            logits_warper: Optional[LogitsProcessorList] = None,
            **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        """
        Overwritten to handle number embeddings in the generation loop.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if do_sample:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            #######################
            # Customized code start
            #######################
            # Next number_embeddings should be "number_predictions" from model output if
            # the corresponding token is a number token, otherwise it should be 1
            next_number_embeddings = torch.where(
                next_tokens == self.tokenizer.get_num_token_ids()[0],
                outputs["number_predictions"].squeeze(),
                torch.ones_like(outputs["number_predictions"].squeeze()),
            )

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
                next_number_embeddings = next_number_embeddings * unfinished_sequences + 1 * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs["decoder_number_embeddings"] = torch.cat([model_kwargs["decoder_number_embeddings"], next_number_embeddings[:, None]], dim=-1)
            #######################
            # Customized code end
            #######################
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                    num_predictions=model_kwargs.get("decoder_number_embeddings"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids, model_kwargs.get("decoder_number_embeddings")

        

    def _beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool,
        logits_warper: Optional[LogitsProcessorList] = None,
        **model_kwargs,
    ) -> Union[GenerateBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`:
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step. Only required with sampling strategies (i.e. `do_sample` is set in
                `generation_config`)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.GenerateBeamDecoderOnlyOutput`], [`~generation.GenerateBeamEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateBeamDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateBeamEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config.pad_token_id
        eos_token_id = generation_config.eos_token_id
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        sequential = generation_config.low_memory
        do_sample = generation_config.do_sample
        if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
            raise ValueError(
                "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                f"{logits_warper})."
            )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape
        model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        #######################
        # Customized code start
        #######################
        # changed condition
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)))
        #######################
        # Customized code end
        #######################
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False

        decoder_prompt_len = input_ids.shape[-1]  # record the prompt length of decoder

        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # if sequential is True, split the input to batches of batch_size and run sequentially
            if sequential:
                if any(
                    model_name in self.__class__.__name__.lower()
                    for model_name in [
                        "fsmt",
                        "reformer",
                        "bloom",
                        "ctrl",
                        "gpt_bigcode",
                        "transo_xl",
                        "xlnet",
                        "cpm",
                        "jamba",
                    ]
                ):
                    raise RuntimeError(
                        f"Currently generation for {self.__class__.__name__} is not supported "
                        f"for `low_memory beam_search`. Please open an issue on GitHub if you need this feature."
                    )

                inputs_per_sub_batches = _split_model_inputs(
                    model_inputs, split_size=batch_size, full_batch_size=batch_beam_size
                )
                outputs_per_sub_batch = [
                    self(
                        **inputs_per_sub_batch,
                        return_dict=True,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                    )
                    for inputs_per_sub_batch in inputs_per_sub_batches
                ]

                outputs = stack_model_outputs(outputs_per_sub_batch)

            else:  # Unchanged original behavior
                outputs = self(
                    **model_inputs,
                    return_dict=True,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            # Clone is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].clone()
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            if do_sample:
                next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)
                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Beam token selection: pick 1 + eos_token_id.shape[0] next tokens for each beam so we have at least 1
            # non eos token per beam.
            n_eos_tokens = eos_token_id.shape[0] if eos_token_id is not None else 0
            n_tokens_to_keep = max(2, 1 + n_eos_tokens) * num_beams
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=n_tokens_to_keep)
                next_token_scores = torch.gather(next_token_scores, -1, next_tokens)
                next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, _indices)
            else:
                next_token_scores, next_tokens = torch.topk(
                    next_token_scores, n_tokens_to_keep, dim=1, largest=True, sorted=True
                )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
                decoder_prompt_len=decoder_prompt_len,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]
            #######################
            # Customized code start
            #######################
            # Next number_embeddings should be "number_predictions" from model output
            next_number_embeddings = torch.where(
                beam_next_tokens == self.tokenizer.get_num_token_ids()[0],
                outputs["number_predictions"].squeeze(),
                torch.ones_like(outputs["number_predictions"].squeeze()),
                )
            
            # select the number embeddings corresponding to the selected beams
            # next_number_embeddings = next_number_embeddings[beam_idx]

            if not (next_number_embeddings[beam_next_tokens != self.tokenizer.get_num_token_ids()[0]] == 1).all():
                raise Exception("Number embeddings should be 1 for non-number tokens")

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
            model_kwargs["decoder_number_embeddings"] = torch.cat([model_kwargs["decoder_number_embeddings"][beam_idx, :], next_number_embeddings[:, None]], dim=-1)

            if not (model_kwargs["decoder_number_embeddings"][input_ids != self.tokenizer.get_num_token_ids()[0]] == 1).all():
                raise Exception("Number embeddings should be 1 for non-number tokens")

            #####################
            # Customized code end
            #####################
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            # IMPORTANT: Note that this should appear BEFORE the call to _reorder_cache() to save the maximum memory
            # (that way the memory peak does not include outputs.logits)
            del outputs

            if model_kwargs.get("past_key_values", None) is not None:
                model_kwargs["past_key_values"] = self._temporary_reorder_cache(
                    model_kwargs["past_key_values"], beam_idx
                )
            #######################
            # Customized code start
            #######################
            # only commented out condition, so that beam_indices are always saved and can be used for extracting the number embeddings later
            #if return_dict_in_generate and output_scores:
            beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))
            #######################
            # Customized code end
            #######################
            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or all(stopping_criteria(input_ids, scores)):
                this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
            decoder_prompt_len=decoder_prompt_len,
        )

        #######################
        # Customized code start
        #######################

        # As the finalize method of the beam_scorer does not return the index of the highest scoring beam, we need to extract it manually again.
        # Alternatively, the beam scorer and generate function could be overwritten to return the index of the highest scoring beam.
        # The follwoing code is copied from the finalize method of the beam scorer and adapted to return the index of the highest scoring beam.
        # The beams should already be finalized and we just have to retrieve the best again.

        idx_final_beams = []
        for i in range(batch_size):
            beam_hyps_in_batch = beam_scorer._beam_hyps[i * beam_scorer.num_beam_groups : (i + 1) * beam_scorer.num_beam_groups]
            candidate_beams = [beam for beam_hyp in beam_hyps_in_batch for beam in beam_hyp.beams]

            # get the index of the highest scoring beam
            best_beam_idx = max(enumerate(candidate_beams), key=lambda x: x[1][0])[0]
            idx_final_beams.append(best_beam_idx+i*num_beams)

        # extract the number embeddings corresponding to the highest scoring beams
        sequence_outputs['number_embeddings'] = model_kwargs["decoder_number_embeddings"][idx_final_beams]

        if not (sequence_outputs["number_embeddings"][sequence_outputs['sequences'] != self.tokenizer.get_num_token_ids()[0]] == 1).all():
            raise Exception("Number embeddings should be 1 for non-number tokens")

        #######################
        # Customized code end
        #######################


        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return GenerateBeamEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateBeamDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    logits=raw_logits,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return sequence_outputs["sequences"], sequence_outputs["number_embeddings"]