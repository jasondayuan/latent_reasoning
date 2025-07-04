from transformers import Trainer
from ..data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from ..models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation

class LatentTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        def data_collator(features):
            return features
        self.data_collator = data_collator

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # Because we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]
    
    def _prepare_inputs(self, inputs):
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # # Regular generation path
        # with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
        #     # Generate the prompt completion ids
        #     prompt_completion_ids = unwrapped_model.generate(
        #         input_ids=prompt_ids,
        #         attention_mask=prompt_mask,
        #         max_new_tokens=self.args.max_completion_length,
        #         do_sample=False,
        #         use_cache=True,
        #     )

        # # Compute prompt length and extract completion ids
        # prompt_length = prompt_ids.size(1)
        # prompt_ids = prompt_completion_ids[:, :prompt_length]
        # completion_ids = prompt_completion_ids[:, prompt_length:]

        import pdb; pdb.set_trace()