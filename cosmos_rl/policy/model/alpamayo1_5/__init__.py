from cosmos_rl.policy.model.base import BaseModel, ModelRegistry
from cosmos_rl.policy.model.alpamayo1_5.weight_mapper import Alpamayo15ModelWeightMapper
from my_examples.alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


from cosmos_rl.utils.parallelism import ParallelDims

import torch
from transformers import AutoConfig

@ModelRegistry.register(Alpamayo15ModelWeightMapper)
class Alpamayo15_VLAModel(BaseModel):
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.model = self.model.to(dtype=torch.get_default_dtype())  #'cuda'

    @staticmethod
    def supported_model_types():
        raise NotImplementedError

    @property
    def parallelize_fn(self):
        raise NotImplementedError

    def apply_pipeline_split(self, pp_rank, pp_size):
        raise NotImplementedError
    
    def get_position_ids(self, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Method to get the position ids of the model.
        This function is declared due to that `Context Parallelism`
        requires the shuffle of both `input_ids` and `position_ids`.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]:
                - Tensor of position ids
                - Tensor of input ids
                - Sequence dimension index of position ids.
        """
        raise NotImplementedError

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims: ParallelDims,
        device: torch.device,
        revision: Optional[str] = None,
    ):
        """
        Load weights from a HuggingFace model.

        Args:
            model_name_or_path (str): The name or path of the model.
            parallel_dims (ParallelDims): The parallel dimensions.
            device (torch.device): The device to load the weights.
        """
        raise NotImplementedError

    def separate_model_parts(self) -> List[torch.nn.Module]:
        """
        Model parts that should be trained in separate optimizers. (i.e. Multi-model training)
        """
        raise NotImplementedError

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: Optional[int] = None,
    ) -> "Alpamayo15_VLAModel":
        dtype = hf_config.model_dtype

        model = Alpamayo1_5.from_pretrained(model_name_or_path, dtype=dtype)
        return cls(hf_config, model)

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """
        Get the number of parameters and flops of the model.
        Args:
            seq_len (int): The sequence length of the model.
        Returns:
            tuple[int, int]: The number of parameters and flops of the model.
        """
        raise NotImplementedError
    

    
