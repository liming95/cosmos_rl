import torch
import re
from typing import List, Tuple
from cosmos_rl.policy.model.base import WeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig
from functools import cached_property


class Alpamayo15ModelWeightMapper(WeightMapper):
    pass