import torch
import os
from src.models.image_model import DMCI
from src.models.video_model import DMC
from src.utils.common import get_state_dict

print("average video decoding time: 38 ms/frame")
print("average audio decoding time: 10 ms/frame")
