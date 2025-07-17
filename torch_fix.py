import torch
import torch.serialization
from ultralytics.nn.tasks import DetectionModel

# Add safe globals for ultralytics model loading
torch.serialization.add_safe_globals([DetectionModel])

# Alternative: Create a patched torch.load function
original_load = torch.load

def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = patched_load
