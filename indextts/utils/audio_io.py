import warnings
from typing import Tuple

import torch
import torchaudio


def safe_torchaudio_load(
    path: str,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, int]:
    """Load audio with torchaudio, falling back when torchcodec is missing."""
    try:
        return torchaudio.load(path, *args, **kwargs)
    except Exception:
        warnings.warn(">> TorchCodec backend is unavailable; falling back to soundfile-based loader.")
        
        import soundfile as sf
        audio_np, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = torch.from_numpy(audio_np.T)
        return audio, sr
