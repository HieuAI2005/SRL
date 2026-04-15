"""srl.networks.encoders.pretrained — pre-trained backbone encoders.

Registers the following encoder types into ``EncoderRegistry``:

Vision — torchvision (requires ``torchvision``):
    - ``resnet``       ResNet-18 / 34 / 50 / 101 / 152
    - ``efficientnet`` EfficientNet-B0 … B7
    - ``vit``          ViT-B/16, ViT-B/32, ViT-L/16, ViT-L/32

Vision — HuggingFace Hub (requires ``transformers``):
    - ``hf_vision``    Any HuggingFace vision model (ViT, Swin, ConvNeXt, …)

Language (requires ``transformers``):
    - ``huggingface``  Any encoder-only HuggingFace model (BERT, DistilBERT, …)

Imports are intentionally lazy (deferred until the class is first
instantiated) so that missing optional dependencies do not raise an
``ImportError`` at package import time — only when you actually try to build
one of these encoders will you see the error.

The *registration* decorators are applied at module load, which is why we
import the modules here.
"""

from srl.networks.encoders.pretrained import huggingface  # noqa: F401
from srl.networks.encoders.pretrained import vision       # noqa: F401

__all__ = ["vision", "huggingface"]
