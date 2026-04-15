"""srl.networks.encoders — encoder modules (MLP, CNN, LSTM, fusion, etc.)."""

# Register pre-trained encoders (resnet, efficientnet, vit, huggingface).
# Importing this sub-package triggers the @register_encoder decorators.
from srl.networks.encoders import pretrained  # noqa: F401
