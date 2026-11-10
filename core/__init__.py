# Applied on first import of anything in core, before any module here loads
# ONNX Runtime or huggingface_hub.
from core.privacy import disable_telemetry

disable_telemetry()
