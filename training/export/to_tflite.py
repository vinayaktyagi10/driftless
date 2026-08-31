"""Export a trained checkpoint to TFLite for the Android app
(android/app/src/main/assets/models/) and to ONNX for the C++ edge engine
(edge-engine/). PyTorch -> ONNX -> TFLite, via onnx2tf, since PyTorch has no
first-party TFLite exporter.
"""

if __name__ == "__main__":
    raise NotImplementedError("wire up once a checkpoint exists")
