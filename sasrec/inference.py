import torch
import numpy as np


class SASRecPredictor:
    """Standard PyTorch inference predictor for SASRec."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor) -> np.ndarray:
        input_ids = input_ids.to(self.device)
        hidden = self.model(input_ids)
        lengths = (input_ids != 0).sum(dim=1) - 1
        rows = torch.arange(input_ids.size(0), device=self.device)
        last_hidden = hidden[rows, lengths]

        all_scores = torch.matmul(last_hidden, self.model.item_emb.weight.T)
        return all_scores.cpu().numpy()


class ONNXPredictor:
    def __init__(self, onnx_path: str, device: str = "cpu"):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        if "cuda" in str(device).lower():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        so = ort.SessionOptions()
        so.log_severity_level = 3  # Suppress memcpy warnings

        self.session = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, input_ids: torch.Tensor) -> np.ndarray:
        inp = {self.input_name: input_ids.numpy()}
        all_scores = self.session.run(None, inp)[0]
        return all_scores
