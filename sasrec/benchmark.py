import argparse
import time
import torch
import numpy as np
from tqdm import tqdm
from sasrec.model import SASRec
from sasrec.inference import SASRecPredictor, ONNXPredictor
import onnxruntime as ort


class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        hidden = self.model(input_ids)
        lengths = ((input_ids != 0).sum(dim=1) - 1).clamp(min=0)
        lengths_expanded = (
            lengths.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden.size(2))
        )
        last_hidden = torch.gather(hidden, 1, lengths_expanded).squeeze(1)
        all_scores = torch.matmul(last_hidden, self.model.item_emb.weight.T)
        return all_scores


def export_onnx(model, maxlen, batch_size, onnx_path):
    print("Exporting model to ONNX...")
    model.eval()
    wrapper = ONNXExportWrapper(model)
    dummy_input = torch.randint(1, 100, (batch_size, maxlen), dtype=torch.long)
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_ids"],
        output_names=["scores"],
        dynamic_axes={"input_ids": {0: "batch_size"}, "scores": {0: "batch_size"}},
    )
    print(f"Exported successfully to {onnx_path}")


def generate_dummy_data(num_samples, maxlen):
    return torch.randint(1, 100, (num_samples, maxlen), dtype=torch.long)


def run_latency(predictor, maxlen, num_requests=1000):
    print("--- LATENCY BENCHMARK (batch_size=1) ---")
    data = generate_dummy_data(num_requests, maxlen)
    for i in range(100):
        _ = predictor.predict(data[i : i + 1])

    latencies = []
    for i in tqdm(range(num_requests), desc="Measuring latency"):
        batch = data[i : i + 1]
        start = time.perf_counter()
        _ = predictor.predict(batch)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)

    latencies = np.array(latencies)
    print(f"p50: {np.percentile(latencies, 50):.2f} ms")
    print(f"p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"p99: {np.percentile(latencies, 99):.2f} ms")
    print()


def run_throughput(predictor, maxlen, num_batches=100):
    print("--- DYNAMIC THROUGHPUT BENCHMARK ---")
    batch_sizes = [64, 128, 256, 512, 1024, 2048]

    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        data = generate_dummy_data(num_batches * batch_size, maxlen)

        try:
            # Warmup
            _ = predictor.predict(data[:batch_size])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            for i in tqdm(range(num_batches), desc=f"BS={batch_size}", leave=False):
                batch = data[i * batch_size : (i + 1) * batch_size]
                _ = predictor.predict(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()

            users_sec = (num_batches * batch_size) / (end - start)
            print(f"-> Throughput (BS={batch_size}): {users_sec:.0f} users/sec")

        except (RuntimeError, Exception) as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                print(
                    f"-> OOM (Out of Memory) at batch size {batch_size}. Stopping search."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            else:
                print(f"-> Error at batch size {batch_size}: {e}")
                break
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["latency", "throughput", "all"], default="all"
    )
    parser.add_argument(
        "--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda"
    )
    parser.add_argument("--no-onnx", action="store_true")

    parser.add_argument("--item_num", default=50000, type=int)
    parser.add_argument("--max_length", default=200, type=int)
    parser.add_argument("--hidden_units", default=256, type=int)
    parser.add_argument("--num_blocks", default=2, type=int)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--attn_types", default=None, type=str)
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    attn_types = args.attn_types.split(",") if args.attn_types is not None else None

    model = SASRec(
        item_num=args.item_num,
        maxlen=args.max_length,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        attn_types=attn_types,
    )

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    model.to(args.device)
    use_onnx = not args.no_onnx

    print("\n" + "=" * 60)
    print("SASRec Benchmark Configuration")
    print("=" * 60)
    print(f"Mode         : {args.mode.upper()}")
    print(f"Device       : {args.device.upper()}")
    print(f"Use ONNX     : {use_onnx}")
    print(f"Checkpoint   : {args.checkpoint or 'Random Initialization'}")
    print(
        f"Architecture : {args.num_blocks} blocks, {args.num_heads} heads, d={args.hidden_units}"
    )
    print(f"Context Len  : {args.max_length}")
    print(f"Catalog Size : {args.item_num:,}")
    print("=" * 60 + "\n")

    if use_onnx:
        onnx_file = "sasrec_model.onnx"
        export_onnx(
            model.cpu(), maxlen=args.max_length, batch_size=1, onnx_path=onnx_file
        )
        predictor = ONNXPredictor(onnx_file, device=args.device)
    else:
        predictor = SASRecPredictor(model, args.device)

    if args.mode in ["latency", "all"]:
        run_latency(predictor, maxlen=args.max_length)

    if args.mode in ["throughput", "all"]:
        run_throughput(predictor, maxlen=args.max_length)
