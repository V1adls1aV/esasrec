import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from sasrec.data import (
    CausalLMDataset,
    PaddingCollateFn,
    download_and_preprocess,
    load_data,
    split_leave_one_out,
)
from sasrec.evaluate import evaluate, validate_fast
from sasrec.losses import (
    compute_full_softmax_loss,
    compute_sampled_bce_loss,
    compute_sampled_ce_loss,
)
from sasrec.model import SASRec


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one_epoch(
    model, dataloader, optimizer, device, loss_type="cross_entropy", use_sampling=False
):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        if use_sampling:
            hidden = model(input_ids)
            negatives = batch["negatives"].to(device)
            if loss_type == "cross_entropy":
                loss = compute_sampled_ce_loss(
                    hidden, labels, negatives, model.item_emb
                )
            else:
                loss = compute_sampled_bce_loss(
                    hidden, labels, negatives, model.item_emb
                )
        else:
            hidden = model(input_ids)
            logits = torch.matmul(hidden, model.item_emb.weight.T)
            loss = compute_full_softmax_loss(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="ml-20m", choices=["ml-1m", "ml-20m"]
    )
    parser.add_argument("--data_dir", type=str, default="data")

    parser.add_argument("--hidden_units", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=200)
    parser.add_argument(
        "--attn_types",
        type=str,
        default=None,
        help="Comma-separated attention types, e.g., 'standard,linear'. Implies num_blocks.",
    )

    parser.add_argument(
        "--loss", type=str, default="cross_entropy", choices=["cross_entropy", "bce"]
    )
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--full_negative_sampling", action="store_true")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)

    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--top_k", type=int, nargs="+", default=[10])

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_best", action="store_true")
    parser.add_argument("--resume_path", type=str, default=None)
    parser.add_argument("--layers_mask", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "training.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)],
    )
    print = logging.info

    print("=" * 70)
    print("SASRec+ Training Script")
    print("=" * 70)

    if args.data_path is None:
        args.data_path = download_and_preprocess(
            dataset_name=args.dataset, output_dir=args.data_dir
        )

    print("\nConfig:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    user_sequences, num_items = load_data(args.data_path)
    (train_sequences, val_sequences, val_targets, test_sequences, test_targets) = (
        split_leave_one_out(user_sequences)
    )

    train_seqs_list = list(train_sequences.values())

    use_sampling = args.num_negatives is not None
    full_neg = args.full_negative_sampling if use_sampling else False

    train_dataset = CausalLMDataset(
        train_seqs_list,
        max_length=args.max_length,
        num_negatives=args.num_negatives,
        full_negative_sampling=full_neg,
        num_items=num_items,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=PaddingCollateFn(),
        pin_memory=True,
    )

    if args.attn_types is not None:
        attn_types = args.attn_types.split(",")
        if len(attn_types) != args.num_blocks:
            print(f"Overwriting num_blocks from {args.num_blocks} to {len(attn_types)}")
            args.num_blocks = len(attn_types)
    else:
        attn_types = ["standard"] * args.num_blocks

    if args.layers_mask is not None:
        arguments = args.layers_mask.split(",")
        layers_mask = arguments[0]
        starting_epoch = int(arguments[1])
    else:
        layers_mask = None

    model = SASRec(
        item_num=num_items,
        maxlen=args.max_length,
        hidden_units=args.hidden_units,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout_rate=args.dropout_rate,
        attn_types=attn_types,
        layers_mask=layers_mask,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    best_path = os.path.join(args.save_dir, "best_model.pt")
    checkpoint_path = args.resume_path or best_path
    best_ndcg = -1.0
    epochs_no_improve = 0
    start_epoch = 1

    if args.resume_best:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_ndcg = checkpoint.get("best_ndcg", best_ndcg)
            epochs_no_improve = checkpoint.get("epochs_no_improve", 0)
            start_epoch = checkpoint.get("epoch", 0) + 1
        else:
            model.load_state_dict(checkpoint)
        print(f"Resumed training from {checkpoint_path} (start epoch: {start_epoch})")

    print(f"\n{'=' * 70}")
    print("Starting training...")
    eval_start = time.time()
    print(f"{'=' * 70}\n")

    start_time = time.time()

    epoch_count = 0
    for epoch in range(start_epoch, start_epoch + args.max_epochs):
        if layers_mask is not None and epoch == starting_epoch:
            before = list(model.layers_mask)
            model.layers_mask = layers_mask = [1] * model.num_blocks
            print(
                f"\nСменил маску слоев с {''.join(map(str, before))} на {''.join(map(str, model.layers_mask))}\n"
            )
        epoch_start = time.time()
        epoch_count += 1

        avg_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_type=args.loss,
            use_sampling=use_sampling,
        )

        epoch_time = time.time() - epoch_start

        val_metrics = validate_fast(
            model,
            val_sequences,
            val_targets,
            num_items,
            args.max_length,
            device,
            k=10,
            batch_size=256,
            max_users=args.val_size,
        )

        val_ndcg = val_metrics["NDCG@10"]
        val_hr = val_metrics["HR@10"]

        improved = val_ndcg > best_ndcg
        if improved:
            best_ndcg = val_ndcg
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_ndcg": best_ndcg,
                    "epochs_no_improve": epochs_no_improve,
                    "args": vars(args),
                },
                best_path,
            )
            marker = " * (saved)"
        else:
            epochs_no_improve += 1
            marker = ""

        print(
            f"Epoch {epoch:3d}/{start_epoch + args.max_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val HR@10: {val_hr:.4f} | "
            f"Val NDCG@10: {val_ndcg:.4f} | "
            f"Time: {epoch_time:.1f}s{marker}"
        )

        if epochs_no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch - args.patience} (best result)")
            break

    total_time = time.time() - start_time
    if epoch_count:
        print(f"\nTotal training time: {total_time:.1f}s ({total_time / 60:.1f}min)")
        print(f"\nMean training time: {total_time / epoch_count:.1f}s")
        print(f"Best validation NDCG@10: {best_ndcg:.4f}")

    print(f"\n{'=' * 70}")
    print("Final evaluation on test set (full-catalog ranking)...")
    print(f"{'=' * 70}\n")

    best_checkpoint = load_checkpoint(best_path, device)
    if isinstance(best_checkpoint, dict) and "model_state_dict" in best_checkpoint:
        model.load_state_dict(best_checkpoint["model_state_dict"])
    else:
        model.load_state_dict(best_checkpoint)

    for k in args.top_k:
        print(f"\n--- Test metrics @{k} ---")
        eval_time = time.time() - eval_start
        test_metrics = evaluate(
            model,
            test_sequences,
            test_targets,
            num_items,
            args.max_length,
            device,
            k=k,
            batch_size=256,
            filter_seen=True,
        )
        for name, value in test_metrics.items():
            print(f"  {name}: {value:.4f}")
        print(f"\nEvaluation time: {time.time() - eval_time:.1f}s")

    print("\n--- Validation metrics @10 (best model) ---")
    val_final = evaluate(
        model,
        val_sequences,
        val_targets,
        num_items,
        args.max_length,
        device,
        k=10,
        batch_size=256,
        filter_seen=True,
    )
    for name, value in val_final.items():
        print(f"  {name}: {value:.4f}")

    print(f"\n{'=' * 70}")
    print("Done!")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
