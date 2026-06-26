import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from datasets.unitarity import choi_su2
from model import PFNRegressor, RiemannDistribution, eval_step, train_step
from utility import Random_Kraus as kraus


def make_args():
    defaults = SimpleNamespace(
        run_name="unitarity3d",
        output_dir="jobs/unitarity3d",
        emb_size=512,
        n_heads=8,
        ff_hidden=1024,
        n_layers=16,
        dropout=0.0,
        decoder_hidden=1024,
        train_datasets=5_000_000,
        validation_datasets=1_000_000,
        max_num_points=80,
        max_context_points=None,
        batch_size=100,
        validation_batch_size=100,
        train_seed=777,
        validation_seed=2026,
        num_buckets=1000,
        lr=3e-5,
        min_lr=1e-6,
        weight_decay=0.0,
        grad_clip=1.0,
        log_every=1000,
        checkpoint_every=10_000,
        max_steps=None,
        d_env=2,
        r_dec=2,
    )
    parser = argparse.ArgumentParser(description="Train 3D unitarity PFN with uniform training and Haar evaluation.")
    parser.add_argument("--run-name", default=defaults.run_name)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Path to a checkpoint.pt with model/optimizer state to continue training.",
    )
    parser.add_argument("--emb-size", type=int, default=defaults.emb_size)
    parser.add_argument("--n-heads", type=int, default=defaults.n_heads)
    parser.add_argument("--ff-hidden", type=int, default=defaults.ff_hidden)
    parser.add_argument("--n-layers", type=int, default=defaults.n_layers)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--decoder-hidden", type=int, default=defaults.decoder_hidden)
    parser.add_argument("--train-datasets", type=int, default=defaults.train_datasets)
    parser.add_argument("--validation-datasets", type=int, default=defaults.validation_datasets)
    parser.add_argument("--max-num-points", type=int, default=defaults.max_num_points)
    parser.add_argument("--max-context-points", type=int, default=defaults.max_context_points)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--validation-batch-size", type=int, default=defaults.validation_batch_size)
    parser.add_argument("--train-seed", type=int, default=defaults.train_seed)
    parser.add_argument("--validation-seed", type=int, default=defaults.validation_seed)
    parser.add_argument("--num-buckets", type=int, default=defaults.num_buckets)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--min-lr", type=float, default=defaults.min_lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--grad-clip", type=float, default=defaults.grad_clip)
    parser.add_argument("--log-every", type=int, default=defaults.log_every)
    parser.add_argument("--checkpoint-every", type=int, default=defaults.checkpoint_every)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--d-env", type=int, default=defaults.d_env)
    parser.add_argument("--r-dec", type=int, default=defaults.r_dec)
    return parser.parse_args()


def make_scheduler(optimizer, total_steps, base_lr, min_lr, warmup_fraction=0.1):
    if min_lr < 0:
        raise ValueError("min_lr must be >= 0.")
    if min_lr > base_lr:
        raise ValueError("min_lr must be <= base_lr.")

    warmup_steps = max(1, int(total_steps * warmup_fraction))
    min_lr_fraction = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_resume_checkpoint(model, optimizer, resume_path, device):
    checkpoint = torch.load(resume_path, map_location=device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Resume checkpoint does not match model. missing={missing}, unexpected={unexpected}"
        )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
    return checkpoint


def set_scheduler_to_step(scheduler, step):
    if step <= 0:
        return
    # Checkpoints from this script do not include scheduler state, so restore
    # the deterministic LambdaLR position from the global training step.
    scheduler.step(step)


def normalize_angles(alpha, beta, gamma):
    return torch.stack(
        [alpha / (2 * math.pi), beta / math.pi, gamma / (2 * math.pi)],
        dim=-1,
    )


def uniform_angles(num_datasets, num_points, device):
    u = torch.rand(num_datasets * num_points, 3, device=device, dtype=torch.float64)
    u = u.reshape(num_datasets, num_points, 3)

    alpha = u[..., 0] * 2 * math.pi
    beta = u[..., 1] * math.pi
    gamma = u[..., 2] * 2 * math.pi
    x = normalize_angles(alpha, beta, gamma)
    return x, alpha, beta, gamma


def haar_angles(num_datasets, num_points, device):
    shape = (num_datasets, num_points)
    alpha = torch.rand(shape, device=device, dtype=torch.float64) * 2 * math.pi
    beta = torch.acos(1 - 2 * torch.rand(shape, device=device, dtype=torch.float64))
    gamma = torch.rand(shape, device=device, dtype=torch.float64) * 2 * math.pi
    x = normalize_angles(alpha, beta, gamma)
    return x, alpha, beta, gamma


def unitarity_from_angles(alpha, beta, gamma, batch_size, d_env, r_dec, device):
    y_batches = []
    num_datasets = alpha.shape[0]

    for start in range(0, num_datasets, batch_size):
        end = min(start + batch_size, num_datasets)
        batch_n = end - start

        j_in = choi_su2(alpha[start:end], beta[start:end], gamma[start:end]).to(device)
        combs = kraus.generate_one_slot_kraus_comb(
            num_samples=batch_n,
            sys_dim=(2, 2, 2, 2),
            env_dim=d_env,
            r_enc=1,
            r_dec=r_dec,
        ).to(device)

        j_out = torch.einsum("maoi,mnij,mapj->mnop", combs, j_in, combs.conj())
        y = (j_out @ j_out).diagonal(dim1=-2, dim2=-1).sum(-1).real / 4
        y_batches.append(y.unsqueeze(-1))

        del combs, j_in, j_out

    return torch.cat(y_batches, dim=0)


def sample_uniform_batch(args, num_datasets, device):
    x, alpha, beta, gamma = uniform_angles(num_datasets, args.max_num_points, device)
    y = unitarity_from_angles(alpha, beta, gamma, num_datasets, args.d_env, args.r_dec, device)
    return x, y


def sample_haar_batch(args, num_datasets, device):
    x, alpha, beta, gamma = haar_angles(num_datasets, args.max_num_points, device)
    y = unitarity_from_angles(alpha, beta, gamma, num_datasets, args.d_env, args.r_dec, device)
    return x, y


def build_model(args, device):
    return PFNRegressor(
        x_dim=3,
        y_dim=1,
        emb_size=args.emb_size,
        n_heads=args.n_heads,
        ff_hidden=args.ff_hidden,
        n_layers=args.n_layers,
        num_buckets=args.num_buckets,
        dropout=args.dropout,
        use_x_norm=False,
        decoder_hidden=args.decoder_hidden,
    ).to(device)


def transformer_config_from_args(args):
    return {
        "run_name": args.run_name,
        "emb_size": args.emb_size,
        "n_heads": args.n_heads,
        "ff_hidden": args.ff_hidden,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "decoder_hidden": args.decoder_hidden,
    }


def hyperparameters_from_args(args):
    return {
        "train_datasets": args.train_datasets,
        "validation_datasets": args.validation_datasets,
        "max_num_points": args.max_num_points,
        "max_context_points": args.max_context_points,
        "batch_size": args.batch_size,
        "validation_batch_size": args.validation_batch_size,
        "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "num_buckets": args.num_buckets,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "log_every": args.log_every,
        "checkpoint_every": args.checkpoint_every,
        "max_steps": args.max_steps,
        "d_env": args.d_env,
        "r_dec": args.r_dec,
        "training_sampler": "iid_uniform_normalized_cube",
        "evaluation_sampler": "haar_su2",
    }


def make_configs(args, total_params, trainable_params):
    transformer_config = transformer_config_from_args(args)
    hyperparameters = hyperparameters_from_args(args)
    parameter_summary = {
        "transformer": transformer_config,
        "hyperparameters": hyperparameters,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
    }
    return transformer_config, hyperparameters, parameter_summary


def train_uniform(
    model,
    dist,
    optimizer,
    scheduler,
    args,
    paths,
    configs,
    device,
    start_step=0,
    start_seen=0,
    resume_path=None,
):
    checkpoint_path, log_path = paths
    transformer_config, hyperparameters, parameter_summary = configs

    total_steps = math.ceil(args.train_datasets / args.batch_size)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)

    seed = args.train_seed if start_step == 0 else args.train_seed + start_step
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    seen_train = start_seen
    running_loss = 0.0
    running_steps = 0
    start_time = time.time()

    with log_path.open("w") as log:
        log.write("uniform_train_haar_eval\n")
        log.write(f"run_dir: {log_path.parent}\n")
        log.write(json.dumps(parameter_summary, indent=2) + "\n")
        if resume_path is not None:
            log.write("\n[resume]\n")
            log.write(f"path: {resume_path}\n")
            log.write(f"start_step: {start_step}\n")
            log.write(f"start_seen_train_datasets: {start_seen}\n")
            log.write(f"rng_seed: {seed}\n")
        log.write("\n[train_uniform]\n")
        log.write("step,loss,lr,minutes\n")

        for step in range(start_step + 1, total_steps + 1):
            current_batch_size = min(args.batch_size, args.train_datasets - seen_train)
            x_batch, y_batch = sample_uniform_batch(args, current_batch_size, device)

            loss = train_step(
                model=model,
                optimizer=optimizer,
                dist=dist,
                x_all=x_batch,
                y_all=y_batch,
                max_context=args.max_context_points,
                max_grad_norm=args.grad_clip,
            )
            scheduler.step()

            seen_train += current_batch_size
            running_loss += loss
            running_steps += 1

            if step % args.log_every == 0 or step == total_steps:
                avg_loss = running_loss / running_steps
                lr = optimizer.param_groups[0]["lr"]
                minutes = (time.time() - start_time) / 60
                line = f"{step},{avg_loss:.6f},{lr:.3e},{minutes:.1f}"
                print(line, flush=True)
                log.write(line + "\n")
                log.flush()
                running_loss = 0.0
                running_steps = 0

            if step % args.checkpoint_every == 0:
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step,
                        "seen_train_datasets": seen_train,
                        "transformer": transformer_config,
                        "hyperparameters": hyperparameters,
                    },
                    checkpoint_path,
                )

            del x_batch, y_batch

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": total_steps,
            "seen_train_datasets": seen_train,
            "transformer": transformer_config,
            "hyperparameters": hyperparameters,
        },
        checkpoint_path,
    )
    return {
        "steps": total_steps,
        "start_step": start_step,
        "continued_steps": total_steps - start_step,
        "seen_train_datasets": seen_train,
        "start_seen_train_datasets": start_seen,
    }


def evaluate_haar(model, dist, args, device):
    model.eval()
    torch.manual_seed(args.validation_seed)
    torch.cuda.manual_seed_all(args.validation_seed)

    eval_steps = math.ceil(args.validation_datasets / args.validation_batch_size)
    seen_eval = 0
    total_loss = 0.0
    total_mae = 0.0
    start_time = time.time()

    with torch.no_grad():
        for _ in range(eval_steps):
            current_batch_size = min(args.validation_batch_size, args.validation_datasets - seen_eval)
            x_eval, y_eval = sample_haar_batch(args, current_batch_size, device)
            out = eval_step(model, dist, x_eval, y_eval, max_context=args.max_context_points)

            total_loss += out["loss"] * current_batch_size
            total_mae += out["mae"] * current_batch_size
            seen_eval += current_batch_size

            del x_eval, y_eval, out

    return {
        "validation_datasets": seen_eval,
        "validation_loss": total_loss / seen_eval,
        "validation_mae": total_mae / seen_eval,
        "minutes": (time.time() - start_time) / 60,
        "sampler": "haar_su2",
    }


def main():
    args = make_args()
    torch.set_default_dtype(torch.float64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This unitarity generator is set up for CUDA.")

    print(device, torch.cuda.get_device_name(0), flush=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = run_dir / "checkpoint.pt"
    weights_path = run_dir / "model_weights.pt"
    metrics_path = run_dir / "metrics.json"
    params_path = run_dir / "model_parameters.json"
    log_path = run_dir / "run.log"

    print(f"run_dir: {run_dir}", flush=True)
    print("[configuration]", flush=True)
    print(json.dumps(transformer_config_from_args(args), indent=2), flush=True)
    print("[hyperparameters]", flush=True)
    print(json.dumps(hyperparameters_from_args(args), indent=2), flush=True)

    torch.manual_seed(args.train_seed)
    torch.cuda.manual_seed_all(args.train_seed)

    model = build_model(args, device)
    boundaries = torch.linspace(
        0.25,
        1.0,
        args.num_buckets + 1,
        device=device,
        dtype=torch.float64,
    )
    dist = RiemannDistribution(boundaries, use_tails=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"total parameters: {total_params:,}", flush=True)
    print(f"trainable parameters: {trainable_params:,}", flush=True)

    transformer_config, hyperparameters, parameter_summary = make_configs(
        args,
        total_params,
        trainable_params,
    )
    params_path.write_text(json.dumps(parameter_summary, indent=2))

    total_steps = math.ceil(args.train_datasets / args.batch_size)
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    scheduler = make_scheduler(optimizer, total_steps, args.lr, args.min_lr)
    resume_checkpoint = None
    resume_step = 0
    resume_seen = 0
    if args.resume_path is not None:
        resume_checkpoint = load_resume_checkpoint(model, optimizer, args.resume_path, device)
        resume_step = int(resume_checkpoint.get("step", 0))
        resume_seen = int(resume_checkpoint.get("seen_train_datasets", resume_step * args.batch_size))
        set_scheduler_to_step(scheduler, resume_step)
        print(f"[resume] path: {args.resume_path}", flush=True)
        print(f"[resume] start_step: {resume_step}", flush=True)
        print(f"[resume] start_seen_train_datasets: {resume_seen}", flush=True)

    train_result = train_uniform(
        model=model,
        dist=dist,
        optimizer=optimizer,
        scheduler=scheduler,
        args=args,
        paths=(checkpoint_path, log_path),
        configs=(transformer_config, hyperparameters, parameter_summary),
        device=device,
        start_step=resume_step,
        start_seen=resume_seen,
        resume_path=args.resume_path,
    )
    validation_result = evaluate_haar(model, dist, args, device)
    print(validation_result)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "transformer": transformer_config,
            "hyperparameters": hyperparameters,
            "parameter_summary": parameter_summary,
        },
        weights_path,
    )

    metrics = {
        "transformer": transformer_config,
        "hyperparameters": hyperparameters,
        "train": train_result,
        "validation": validation_result,
        "data_mode": "uniform_train_haar_eval",
        "checkpoint": str(checkpoint_path),
        "model_weights": str(weights_path),
        "model_parameters": str(params_path),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
