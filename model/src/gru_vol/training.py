"""Seeded training loop with AdamW, early stopping on validation MSE, checkpointing, and wall-clock compute accounting."""

import copy
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


def set_seed(seed):
    """same seed -> same init weights + same batch order, call before build_model"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def predict(model, X, device, batch_size=1024):
    """ratio preds for array X (n, 60, 4), eval mode, no grad"""
    model.eval()
    preds = []
    with torch.no_grad():  # no gradient tracking, only predicting
        for i in range(0, len(X), batch_size):
            x = torch.as_tensor(X[i:i + batch_size], device=device)
            preds.append(model(x).cpu())
    return torch.cat(preds).numpy()


def val_mse(model, X_val, var60_val, target_val, device):
    """val mse in original units: ratio pred * var_60 vs realized 5 day var"""
    forecast = predict(model, X_val, device) * var60_val
    return float(np.mean((forecast - target_val) ** 2))


def train(model, data, cfg, seed, device, y_teacher=None, alpha=0.0, ckpt_path=None):
    """
    data: loaded splits.npz
    y_teacher: cached teacher ratio preds on train, only for distilled student
    alpha: distillation weight, 0 = plain mse on true ratio (teacher, direct student)
    """
    t = cfg["training"]
    model.to(device)

    # train batches, teacher preds ride along so rows stay lined up
    tensors = [torch.as_tensor(data["X_train"]), torch.as_tensor(data["y_train"])]
    if y_teacher is not None:
        tensors.append(torch.as_tensor(y_teacher, dtype=torch.float32))
    gen = torch.Generator().manual_seed(seed)  # fixed batch order per seed
    loader = DataLoader(TensorDataset(*tensors), batch_size=t["batch_size"], shuffle=True, generator=gen)

    opt = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    mse = torch.nn.MSELoss()

    best, best_state, bad_epochs, history = float("inf"), None, 0, []
    start = time.time()
    for epoch in tqdm(range(1, t["max_epochs"] + 1), desc="epochs"):  # progress bar
        model.train()
        total = 0.0
        for batch in loader:
            x, y = batch[0].to(device), batch[1].to(device)
            pred = model(x)
            loss = mse(pred, y)
            if alpha > 0:
                # eq 3: blend true ratio w/ teacher ratio
                loss = (1 - alpha) * loss + alpha * mse(pred, batch[2].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(x)

        v = val_mse(model, data["X_validation"], data["var60_validation"], data["target_validation"], device)
        history.append({"epoch": epoch, "train_loss": total / len(loader.dataset), "val_mse": v})

        # early stopping on val mse in original units, keep best weights only
        if v < best:
            best, best_state, bad_epochs = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= t["early_stopping_patience"]:
                break

    model.load_state_dict(best_state)
    if ckpt_path:
        torch.save(best_state, ckpt_path)
    best_epoch = min(history, key=lambda h: h["val_mse"])["epoch"]
    return {"best_val_mse": best, "best_epoch": best_epoch, "epochs_run": epoch,
            "train_seconds": time.time() - start, "history": history}