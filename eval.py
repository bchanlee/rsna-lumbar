import argparse
import torch
import pytorch_lightning as pl
import numpy as np

from importlib import import_module
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file (without .py)")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--print_preds", action="store_true", help="Prints predictions")
    return parser.parse_args()

def main():
    args = parse_args()

    cfg = import_module(f"configs.{args.config}").cfg


    print("\nRunning Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Fold: {cfg.fold}")


    wts = torch.load(args.checkpoint)["state_dict"]
    wts = {k.replace("model.", ""): v for k, v in wts.items()}
    model = import_module(f"models.{cfg.model}").Net(cfg)
    model.load_state_dict(wts)
    model.eval()

    ds_class = import_module(f"datasets.{cfg.dataset}").Dataset
    val_dataset = ds_class(cfg, "val")
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.val_batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            logits = model(batch)
            prob = torch.sigmoid(logits['logits'])
            all_preds.append(prob.cpu().numpy())
            all_labels.append(batch['y'].cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(preds.shape[0]), np.argmax(preds, axis=1)] = 1

    true_count = 0
    false_count = 0
    for i in range(len(preds)):
        if args.print_preds:
            print(one_hot[i], labels[i])
        if (one_hot[i] == labels[i]).all():
            true_count += 1
        else:
            false_count += 1
    
    class_indices = labels.argmax(axis=1)  # get class index for each sample
    counts = np.bincount(class_indices, minlength=3)

    print("Normal / Mild:", counts[0])
    print("Moderate:", counts[1])
    print("Severe:", counts[2])
    print("True:", true_count, "False:", false_count) # actual metric is sample weighted log loss

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
