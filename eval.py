import argparse
import torch
import pytorch_lightning as pl
import numpy as np

from importlib import import_module
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Config file (without .py)")
    parser.add_argument("task", type=str, help="cls, reg")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--print_preds", action="store_true", help="Prints predictions")
    return parser.parse_args()

def eval_cls(preds, labels, print_preds):
    one_hot = np.zeros_like(preds)
    one_hot[np.arange(preds.shape[0]), np.argmax(preds, axis=1)] = 1

    true_count = 0
    false_count = 0
    true_normal_mild = 0
    true_mod = 0
    true_severe = 0
    for i in range(len(preds)):
        if print_preds:
            print(one_hot[i], labels[i])
        if (one_hot[i] == labels[i]).all():
            true_count += 1
            if labels[i][0] == 1:
                true_normal_mild += 1
            elif labels[i][1] == 1:
                true_mod += 1
            elif labels[i][2] == 1:
                true_severe += 1
            else:
                print("error: no class found")
        else:
            false_count += 1
    
    class_indices = labels.argmax(axis=1)  # get class index for each sample
    counts = np.bincount(class_indices, minlength=3)

    print(f"True Normal / Mild: {true_normal_mild}/{counts[0]}")
    print(f"True Moderate: {true_mod}/{counts[1]}")
    print(f"True Severe: {true_severe}/{counts[2]}")
    print(f"True Total: {true_count}/{true_count + false_count}")

def eval_reg(preds, labels, print_preds):
    for i in range(len(preds)):
        if print_preds:
            print(preds[i], labels[i])
    print(f"Total: {len(preds)}")

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

    with torch.inference_mode():
        for batch in tqdm(val_loader):
            logits = model(batch)
            if args.task == "cls":
                prob = torch.softmax(logits['logits'], dim=1)
                y = batch['y']
            elif args.task == "reg":
                # TODO: scale by some number like original image sz
                prob = torch.sigmoid(logits['logits'])
                y = batch['y']
            else:
                raise ValueError(f"Unknown task: {args.task}")
            all_preds.append(prob.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)

    if args.task == "cls":
        eval_cls(preds, labels, args.print_preds)

    elif args.task == "reg":
        eval_reg(preds, labels, args.print_preds)
    else:
        raise ValueError(f"Unknown task: {args.task}")
    

    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
