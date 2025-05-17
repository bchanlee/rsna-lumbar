import pandas as pd

pred_gen_df = pd.read_csv("data/generated_crops_preds.csv")
pred_gt_df = pd.read_csv("data/gt_crops_preds.csv")
target_df = pd.read_csv("data/train_narrow.csv")

df = (
    pd.merge(pred_gen_df, pred_gt_df, on=["study_id", "level"], how="inner")
      .merge(target_df, on=["study_id", "level"], how="inner")
)

for mode in ["gt", "gen"]:
    print(f"Mode: {mode}")
    # True Positives per class (both pred and true = 1)
    true_normal = ((df[f"normal_mild_pred_{mode}"] == 1) & (df["normal_mild"] == 1)).sum()
    true_moderate = ((df[f"moderate_pred_{mode}"] == 1) & (df["moderate"] == 1)).sum()
    true_severe = ((df[f"severe_pred_{mode}"] == 1) & (df["severe"] == 1)).sum()

    # Total samples per class (ground truth = 1)
    total_normal = df["normal_mild"].sum()
    total_moderate = df["moderate"].sum()
    total_severe = df["severe"].sum()

    print(f"True Normal/Mild: {true_normal} / {total_normal}")
    print(f"True Moderate: {true_moderate} / {total_moderate}")
    print(f"True Severe: {true_severe} / {total_severe}")
    # Overall true/total
    total_true = true_normal + true_moderate + true_severe
    total_samples = total_normal + total_moderate + total_severe
    print(f"True Total: {total_true} / {total_samples}")