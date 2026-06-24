import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm
# from dataset.medical_few import MedDataset
from dataset.video_csv import VideoCSVDataset
from CLIP.clip import create_model
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, confusion_matrix, matthews_corrcoef, recall_score, precision_score
from loss import  Loss_detection
from Prompt.promptChooser import PromptChooser

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings

warnings.filterwarnings("ignore")
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Global log function (will be set in main)
_global_log_line = None

CLASS_INDEX = {'Brain': 3, 'Liver': 2, 'Retina_RESC': 1, 'Retina_OCT2017': -1, 'Chest': -2, 'Histopathology': -3, 'Bleeding': -4}

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='DBT-Bleed training')
    parser.add_argument('--exp_name', type=str, help='name it')

    parser.add_argument('--model_name', type=str, default='ViT-L-14-336', help='ViT-B-16-plus-240, ViT-L-14-336')
    parser.add_argument('--pretrain', type=str, default='openai', help='laion400m, openai')

    parser.add_argument('--obj', type=str, default='Bleeding', help='target dataset, it should be in CLASS_INDEX')
    parser.add_argument('--data_path', type=str, default='./data/')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Base output dir for logs/models/predictions/results')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_model', action="store_true", default=True)
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Explicit checkpoint path (video mode). If empty, uses best_auc.pth in models dir.')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--csv_dir', type=str, default='', help='Directory containing train/val/test CSVs')
    parser.add_argument('--test_csv', type=str, default='', help='Path to test CSV')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--sample_mode', type=str, default='entropy_segment',
                        help="Frame sampling: 'uniform', 'uniform_jitter', "
                             "'entropy_segment', 'entropy_segment_jitter'")
    parser.add_argument('--entropy_method', type=str, default='entropy',
                        choices=['entropy'],
                        help='Scoring method for entropy_segment modes: entropy (Red channel)')
    parser.add_argument('--segment_size', type=int, default=4,
                        help='Segment size for hierarchical elimination in entropy_segment modes')
    parser.add_argument("--epoch", type=int, default=60, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24],
                        help="Layers to apply adapter on")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=16, help="number of samples for training")
    parser.add_argument('--iterate', type=int, default=0)

    parser.add_argument('--text_mood', type=str, default='learnable_all',
                        help=" the prompt strategy: fix, learnable_all, learnable_abnormal")
    parser.add_argument('--contrast_mood', type=str, default='yes', help="no, yes")
    # parser.add_argument('--adapter_res_mood', type=float, default=0.0, help="whether to apply residual after adapters, 0.0(for no residual), 0.2,0.4,0.8... or any value < 1")

    parser.add_argument('--dec_type', type=str, default='mean', help="aggregation of adapters outputs: mean, max, both,"
                                                                     " they account for global or localized decisions")
    parser.add_argument('--video_agg', type=str, default='temporal', choices=['temporal', 'vote'],
                        help="Video-level aggregation: 'temporal' (default, uses temporal adapter), "
                             "'vote' (majority voting baseline: score each frame independently, "
                             "fraction of frames > 0.5 = video score)")
    parser.add_argument('--loss_type', type=str, default='sigmoid', help="Loss type: softmax, sigmoid")
    parser.add_argument('--threshold', type=float, default=None,
                        help='Fixed threshold for binary classification. If not set, optimal threshold is computed from PR curve.')

    parser.add_argument('--prompt_style', type=str, default='default', choices=['default', 'surgical'],
                        help="Prompt template style: 'default' (generic MadCLIP), 'surgical' (domain-specific bleeding prompts)")
    parser.add_argument('--visionA', type=str, default="MFCFC", help="Type of adapter layers (MCNNFC, MFCFC, MFCFC_v2, LKA, MViTFC)")
    parser.add_argument('--adapter_hidden', type=int, default=256, help="Hidden dim for MFCFC_v2 bottleneck")
    parser.add_argument('--adapter_dropout', type=float, default=0.1, help="Dropout rate for MFCFC_v2 / LKA")
    parser.add_argument('--lka_bottleneck', type=int, default=8, help="LKA bottleneck width d_hat (default 8)")
    parser.add_argument('--lka_kernel_size', type=int, default=7, help="LKA depthwise conv kernel size (default 7)")
    parser.add_argument('--temporal_per_layer', action="store_true", default=True,
                        help="One temporal adapter per adapter-layer per branch (default: enabled)")
    parser.add_argument('--shared_temporal_adapter', action="store_false", dest='temporal_per_layer',
                        help="Use one shared temporal adapter per branch instead of per-layer")
    parser.add_argument('--temporal_layers', type=int, default=6,
                        help="Number of transformer layers in temporal adapter (default: 2, ActionCLIP uses 6)")
    parser.add_argument('--temporal_pool_mode', type=str, default='max', choices=['mean', 'max'],
                        help="Pooling mode for Transf temporal adapter: 'mean' (default) or 'max'")
    parser.add_argument('--temporal_adapter_type', type=str, default='transf',
                        choices=['transf', 'max', 'mean', 'pos_max', 'pos_mean'],
                        help="Temporal adapter type: "
                             "'transf' (ActionCLIP Transformer, default), "
                             "'max' (baseline: simple max pooling, 0 params), "
                             "'mean' (baseline: simple mean pooling, 0 params), "
                             "'pos_max' (baseline: positional embeddings + max pooling), "
                             "'pos_mean' (baseline: positional embeddings + mean pooling)")
    parser.add_argument('--pit_surgical', action='store_true',
                        help='Enable MCC (Matthews Correlation Coefficient) metric for Pit Surgical dataset evaluation')
    parser.add_argument('--no_sev_metrics', action='store_true',
                        help='Disable severity-weighted metric calculation')


    args = parser.parse_args()
    setup_seed(args.seed)
    from CLIP.adapter_shared import CLIP_Inplanted

    run_root = Path(args.output_dir) / args.exp_name
    models_dir = run_root / "models"
    predictions_dir = run_root / "predictions"
    results_dir = run_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = results_dir / f"test_{timestamp}.log"
    # log_path = results_dir / f"val_{timestamp}.log"

    def log_line(msg: str):
        timestamped_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(timestamped_msg)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(timestamped_msg + "\n")

    # Make log_line globally accessible for error handling
    global _global_log_line
    _global_log_line = log_line

    # Log configuration details
    log_line("=" * 80)
    log_line("TESTING/INFERENCE CONFIGURATION")
    log_line("=" * 80)
    log_line(f"Experiment Name: {args.exp_name}")
    log_line(f"Output Directory: {args.output_dir}")
    log_line(f"Log File: {log_path}")
    log_line(f"Checkpoint: {args.checkpoint if args.checkpoint else 'auto (best_auc.pth)'}")
    log_line("")
    log_line("--- Model Configuration ---")
    log_line(f"Model Name: {args.model_name}")
    log_line(f"Pretrain: {args.pretrain}")
    log_line(f"Image Size: {args.img_size}")
    log_line(f"Vision Adapter Type: {args.visionA}")
    log_line(f"Features List (Layers): {args.features_list}")
    log_line("")
    log_line("--- Video Configuration ---")
    if hasattr(args, 'csv_dir') and args.csv_dir:
        log_line(f"CSV Directory: {args.csv_dir}")
        log_line(f"Number of Frames: {args.num_frames}")
        log_line(f"Sample Mode: {args.sample_mode}")
        log_line(f"Temporal Per Layer: {getattr(args, 'temporal_per_layer', False)}")
        log_line(f"Temporal Transformer Layers: {getattr(args, 'temporal_layers', 2)}")
    log_line("")
    log_line("--- Inference Configuration ---")
    log_line(f"Batch Size: {args.batch_size}")
    log_line(f"Seed: {args.seed}")
    log_line("")
    log_line("--- Prompt & Loss Configuration ---")
    log_line(f"Object/Domain: {args.obj}")
    log_line(f"Text Mood: {args.text_mood}")
    log_line(f"Prompt Style: {args.prompt_style}")
    log_line(f"Decision Type: {args.dec_type}")
    log_line(f"Loss Type: {args.loss_type}")
    log_line(f"Fixed Threshold: {args.threshold if args.threshold is not None else 'auto (optimal from PR curve)'}")
    log_line("")
    log_line("--- System Configuration ---")
    log_line(f"Device: {device}")
    log_line(f"CUDA Available: {torch.cuda.is_available()}")
    log_line(f"GPU Count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log_line(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    log_line("=" * 80)
    log_line("")

    ############################
    # Instantiating CLIP model #
    ############################
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                             pretrained=args.pretrain, require_pretrained=True)
    clip_model.to(device)
    clip_model.eval()

    # add adapters on CLIP
    model = CLIP_Inplanted(args, clip_model=clip_model).to(device)
    model.eval()

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        log_line(f"Using {torch.cuda.device_count()} GPUs for inference")
        model = torch.nn.DataParallel(model)
    else:
        log_line("Using single GPU for inference")

    # Helper to access underlying model (handles DataParallel)
    def get_model(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    def _resolve_csv(csv_dir, csv_path, default_name):
        if csv_path:
            return csv_path
        if csv_dir:
            return os.path.join(csv_dir, default_name)
        return ""

    want_video = bool(args.csv_dir or args.test_csv)
    test_csv = _resolve_csv(args.csv_dir, args.test_csv, "test.csv")
    # test_csv = _resolve_csv(args.csv_dir, args.test_csv, "val.csv")
    use_video = False
    if want_video:
        if not test_csv or not os.path.exists(test_csv):
            raise FileNotFoundError("Test CSV not found for video mode")
        use_video = True

    # load test dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if use_video:
        test_dataset = VideoCSVDataset(
            csv_path=test_csv,
            num_frames=args.num_frames,
            resize=args.img_size,
            sample_mode=args.sample_mode,
            is_train=False,
            return_path=True,
            entropy_method=args.entropy_method,
            segment_size=args.segment_size,
        )
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)


    loss_det = Loss_detection(args= args, device=device, loss_type=args.loss_type,dec_type=args.dec_type, lr=args.learning_rate)



    # text maker
    text_chooser = PromptChooser(clip_model, args, device)


    # loading the saved checkpoint
    if use_video:
        if args.checkpoint:
            ckp_path = args.checkpoint
        else:
            ckp_path = models_dir / "best_auc.pth"
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')

    checkpoint = torch.load(ckp_path)
    log_line(f"Loaded checkpoint: {ckp_path}")
    model_unwrapped = get_model(model)
    model_unwrapped.normal_det_adapters.load_state_dict(checkpoint["normal_det_adapters"])
    model_unwrapped.abnormal_det_adapters.load_state_dict(checkpoint["abnormal_det_adapters"])
    if "temporal_normal" in checkpoint and "temporal_abnormal" in checkpoint:
        model_unwrapped.temporal_normal.load_state_dict(checkpoint["temporal_normal"])
        model_unwrapped.temporal_abnormal.load_state_dict(checkpoint["temporal_abnormal"])
    else:
        log_line("Warning: temporal adapter weights not found in checkpoint.")

    # Load text chooser parameters based on text_mood
    if text_chooser.text_mood == 'fix':
        if 'text_features_fix' in checkpoint:
            text_chooser.text_features_fix = checkpoint['text_features_fix']
        else:
            raise KeyError("Missing 'text_features_fix' in checkpoint for text_mood='fix'")
    elif text_chooser.text_mood == 'learnable_all':
        if 'prompt_maker_normal' in checkpoint and 'prompt_maker_abnormal' in checkpoint:
            text_chooser.prompt_maker_normal.load_state_dict(checkpoint['prompt_maker_normal'])
            text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint['prompt_maker_abnormal'])
        else:
            raise KeyError(
                "Missing 'prompt_maker_normal' or 'prompt_maker_abnormal' in checkpoint for text_mood='learnable_all'")

    else:  # only learnable abnormal
        if 'prompt_maker_abnormal' in checkpoint:
            text_chooser.prompt_maker_abnormal.load_state_dict(checkpoint['prompt_maker_abnormal'])
        else:
            raise KeyError("Missing 'prompt_maker_abnormal' in checkpoint for text_mood='only learnable abnormal'")



    if use_video:
        metrics, sim_text, predictions = test(args, model, test_loader, text_chooser, loss_det, use_video=True, fixed_threshold=args.threshold)
        log_line("=" * 80)
        log_line("TEST RESULTS")
        log_line("=" * 80)
        log_line(f"AUC (ROC): {metrics['auc']:.4f}")
        thr_label = "Fixed Threshold" if args.threshold is not None else "Optimal Threshold"
        log_line(f"{thr_label}: {metrics['best_thr']:.4f}")
        log_line("")
        log_line("--- F1 Scores (at optimal threshold) ---")
        log_line(f"F1 Macro:               {metrics['f1_macro']:.4f}")
        log_line(f"F1 Weighted:            {metrics['f1_weighted']:.4f}")
        log_line(f"F1 Normal (class 0):    {metrics['f1_normal']:.4f}")
        log_line(f"F1 Bleeding (class 1):  {metrics['f1_bleeding']:.4f}")
        log_line(f"Best F1 (bleeding):     {metrics['best_f1']:.4f}")
        log_line("")
        log_line("--- Recall ---")
        log_line(f"Recall Macro:           {metrics['recall_macro']:.4f}")
        log_line(f"Recall Weighted:        {metrics['recall_weighted']:.4f}")
        log_line(f"Recall Normal (class 0):    {metrics['recall_normal']:.4f}")
        log_line(f"Recall Bleeding (class 1):  {metrics['recall_bleeding']:.4f}")
        log_line("")
        log_line("--- Precision ---")
        log_line(f"Precision Normal (class 0):    {metrics['precision_normal']:.4f}")
        log_line(f"Precision Bleeding (class 1):  {metrics['precision_bleeding']:.4f}")
        log_line("")
        log_line("--- Average Precision (AP) ---")
        log_line(f"AP Macro:               {metrics['ap_macro']:.4f}")
        log_line(f"AP Weighted:            {metrics['ap_weighted']:.4f}")
        log_line(f"AP Normal (class 0):    {metrics['ap_normal']:.4f}")
        log_line(f"AP Bleeding (class 1):  {metrics['ap_bleeding']:.4f}")
        log_line("")
        log_line("--- Confusion Matrix ---")
        cm = metrics['confusion_matrix']
        log_line(f"              Predicted")
        log_line(f"           Normal  Bleeding")
        log_line(f"Actual Normal   {int(cm[0,0]):4d}    {int(cm[0,1]):4d}   (TN, FP)")
        log_line(f"       Bleeding {int(cm[1,0]):4d}    {int(cm[1,1]):4d}   (FN, TP)")
        log_line("")
        log_line(f"True Negatives (TN):  {int(cm[0,0])}")
        log_line(f"False Positives (FP): {int(cm[0,1])}")
        log_line(f"False Negatives (FN): {int(cm[1,0])}")
        log_line(f"True Positives (TP):  {int(cm[1,1])}")
        if args.pit_surgical:
            log_line("")
            log_line("--- Matthews Correlation Coefficient (MCC) ---")
            log_line(f"MCC:                  {metrics['mcc']:.4f}")
            log_line(f"Interpretation: +1=perfect, 0=random, <0=worse than random")

        # Severity-weighted metrics
        sev_metrics = None
        if not args.no_sev_metrics:
            sev_map = _load_sev_from_csv(test_csv)
            sev_metrics = _compute_sev_weighted_metrics(predictions, metrics["best_thr"], sev_map)
        if sev_metrics is not None:
            log_line("")
            log_line("--- Severity-Weighted Metrics (Bleeding) ---")
            log_line(f"Weights per sev level: {_SEV_WEIGHTS}")
            log_line("")
            log_line(f"{'Sev':>3s}  {'N':>5s}  {'F1':>7s}  {'AP':>7s}  {'Recall':>7s}")
            log_line(f"{'-'*3}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
            for sev_level in range(6):
                s = sev_metrics[sev_level]
                f1_str = f"{s['f1']:.4f}" if not np.isnan(s['f1']) else "   N/A"
                ap_str = f"{s['ap']:.4f}" if not np.isnan(s['ap']) else "   N/A"
                rc_str = f"{s['recall']:.4f}" if not np.isnan(s['recall']) else "   N/A"
                log_line(f"  {sev_level}  {s['n']:5d}  {f1_str:>7s}  {ap_str:>7s}  {rc_str:>7s}")
            log_line(f"{'-'*3}  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*7}")
            log_line(f"Sev-weighted F1:      {sev_metrics['weighted_f1']:.4f}")
            log_line(f"Sev-weighted AP:      {sev_metrics['weighted_ap']:.4f}")
            log_line(f"Sev-weighted Recall:  {sev_metrics['weighted_recall']:.4f}")

        log_line("=" * 80)
        pred_path = predictions_dir / f"predictions_{timestamp}.csv"
        with open(pred_path, "w", encoding="utf-8") as f:
            f.write("clip_path,gt,score\n")
            for row in predictions:
                f.write(f"{row['clip_path']},{row['gt']},{row['score']}\n")
        log_line(f"Saved predictions to {pred_path}")
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # result, auc, pauc, sim_text = test(args, model, test_loader, text_chooser, loss_det, use_video=use_video)


def _load_sev_from_csv(test_csv_path: str) -> dict:
    """Load severity scores from test CSV. Returns {clip_path_basename: int(sev)}."""
    sev_map = {}
    with open(test_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if "sev" not in (reader.fieldnames or []):
            return sev_map
        for row in reader:
            clip_name = os.path.basename(row["clip_path"])
            sev_map[clip_name] = int(row["sev"])
    return sev_map


_SEV_WEIGHTS = [0.02, 0.06, 0.12, 0.19, 0.26, 0.33]


def _compute_sev_weighted_metrics(predictions, best_thr, sev_map):
    """Compute per-severity and severity-weighted F1, AP, Recall for bleeding.

    Returns dict with per-sev values and weighted totals, or None if sev_map is empty.
    """
    if not sev_map:
        return None

    scores_arr = np.array([p["score"] for p in predictions], dtype=np.float32)
    labels_arr = np.array([p["gt"] for p in predictions], dtype=np.int32)
    sevs_arr = np.array(
        [sev_map.get(os.path.basename(p["clip_path"]), -1) for p in predictions],
        dtype=np.int32,
    )
    preds_arr = (scores_arr >= best_thr).astype(np.int32)

    result = {}
    weighted_f1 = 0.0
    weighted_ap = 0.0
    weighted_recall = 0.0

    for sev_level in range(6):
        mask = sevs_arr == sev_level
        n = int(mask.sum())
        w = _SEV_WEIGHTS[sev_level]

        if n == 0:
            result[sev_level] = {"n": 0, "f1": float("nan"), "ap": float("nan"), "recall": float("nan")}
            continue

        s_labels = labels_arr[mask]
        s_scores = scores_arr[mask]
        s_preds = preds_arr[mask]

        # F1 for bleeding (class 1) — labels=[0,1] ensures index 1 always exists
        f1_per = f1_score(s_labels, s_preds, labels=[0, 1], average=None, zero_division=0)
        sev_f1 = float(f1_per[1])

        # AP for bleeding (class 1)
        # Per-group AP is undefined when all samples share the same label
        # (sev=0 is all-normal, sev=1-5 is all-bleeding), so we skip per-group AP
        # and compute sev-weighted AP separately below.
        if s_labels.max() == s_labels.min():
            sev_ap = float("nan")
        else:
            sev_ap = float(average_precision_score(s_labels, s_scores))

        # Recall for bleeding (class 1) — labels=[0,1] ensures index 1 always exists
        recall_per = recall_score(s_labels, s_preds, labels=[0, 1], average=None, zero_division=0)
        sev_recall = float(recall_per[1])

        result[sev_level] = {"n": n, "f1": sev_f1, "ap": sev_ap, "recall": sev_recall}

        if not np.isnan(sev_f1):
            weighted_f1 += w * sev_f1
        if not np.isnan(sev_recall):
            weighted_recall += w * sev_recall

    # Severity-weighted AP: assign each sample a weight based on its sev level,
    # then compute a single weighted AP across all samples.
    sample_weights = np.array(
        [_SEV_WEIGHTS[s] if 0 <= s <= 5 else 0.0 for s in sevs_arr], dtype=np.float64,
    )
    if labels_arr.max() != labels_arr.min() and sample_weights.sum() > 0:
        weighted_ap = float(average_precision_score(labels_arr, scores_arr, sample_weight=sample_weights))
    else:
        weighted_ap = float("nan")

    result["weighted_f1"] = weighted_f1
    result["weighted_ap"] = weighted_ap
    result["weighted_recall"] = weighted_recall

    return result


def _compute_video_metrics(scores, labels, fixed_threshold=None):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.max() == labels.min():
        return {
            "auc": float("nan"),
            "ap_bleeding": float("nan"),
            "ap_normal": float("nan"),
            "ap_macro": float("nan"),
            "ap_weighted": float("nan"),
            "best_f1": float("nan"),
            "best_thr": 0.5,
            "f1_macro": float("nan"),
            "f1_weighted": float("nan"),
            "f1_normal": float("nan"),
            "f1_bleeding": float("nan"),
            "recall_macro": float("nan"),
            "recall_weighted": float("nan"),
            "recall_normal": float("nan"),
            "recall_bleeding": float("nan"),
            "precision_normal": float("nan"),
            "precision_bleeding": float("nan"),
            "confusion_matrix": np.zeros((2, 2)),
            "mcc": float("nan"),
        }

    auc = roc_auc_score(labels, scores)

    # AP for bleeding (positive class): standard AP
    ap_bleeding = average_precision_score(labels, scores)

    # AP for normal (negative class): invert labels and scores
    ap_normal = average_precision_score(1 - labels, 1 - scores)

    # AP macro: average of both classes
    ap_macro = (ap_bleeding + ap_normal) / 2.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-12)
    if thresholds.size > 0:
        best_idx = int(np.nanargmax(f1_scores[:-1]))
        best_f1 = float(f1_scores[best_idx])
        best_thr = float(thresholds[best_idx])
    else:
        best_f1 = float(f1_scores[0])
        best_thr = 0.5

    if fixed_threshold is not None:
        best_thr = fixed_threshold

    # Compute predictions at best threshold
    preds = (scores >= best_thr).astype(np.int32)

    # F1 macro and per-class F1
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0)
    f1_macro_val = f1_score(labels, preds, average='macro', zero_division=0)
    f1_weighted_val = f1_score(labels, preds, average='weighted', zero_division=0)
    f1_normal = float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0
    f1_bleeding = float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0

    # Recall per-class and macro/weighted
    recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
    recall_macro_val = recall_score(labels, preds, average='macro', zero_division=0)
    recall_weighted_val = recall_score(labels, preds, average='weighted', zero_division=0)
    recall_normal = float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0
    recall_bleeding = float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0

    # Precision per-class for completeness
    precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
    precision_normal = float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0
    precision_bleeding = float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0

    # Weighted AP - computed using sample weights based on class distribution
    # For binary: weighted AP approximated by weighted average of per-class APs
    class_counts = np.bincount(labels.astype(int))
    total = class_counts.sum()
    if total > 0 and len(class_counts) >= 2:
        weight_normal = class_counts[0] / total
        weight_bleeding = class_counts[1] / total
        ap_weighted = weight_normal * ap_normal + weight_bleeding * ap_bleeding
    else:
        ap_weighted = ap_macro

    # Confusion matrix [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    # MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(labels, preds)

    return {
        "auc": float(auc),
        "ap_bleeding": float(ap_bleeding),
        "ap_normal": float(ap_normal),
        "ap_macro": float(ap_macro),
        "ap_weighted": float(ap_weighted),
        "best_f1": best_f1,
        "best_thr": best_thr,
        "f1_macro": float(f1_macro_val),
        "f1_weighted": float(f1_weighted_val),
        "f1_normal": f1_normal,
        "f1_bleeding": f1_bleeding,
        "recall_macro": float(recall_macro_val),
        "recall_weighted": float(recall_weighted_val),
        "recall_normal": recall_normal,
        "recall_bleeding": recall_bleeding,
        "precision_normal": precision_normal,
        "precision_bleeding": precision_bleeding,
        "confusion_matrix": cm,
        "mcc": float(mcc),
    }


def test(args, model, test_loader, text_chooser, loss_det, use_video=False, fixed_threshold=None):
    image_list = []
    gt_list = []
    gt_mask_list = []
    det_final = []
    seg_final = []

    with torch.no_grad():
        text_features = text_chooser()  # [768,2]
        sim_text = F.cosine_similarity(text_features[:, 0], text_features[:, 1], dim=0)

    predictions = []
    if use_video:
        for (video, y, clip_path) in tqdm(test_loader):
            video = video.to(device)

            if args.video_agg == 'vote':
                # --- Majority Voting Baseline ---
                # Each frame is scored independently (no temporal adapter context).
                # video: [B, T, C, H, W] → reshape to [B*T, 1, C, H, W] so each
                # frame is a standalone single-frame "video".
                B_v, T_v, C_v, H_v, W_v = video.shape
                frames = video.reshape(B_v * T_v, 1, C_v, H_v, W_v)

                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, det_model_frames, _ = model(frames, text_features)
                    # Accumulate per-frame scores across adapter layers → [B*T]
                    frame_scores_raw = 0
                    for layer in range(len(det_model_frames)):
                        frame_scores_raw += loss_det.validation(
                            det_model_frames[layer].unsqueeze(1)  # [B*T, 1, 2]
                        )

                # Reshape to [B, T] — each value is a softmax bleeding probability
                frame_scores = frame_scores_raw.reshape(B_v, T_v)

                # OR voting: if ANY frame score > 0.5 → video is positive.
                # Continuous video score = max per-frame score (for AUC).
                vote_score = frame_scores.max(dim=1)[0]  # [B]

                det_final.extend(vote_score.cpu().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                for p, gt_val, score in zip(clip_path, y.cpu().detach().numpy(), vote_score.cpu().numpy()):
                    predictions.append({"clip_path": p, "gt": int(gt_val), "score": float(score)})

            else:
                # --- Existing temporal adapter path (unchanged) ---
                with torch.no_grad(), torch.cuda.amp.autocast():
                    _, det_model, _ = model(video, text_features)
                    anomaly_scores_all = 0
                    for layer in range(len(det_model)):
                        det_scores_cur = det_model[layer]  # [B, 2]
                        anomaly_scores_all += loss_det.validation(det_scores_cur.unsqueeze(1))

                det_final.extend(anomaly_scores_all.cpu().numpy())
                gt_list.extend(y.cpu().detach().numpy())
                for p, gt_val, score in zip(clip_path, y.cpu().detach().numpy(), anomaly_scores_all.cpu().numpy()):
                    predictions.append({"clip_path": p, "gt": int(gt_val), "score": float(score)})
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # for (image, y, mask) in tqdm(test_loader):
            # image = image.to(device)
            # mask[mask > 0.5], mask[mask <= 0.5] = 1, 0

            # with torch.no_grad(), torch.cuda.amp.autocast():
                # _, det_model, seg_model = model(image, text_features)

                # if CLASS_INDEX[args.obj] > 0:
                    # # seg head
                    # anomaly_maps = []
                    # for layer in range(len(seg_model)):
                        # seg_scores_cur = seg_model[layer] # batch, 2,240,240
                        # seg_scores_cur = loss_det.sync_AS(seg_scores_cur)
                        # anomaly_map = 0.5 * (1- seg_scores_cur[:,0,:,:]) + 0.5 * seg_scores_cur[:,1,:,:] # batch, 240,240
                        # anomaly_maps.append(anomaly_map.cpu().numpy())

                    # score_map = np.sum(np.stack(anomaly_maps), axis=0) # summing all anomaly_map of all layer [batch size, 240,240]
                    # seg_final.extend(score_map)
                # #else:
                # # det head
                # anomaly_scores_all = 0
                # for layer in range(len(det_model)):
                    # det_scores_cur = det_model[layer] # [batch, 289,2]
                    # anomaly_scores_all += loss_det.validation(det_scores_cur)

                # det_final.extend(anomaly_scores_all.cpu().numpy())

                # gt_mask_list.extend(mask.squeeze(1).cpu().detach().numpy())
                # gt_list.extend(y.cpu().detach().numpy())
                # image_list.extend(image.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    if use_video:
        det_scores = np.array(det_final)
        det_scores = (det_scores - det_scores.min()) / (
            1e-4 + det_scores.max() - det_scores.min()
        )
        metrics = _compute_video_metrics(det_scores, gt_list, fixed_threshold=fixed_threshold)
        print(f"Video AUC: {metrics['auc']:.4f} | "
              f"AP_macro: {metrics['ap_macro']:.4f} AP_bleeding: {metrics['ap_bleeding']:.4f} | "
              f"F1_macro: {metrics['f1_macro']:.4f} F1_bleeding: {metrics['f1_bleeding']:.4f} (best: {metrics['best_f1']:.4f})")
        return metrics, sim_text, predictions

    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # gt_mask_list = np.asarray(gt_mask_list)
    # gt_mask_list = (gt_mask_list > 0).astype(np.int_)
    # # image_list = np.asarray(image_list)

    # if CLASS_INDEX[args.obj] > 0:
        # seg_scores = np.array(seg_final)
        # seg_scores=  (seg_scores - seg_scores.min()) / ( 1e-4+
                    # seg_scores.max() - seg_scores.min())
        # # compute pixel level auc
        # seg_roc_auc = roc_auc_score(gt_mask_list.flatten(), seg_scores.flatten())
        # print(f'{args.obj} pAUC : {round(seg_roc_auc, 4)}')

        # # Reshape segment scores for image-level evaluation
        # segment_scores_flatten = seg_scores.reshape(seg_scores.shape[0], -1)
        # # Compute image-level ROC AUC
        # roc_auc_im = roc_auc_score(gt_list, np.max(segment_scores_flatten, axis=1))
        # print(f'{args.obj} AUC : {round(roc_auc_im, 4)}')
        # return seg_roc_auc + roc_auc_im, roc_auc_im, seg_roc_auc, sim_text

    # else:
        # det_image_scores_zero = np.array(det_final)
        # det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / ( 1e-4 + det_image_scores_zero.max() - det_image_scores_zero.min())
        # img_roc_auc_det = roc_auc_score(gt_list, det_image_scores_zero)
        # print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)} from det')
        # return img_roc_auc_det, img_roc_auc_det , img_roc_auc_det, sim_text

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TESTING INTERRUPTED BY USER (Ctrl+C)")
        print("=" * 80)
        if _global_log_line:
            _global_log_line("=" * 80)
            _global_log_line("TESTING INTERRUPTED BY USER (Ctrl+C)")
            _global_log_line("=" * 80)
        sys.exit(0)
    except Exception as e:
        error_msg = f"FATAL ERROR: {type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()

        # Print to console
        print(f"\n{'='*80}")
        print("TESTING FAILED WITH ERROR")
        print('='*80)
        print(error_msg)
        print("\nFull traceback:")
        print(error_trace)
        print('='*80)

        # Log to file if available
        if _global_log_line:
            _global_log_line("=" * 80)
            _global_log_line("TESTING FAILED WITH ERROR")
            _global_log_line("=" * 80)
            _global_log_line(error_msg)
            _global_log_line("")
            _global_log_line("Full traceback:")
            for line in error_trace.split('\n'):
                if line:  # Skip empty lines
                    _global_log_line(line)
            _global_log_line("=" * 80)

        sys.exit(1)
