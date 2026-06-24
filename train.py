import os
import sys
import signal
import traceback
import math
from datetime import datetime
from pathlib import Path
import argparse
import random
import numpy as np
import torch
# from fontTools.misc.cython import returns
# from tensorboardX import SummaryWriter
from torch.nn import functional as F
from tqdm import tqdm

from Prompt.promptChooser import PromptChooser
# from dataset.medical_few import MedDataset
from dataset.video_csv import VideoCSVDataset
from CLIP.clip import create_model
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from loss import Loss_detection  # NOTE: FocalLoss, BinaryDiceLoss used only by the (disabled) image-segmentation path
# from utils import augment
import wandb
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
import json
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
KEY_FILE = "key.json" # wandb key for login

# Global log function (will be set in main)
_global_log_line = None

"""
Positive values: Both Segmentation and Detection tasks, Negative values: Only Detection task
"""
CLASS_INDEX = {'Brain':3, 'Liver':2, 'Retina_RESC':1, 'Retina_OCT2017':-1, 'Chest':-2, 'Histopathology':-3, 'Bleeding': -4}


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
    parser.add_argument('--save_model',  action="store_true")
    parser.add_argument('--save_path', type=str, default='./ckpt/')
    parser.add_argument('--img_size', type=int, default=240)
    parser.add_argument('--csv_dir', type=str, default='', help='Directory containing train/val/test CSVs')
    parser.add_argument('--train_csv', type=str, default='', help='Path to train CSV')
    parser.add_argument('--val_csv', type=str, default='', help='Path to val CSV')
    parser.add_argument('--test_csv', type=str, default='', help='Path to test CSV')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    parser.add_argument('--sample_mode', type=str, default='entropy_segment_jitter',
                        help="Frame sampling: 'uniform', 'uniform_jitter', "
                             "'entropy_segment', 'entropy_segment_jitter'")
    parser.add_argument('--entropy_method', type=str, default='entropy',
                        choices=['entropy'],
                        help='Scoring method for entropy_segment modes: entropy (Red channel)')
    parser.add_argument('--segment_size', type=int, default=4,
                        help='Segment size for hierarchical elimination in entropy_segment modes')
    parser.add_argument("--epoch", type=int, default=90, help="epochs")
    parser.add_argument("--epoch_step_size", type=int, default=None,
                        help="Max training steps per epoch. If not set, uses full dataset length.")
    parser.add_argument("--val_epoch_step_size", type=int, default=None,
                        help="Max validation steps per epoch. If not set, uses full validation dataset.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--features_list", type=int, nargs="+", default=[6, 12, 18, 24], help="Layers to apply adapter on")
    parser.add_argument('--seed', type=int, default=111)
    parser.add_argument('--shot', type=int, default=16, help="number of samples for training")
    parser.add_argument('--iterate', type=int, default=0)


    parser.add_argument('--text_mood', type=str, default='learnable_all', help=" the prompt strategy: fix, learnable_all, learnable_abnormal")
    parser.add_argument('--contrast_mood', type=str, default='yes', help="no, yes")
    # parser.add_argument('--adapter_res_mood', type=float, default=0.0, help="whether to apply residual after adapters, 0.0(for no residual), 0.2,0.4,0.8... or any value < 1")


    parser.add_argument('--dec_type', type=str, default='mean', help="aggregation of adapters outputs: mean, max, both,"
                                                                     " they account for global or localized decisions")
    parser.add_argument('--loss_type', type=str, default='sigmoid', help="Loss type: softmax, sigmoid")

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
                        help="Number of transformer layers in temporal adapter (ActionCLIP Transf default: 6)")
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
    parser.add_argument('--use_weighted_sampler', action='store_true', default=True,
                        help='Use weighted random sampling for class imbalance in video training')
    parser.add_argument('--disable_weighted_sampler', action='store_false', dest='use_weighted_sampler',
                        help='Disable weighted random sampling')
    parser.add_argument('--use_lr_schedule', action='store_true', default=True,
                        help='Use warmup + cosine LR schedule')
    parser.add_argument('--disable_lr_schedule', action='store_false', dest='use_lr_schedule',
                        help='Disable LR schedule')
    parser.add_argument('--warmup_epochs', type=float, default=3.0,
                        help='Warmup epochs for LR schedule')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                        help='Minimum LR ratio for cosine schedule')

    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='Path to a .pth checkpoint to load as pre-trained weights before training. '
                             'Adapter weights (spatial + temporal) are always loaded. '
                             'Text prompts are only loaded if --load_prompts is also set.')
    parser.add_argument('--load_prompts', action='store_true',
                        help='When loading a checkpoint, also restore the learnable text prompt weights. '
                             'Useful for same-domain fine-tuning; leave off for a new dataset '
                             '(prompts will be re-initialized from scratch).')

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
    log_path = results_dir / f"train_{timestamp}.log"

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
    log_line("TRAINING CONFIGURATION")
    log_line("=" * 80)
    log_line(f"Experiment Name: {args.exp_name}")
    log_line(f"Output Directory: {args.output_dir}")
    log_line(f"Log File: {log_path}")
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
        log_line(f"Temporal Adapter Type: {getattr(args, 'temporal_adapter_type', 'transf')}")
        log_line(f"Temporal Transformer Layers: {getattr(args, 'temporal_layers', 2)}")
    log_line("")
    log_line("--- Training Configuration ---")
    log_line(f"Batch Size: {args.batch_size}")
    log_line(f"Epochs: {args.epoch}")
    log_line(f"Learning Rate: {args.learning_rate}")
    log_line(f"Seed: {args.seed}")
    log_line("")
    log_line("--- Prompt & Loss Configuration ---")
    log_line(f"Object/Domain: {args.obj}")
    log_line(f"Text Mood: {args.text_mood}")
    log_line(f"Prompt Style: {args.prompt_style}")
    log_line(f"Contrast Mood: {args.contrast_mood}")
    log_line(f"Decision Type: {args.dec_type}")
    log_line(f"Loss Type: {args.loss_type}")
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

    ##############################################
    # start a new wandb run to track this script #
    ##############################################
    # Read wandb API key from key.json
    if os.path.exists(KEY_FILE):
        try:
            with open(KEY_FILE, "r") as f:
                key_data = json.load(f)
                wandb.login(key=key_data["api_key"])
                print("Successfully logged in to wandb using key.json")
        except Exception as e:
            print(f"Failed to read wandb API key from {KEY_FILE}: {e}")
    else:
        print(f"No {KEY_FILE} found. Create a file with format: {{\"api_key\": \"YOUR-WANDB-API-KEY\"}}")

    wandb.init(
        # set the wandb project where this run will be logged
        project="DBT-Bleed",
        name=args.exp_name,
        config=args
    )

    ############################
    # Instantiating CLIP model #
    ############################
    clip_model = create_model(model_name=args.model_name, img_size=args.img_size, device=device,
                             pretrained=args.pretrain, require_pretrained=True)
    clip_model.to(device)
    clip_model.eval()
    for param in clip_model.parameters():
        param.requires_grad = False

    # add adapters on CLIP (adapter/temporal params require grad by default)
    model = CLIP_Inplanted(args, clip_model=clip_model).to(device)
    model.eval()

    # Load pre-trained adapter weights (spatial + temporal) from a prior checkpoint
    _ckpt_data = None  # stored for later prompt loading
    if args.load_checkpoint:
        if not os.path.exists(args.load_checkpoint):
            raise FileNotFoundError(f"--load_checkpoint: file not found: {args.load_checkpoint}")
        _ckpt_data = torch.load(args.load_checkpoint, map_location=device)
        model.normal_det_adapters.load_state_dict(_ckpt_data['normal_det_adapters'])
        model.abnormal_det_adapters.load_state_dict(_ckpt_data['abnormal_det_adapters'])
        model.temporal_normal.load_state_dict(_ckpt_data['temporal_normal'])
        model.temporal_abnormal.load_state_dict(_ckpt_data['temporal_abnormal'])
        log_line(f"Loaded pre-trained adapter weights from: {args.load_checkpoint}")
        if not args.load_prompts:
            log_line("Text prompts will be re-initialized (use --load_prompts to restore them too)")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        log_line(f"Using {torch.cuda.device_count()} GPUs for training")
        model = torch.nn.DataParallel(model)
    else:
        log_line("Using single GPU for training")

    # Helper to access underlying model (handles DataParallel)
    def get_model(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    # Log model parameters
    model_unwrapped = get_model(model)
    total_params = sum(p.numel() for p in model_unwrapped.parameters())
    trainable_params = sum(p.numel() for p in model_unwrapped.parameters() if p.requires_grad)
    temporal_params = sum(p.numel() for p in model_unwrapped.temporal_normal.parameters()) + \
                      sum(p.numel() for p in model_unwrapped.temporal_abnormal.parameters())
    adapter_params = sum(p.numel() for p in model_unwrapped.normal_det_adapters.parameters()) + \
                     sum(p.numel() for p in model_unwrapped.abnormal_det_adapters.parameters())

    log_line("--- Model Parameters ---")
    log_line(f"Total parameters: {total_params:,}")
    log_line(f"Trainable parameters: {trainable_params:,}")
    log_line(f"  - Temporal adapters: {temporal_params:,}")
    log_line(f"  - Spatial adapters: {adapter_params:,}")
    log_line(f"Frozen (CLIP backbone): {total_params - trainable_params:,}")
    log_line("")

    def _resolve_csv(csv_dir, csv_path, default_name):
        if csv_path:
            return csv_path
        if csv_dir:
            return os.path.join(csv_dir, default_name)
        return ""

    want_video = bool(args.csv_dir or args.train_csv or args.val_csv or args.test_csv)
    train_csv = _resolve_csv(args.csv_dir, args.train_csv, "train.csv")
    val_csv = _resolve_csv(args.csv_dir, args.val_csv, "val.csv")
    test_csv = _resolve_csv(args.csv_dir, args.test_csv, "test.csv")
    use_video = False
    if want_video:
        if not train_csv:
            raise ValueError("Video mode requires --train_csv or --csv_dir with train.csv")
        if not os.path.exists(train_csv):
            raise FileNotFoundError(f"Train CSV not found: {train_csv}")
        if not val_csv or not os.path.exists(val_csv):
            raise FileNotFoundError("Validation CSV not found for video mode (val.csv required)")
        eval_csv = val_csv
        use_video = True
        if args.temporal_per_layer:
            log_line("Warning: --temporal_per_layer is enabled. Shared temporal adapter is recommended first for stability.")
    else:
        eval_csv = ""

    # load dataset
    kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
    if use_video:
        train_dataset = VideoCSVDataset(
            csv_path=train_csv,
            num_frames=args.num_frames,
            resize=args.img_size,
            sample_mode=args.sample_mode,
            is_train=True,
            entropy_method=args.entropy_method,
            segment_size=args.segment_size,
        )
        test_dataset = VideoCSVDataset(
            csv_path=eval_csv,
            num_frames=args.num_frames,
            resize=args.img_size,
            sample_mode=args.sample_mode,
            is_train=False,
            entropy_method=args.entropy_method,
            segment_size=args.segment_size,
        )
        train_loader_kwargs = {'batch_size': args.batch_size, **kwargs}
        if args.use_weighted_sampler:
            train_labels = np.array([int(e["gt"]) for e in train_dataset.entries], dtype=np.int64)
            class_counts = np.bincount(train_labels, minlength=2).astype(np.float64)
            class_weights = 1.0 / np.clip(class_counts, a_min=1.0, a_max=None)
            sample_weights = class_weights[train_labels]
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset, sampler=sampler, shuffle=False, **train_loader_kwargs
            )
            log_line(
                f"Weighted sampler enabled: class_counts={class_counts.astype(int).tolist()} "
                f"class_weights={[round(float(w), 6) for w in class_weights]}"
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, **train_loader_kwargs
            )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
        )
        log_line("--- Dataset Information ---")
        log_line(f"Train CSV: {train_csv}")
        log_line(f"Val CSV: {eval_csv}")
        log_line(f"Train samples: {len(train_dataset)}")
        log_line(f"Val samples: {len(test_dataset)}")
        log_line(f"Train batches: {len(train_loader)}")
        log_line(f"Val batches: {len(test_loader)}")
        log_line("")
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # test_dataset = MedDataset(args.data_path, args.obj, args.img_size, args.shot, args.iterate)
        # test_loader = torch.utils.data.DataLoader(
            # test_dataset, batch_size=args.batch_size * 2, shuffle=False, **kwargs
        # )

        # # few-shot image augmentation
        # augment_abnorm_img, augment_abnorm_mask = augment(
            # test_dataset.fewshot_abnorm_img, test_dataset.fewshot_abnorm_mask
        # )
        # augment_normal_img, augment_normal_mask = augment(test_dataset.fewshot_norm_img)

        # augment_fewshot_img = torch.cat([augment_abnorm_img, augment_normal_img], dim=0)
        # augment_fewshot_mask = torch.cat([augment_abnorm_mask, augment_normal_mask], dim=0)

        # augment_fewshot_label = torch.cat(
            # [torch.Tensor([1] * len(augment_abnorm_img)), torch.Tensor([0] * len(augment_normal_img))], dim=0
        # )

        # train_dataset = torch.utils.data.TensorDataset(
            # augment_fewshot_img, augment_fewshot_mask, augment_fewshot_label
        # )
        # train_loader = torch.utils.data.DataLoader(
            # train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs
        # )

    # losses
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # loss_focal = FocalLoss()
    # loss_dice = BinaryDiceLoss()
    loss_det = Loss_detection(args= args, device=device, loss_type=args.loss_type,dec_type=args.dec_type, lr=args.learning_rate)


    # text maker
    text_chooser = PromptChooser(clip_model, args, device)

    # Optionally restore learnable prompt weights from checkpoint
    if _ckpt_data is not None and args.load_prompts:
        loaded_any_prompt = False
        if args.text_mood == 'fix':
            log_line("Skipping prompt loading: --text_mood=fix uses fixed (non-learnable) prompts")
        else:
            if 'prompt_maker_abnormal' in _ckpt_data and hasattr(text_chooser, 'prompt_maker_abnormal'):
                text_chooser.prompt_maker_abnormal.load_state_dict(_ckpt_data['prompt_maker_abnormal'], strict=False)
                loaded_any_prompt = True
            if 'prompt_maker_normal' in _ckpt_data and hasattr(text_chooser, 'prompt_maker_normal'):
                text_chooser.prompt_maker_normal.load_state_dict(_ckpt_data['prompt_maker_normal'], strict=False)
                loaded_any_prompt = True
            if loaded_any_prompt:
                log_line("Restored learnable text prompt weights from checkpoint")
            else:
                log_line("Warning: --load_prompts set but no matching prompt keys found in checkpoint")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    def _is_torch_optimizer(opt):
        return hasattr(opt, "param_groups")

    def _optimizer_has_any_grad(opt):
        if not _is_torch_optimizer(opt):
            return False
        for group in opt.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    return True
        return False

    scheduler_main = None
    scheduler_loss = None
    scheduler_text = None
    if use_video and args.use_lr_schedule:
        steps_per_epoch = max(1, len(train_loader))
        if args.epoch_step_size is not None:
            steps_per_epoch = min(steps_per_epoch, args.epoch_step_size)
        total_steps = max(1, int(args.epoch * steps_per_epoch))
        warmup_steps = int(args.warmup_epochs * steps_per_epoch)

        def lr_lambda(current_step):
            if warmup_steps > 0 and current_step < warmup_steps:
                return float(current_step + 1) / float(warmup_steps)
            progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

        model_unwrapped = get_model(model)
        scheduler_main = torch.optim.lr_scheduler.LambdaLR(model_unwrapped.all_adapters_optimizer, lr_lambda=lr_lambda)
        scheduler_loss = torch.optim.lr_scheduler.LambdaLR(loss_det.optimizer, lr_lambda=lr_lambda)
        if _is_torch_optimizer(text_chooser.text_optimizer):
            scheduler_text = torch.optim.lr_scheduler.LambdaLR(text_chooser.text_optimizer, lr_lambda=lr_lambda)
        log_line(
            f"LR schedule enabled: warmup_epochs={args.warmup_epochs}, min_lr_ratio={args.min_lr_ratio}, "
            f"total_steps={total_steps}, warmup_steps={warmup_steps}"
        )

    best_result = 0
    best_f1 = -1.0
    best_ap = -1.0
    best_auc = -1.0
    log_line(f"Run dir: {run_root}")
    log_line(f"Models dir: {models_dir}")
    log_line(f"Predictions dir: {predictions_dir}")
    log_line(f"Results dir: {results_dir}")
    log_line(f"Batch size: {args.batch_size}")
    if args.epoch_step_size is not None:
        log_line(f"Epoch step size: {args.epoch_step_size} (full dataset: {len(train_loader)} steps)")
    for epoch in range(args.epoch):
        log_line(f"epoch {epoch}:")
        loss_list = []
        if use_video:
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch-1} [Train]",
                              total=args.epoch_step_size or len(train_loader), leave=True)
            for step_i, (video, label) in enumerate(train_pbar):
                if args.epoch_step_size is not None and step_i >= args.epoch_step_size:
                    break
                video = video.to(device)
                with torch.cuda.amp.autocast():
                    text_features = text_chooser()  # [768, 2]
                    _, det_model, _ = model(video, text_features)
                    det_loss = 0
                    video_label = label.to(device)
                    for layer in range(len(det_model)):
                        det_scores_cur = det_model[layer]  # [B, 2]
                        det_loss += loss_det(det_scores_cur.unsqueeze(1), video_label)

                    loss = det_loss
                    model_unwrapped = get_model(model)
                    model_unwrapped.all_adapters_optimizer.zero_grad(set_to_none=True)
                    text_chooser.text_optimizer.zero_grad()
                    loss_det.optimizer.zero_grad()

                    scaler.scale(loss).backward()
                    model_has_grad = _optimizer_has_any_grad(model_unwrapped.all_adapters_optimizer)
                    loss_has_grad = _optimizer_has_any_grad(loss_det.optimizer)
                    text_has_grad = _optimizer_has_any_grad(text_chooser.text_optimizer)

                    if model_has_grad:
                        scaler.unscale_(model_unwrapped.all_adapters_optimizer)
                    if loss_has_grad:
                        scaler.unscale_(loss_det.optimizer)
                    if text_has_grad:
                        scaler.unscale_(text_chooser.text_optimizer)

                    # Gradient clipping to prevent exploding gradients
                    torch.nn.utils.clip_grad_norm_(model_unwrapped.normal_det_adapters.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(model_unwrapped.abnormal_det_adapters.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(model_unwrapped.temporal_normal.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(model_unwrapped.temporal_abnormal.parameters(), max_norm=5.0)
                    if hasattr(text_chooser, 'prompt_maker_normal'):
                        torch.nn.utils.clip_grad_norm_(text_chooser.prompt_maker_normal.parameters(), max_norm=5.0)
                    if hasattr(text_chooser, 'prompt_maker_abnormal'):
                        torch.nn.utils.clip_grad_norm_(text_chooser.prompt_maker_abnormal.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(loss_det.parameters(), max_norm=5.0)

                    if model_has_grad:
                        scaler.step(model_unwrapped.all_adapters_optimizer)
                    if text_has_grad:
                        scaler.step(text_chooser.text_optimizer)
                    else:
                        text_chooser.text_optimizer.step()
                    if loss_has_grad:
                        scaler.step(loss_det.optimizer)
                    scaler.update()

                    if scheduler_main is not None and model_has_grad:
                        scheduler_main.step()
                    if scheduler_text is not None and text_has_grad:
                        scheduler_text.step()
                    if scheduler_loss is not None and loss_has_grad:
                        scheduler_loss.step()

                    loss_list.append(loss.item())
                    train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
        # else:
            # train_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epoch-1} [Train]",
                              # total=args.epoch_step_size or len(train_loader), leave=True)
            # for step_i, (image, gt, label) in enumerate(train_pbar):
                # if args.epoch_step_size is not None and step_i >= args.epoch_step_size:
                    # break
                # image = image.to(device)
                # with torch.cuda.amp.autocast():
                    # text_features = text_chooser() # [768, 2] 2 ensembled prompts => t_n: [:, 0], t_ab :[:,1]
                    # # det: 4 tensors of [289,1]----- seg: 4 tensors of [1,1,240,240]
                    # _ , det_model, seg_model = model(image, text_features)
                    # # det loss
                    # det_loss = 0
                    # image_label = label.to(device)
                    # for layer in range(len(det_model)):
                        # det_scores_cur = det_model[layer] # [batch, 289, 2]
                        # det_loss += loss_det(det_scores_cur, image_label)

                    # if CLASS_INDEX[args.obj] > 0:
                        # # pixel level
                        # seg_loss = 0
                        # mask = gt.squeeze(0).to(device)
                        # mask[mask > 0.5], mask[mask <= 0.5] = 1, 0
                        # for layer in range(len(seg_model)):
                            # seg_scores_cur = seg_model[layer] # [batch size, 2,240,240]
                            # seg_scores_cur = loss_det.sync_AS(seg_scores_cur)
                            # seg_loss += loss_focal(seg_scores_cur, mask)
                            # seg_loss += loss_dice(seg_scores_cur[:, 1, :, :].unsqueeze(1), mask)
                        # loss = seg_loss + det_loss
                        # loss.requires_grad_(True)
                        # model.all_adapters_optimizer.zero_grad()
                        # text_chooser.text_optimizer.zero_grad()
                        # loss_det.optimizer.zero_grad()


                        # loss.backward()


                        # model.all_adapters_optimizer.step()
                        # text_chooser.text_optimizer.step()
                        # loss_det.optimizer.step()
                    # else:
                        # loss = det_loss
                        # loss.requires_grad_(True)
                        # model.all_adapters_optimizer.zero_grad()
                        # text_chooser.text_optimizer.zero_grad()
                        # loss_det.optimizer.zero_grad()

                        # loss.backward()


                        # model.all_adapters_optimizer.step()
                        # text_chooser.text_optimizer.step()
                        # loss_det.optimizer.step()

                    # loss_list.append(loss.item())
                    # train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        log_line(f"Train Loss: {np.mean(loss_list):.4f}")
        if epoch > -1:
            if use_video:
                metrics, sim_text, val_loss = test(args, model, test_loader, text_chooser, loss_det, use_video=True, val_step_limit=args.val_epoch_step_size)
                log_line(
                    f"Val metrics: AUC={metrics['auc']:.4f} AP={metrics['ap']:.4f} "
                    f"F1={metrics['f1']:.4f} Val Loss={val_loss:.4f}"
                )
                wandb.log({
                    'epoch': epoch,
                    "sim_texts": sim_text.item(),
                    "AUC": metrics["auc"],
                    "AP": metrics["ap"],
                    "F1": metrics["f1"],
                    'train_loss': np.mean(loss_list),
                    'val_loss': val_loss,
                })

                if args.save_model:
                    model_unwrapped = get_model(model)
                    save_dict = {
                                'normal_det_adapters': model_unwrapped.normal_det_adapters.state_dict(),
                                'abnormal_det_adapters': model_unwrapped.abnormal_det_adapters.state_dict(),
                                'temporal_normal': model_unwrapped.temporal_normal.state_dict(),
                                'temporal_abnormal': model_unwrapped.temporal_abnormal.state_dict(),
                    }
                    save_dict = text_chooser.save_prompt(save_dict)

                    if metrics["f1"] > best_f1:
                        best_f1 = metrics["f1"]
                        ckp_path = models_dir / "best_f1.pth"
                        torch.save(save_dict, ckp_path)
                        log_line(f"Saved best_f1 model to {ckp_path}")
                    if metrics["ap"] > best_ap:
                        best_ap = metrics["ap"]
                        ckp_path = models_dir / "best_ap.pth"
                        torch.save(save_dict, ckp_path)
                        log_line(f"Saved best_ap model to {ckp_path}")
                    if metrics["auc"] > best_auc:
                        best_auc = metrics["auc"]
                        ckp_path = models_dir / "best_auc.pth"
                        torch.save(save_dict, ckp_path)
                        log_line(f"Saved best_auc model to {ckp_path}")
            # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
            # else:
                # result, auc, pauc, sim_text, val_loss = test(args, model, test_loader, text_chooser, loss_det, use_video=use_video, val_step_limit=args.val_epoch_step_size)
                # log_line(f"Val metrics: AUC={auc:.4f} pAUC={pauc:.4f} Val Loss={val_loss:.4f}")
                # # Log test results
                # wandb.log({
                    # 'epoch': epoch,
                    # "sim_texts": sim_text.item(),
                    # "AUC+pAUC": result,
                    # "AUC": auc,
                    # "pAUC": pauc,
                    # 'train_loss': np.mean(loss_list),
                    # 'val_loss': val_loss,
                # })

                # if result > best_result:
                    # best_result = result
                    # log_line("Best result")
                    # wandb.log({
                        # 'epoch': epoch,
                        # "AUC+pAUCbest": result,
                        # "AUCbest": auc,
                        # "pAUCbest": pauc,
                    # })
                    # if args.save_model:
                        # ckp_path = os.path.join(args.save_path, f'{args.obj}.pth')
                        # model_unwrapped = get_model(model)
                        # save_dict = {
                                    # 'normal_det_adapters': model_unwrapped.normal_det_adapters.state_dict(),
                                    # 'abnormal_det_adapters': model_unwrapped.abnormal_det_adapters.state_dict(),
                                    # 'temporal_normal': model_unwrapped.temporal_normal.state_dict(),
                                    # 'temporal_abnormal': model_unwrapped.temporal_abnormal.state_dict(),
                        # }
                        # # Update save_dict with text-related components
                        # save_dict = text_chooser.save_prompt(save_dict)
                        # # Save the checkpoint
                        # torch.save(save_dict, ckp_path)

def _compute_video_metrics(scores, labels):
    scores = np.asarray(scores, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if labels.max() == labels.min():
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "f1": float("nan"),
        }

    auc = roc_auc_score(labels, scores)
    ap = average_precision_score(labels, scores)
    preds = (scores >= 0.5).astype(np.int32)
    f1 = float(f1_score(labels, preds))

    return {"auc": float(auc), "ap": float(ap), "f1": f1}


def test(args, model, test_loader, text_chooser, loss_det, use_video=False, val_step_limit=None):
    image_list = []
    gt_list = []
    gt_mask_list = []
    det_final = []
    seg_final = []
    val_loss_list = []

    with torch.no_grad():
        text_features = text_chooser()  # [768,2]
        sim_text = F.cosine_similarity(text_features[:, 0], text_features[:, 1], dim=0)

    if use_video:
        for val_step_i, (video, y) in enumerate(tqdm(test_loader, desc="[Validation]",
                                                       total=val_step_limit or len(test_loader), leave=False)):
            if val_step_limit is not None and val_step_i >= val_step_limit:
                break
            video = video.to(device)
            with torch.no_grad(), torch.cuda.amp.autocast():
                _, det_model, _ = model(video, text_features)
                det_loss = 0
                anomaly_scores_all = 0
                video_label = y.to(device)
                for layer in range(len(det_model)):
                    det_scores_cur = det_model[layer]  # [B, 2]
                    det_loss += loss_det(det_scores_cur.unsqueeze(1), video_label)
                    anomaly_scores_all += loss_det.validation(det_scores_cur.unsqueeze(1))

                val_loss_list.append(det_loss.item())
                det_final.extend(anomaly_scores_all.cpu().numpy())
                gt_list.extend(y.cpu().detach().numpy())
    # --- [DBT-Bleed] image-anomaly path disabled (kept for reference) ---
    # else:
        # for val_step_i, (image, y, mask) in enumerate(tqdm(test_loader, desc="[Validation]",
                                                              # total=val_step_limit or len(test_loader), leave=False)):
            # if val_step_limit is not None and val_step_i >= val_step_limit:
                # break
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
                # det_loss = 0
                # anomaly_scores_all = 0
                # image_label = y.to(device)
                # for layer in range(len(det_model)):
                    # det_scores_cur = det_model[layer] # [batch, 289,2]
                    # det_loss += loss_det(det_scores_cur, image_label)
                    # anomaly_scores_all += loss_det.validation(det_scores_cur)

                # val_loss_list.append(det_loss.item())
                # det_final.extend(anomaly_scores_all.cpu().numpy())

                # gt_mask_list.extend(mask.squeeze(1).cpu().detach().numpy())
                # gt_list.extend(y.cpu().detach().numpy())
                # image_list.extend(image.cpu().detach().numpy())

    gt_list = np.array(gt_list)
    if use_video:
        det_scores = np.array(det_final)
        if np.isnan(det_scores).any():
            n_nan = np.isnan(det_scores).sum()
            print(f"WARNING: {n_nan}/{len(det_scores)} NaN in det_scores, replacing with 0.5")
            det_scores = np.nan_to_num(det_scores, nan=0.5)
        det_scores = (det_scores - det_scores.min()) / (1e-4 + det_scores.max() - det_scores.min())
        metrics = _compute_video_metrics(det_scores, gt_list)
        avg_val_loss = float(np.mean(val_loss_list)) if val_loss_list else float("nan")
        print(f"Video AUC: {metrics['auc']:.4f} AP: {metrics['ap']:.4f} F1: {metrics['f1']:.4f} Val Loss: {avg_val_loss:.4f}")
        return metrics, sim_text, avg_val_loss

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
        # avg_val_loss = float(np.mean(val_loss_list)) if val_loss_list else float("nan")
        # return seg_roc_auc + roc_auc_im, roc_auc_im, seg_roc_auc, sim_text, avg_val_loss

    # else:
        # det_image_scores_zero = np.array(det_final)
        # det_image_scores_zero = (det_image_scores_zero - det_image_scores_zero.min()) / ( 1e-4 + det_image_scores_zero.max() - det_image_scores_zero.min())
        # img_roc_auc_det = roc_auc_score(gt_list, det_image_scores_zero)
        # print(f'{args.obj} AUC : {round(img_roc_auc_det, 4)} from det')
        # avg_val_loss = float(np.mean(val_loss_list)) if val_loss_list else float("nan")
        # return img_roc_auc_det, img_roc_auc_det , img_roc_auc_det, sim_text, avg_val_loss

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print("TRAINING INTERRUPTED BY USER (Ctrl+C)")
        print("=" * 80)
        if _global_log_line:
            _global_log_line("=" * 80)
            _global_log_line("TRAINING INTERRUPTED BY USER (Ctrl+C)")
            _global_log_line("=" * 80)
        sys.exit(0)
    except Exception as e:
        error_msg = f"FATAL ERROR: {type(e).__name__}: {str(e)}"
        error_trace = traceback.format_exc()

        # Print to console
        print(f"\n{'='*80}")
        print("TRAINING FAILED WITH ERROR")
        print('='*80)
        print(error_msg)
        print("\nFull traceback:")
        print(error_trace)
        print('='*80)

        # Log to file if available
        if _global_log_line:
            _global_log_line("=" * 80)
            _global_log_line("TRAINING FAILED WITH ERROR")
            _global_log_line("=" * 80)
            _global_log_line(error_msg)
            _global_log_line("")
            _global_log_line("Full traceback:")
            for line in error_trace.split('\n'):
                if line:  # Skip empty lines
                    _global_log_line(line)
            _global_log_line("=" * 80)

        sys.exit(1)
