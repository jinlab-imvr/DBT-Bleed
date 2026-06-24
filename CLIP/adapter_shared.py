import math
import numpy as np
import torch
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        """Construct a layernorm module in the TF style (epsilon inside the square root)."""
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TemporalTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks((x))


class TemporalAdapter(nn.Module):
    """Multi-scale Temporal Adapter (MTA)."""
    def __init__(self, embed_dim: int, num_frames: int, layers: int, heads: int, context_length: int, pool_mode: str = "mean"):
        super().__init__()
        self.num_frames = num_frames
        self.context_length = context_length
        self.pool_mode = pool_mode
        self.frame_position_embeddings = nn.Embedding(context_length, embed_dim)
        self.transformer = TemporalTransformer(width=embed_dim, layers=layers, heads=heads)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, C] per-frame features
        Returns:
            [B, C] video embedding
        """
        b, t, c = x.size()
        if t > self.context_length:
            raise ValueError(
                f"num_frames ({t}) exceeds position embedding size ({self.context_length})"
            )
        x_original = x
        position_ids = torch.arange(t, dtype=torch.long, device=x.device)
        position_ids = position_ids.unsqueeze(0).expand(b, -1)
        frame_position_embeddings = self.frame_position_embeddings(position_ids)
        x = x + frame_position_embeddings
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = x.type(x_original.dtype) + x_original
        if self.pool_mode == "max":
            return x.max(dim=1)[0]
        return x.mean(dim=1, keepdim=False)


class MCNNFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(MCNNFC, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=c_in, out_channels=c_in, kernel_size=1, stride=1),
        )
        self.relu = nn.LeakyReLU(inplace=False)
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.norm1 = nn.InstanceNorm1d(c_in)

    def forward(self, x, det_only=False):
        x = x.permute(1, 2, 0)
        x = self.cnn(x)  # Apply 1D convolution
        # Permute back to [290, batch_size, 768] for LayerNorm
        x = x.permute(2, 0, 1)  # Shape: [seq_len, batch_size, bottleneck]
        x = self.norm1(x)
        x = self.relu(x)
        y = self.fc1(x)
        if det_only:
            return y
        z = self.fc2(x)
        return y, z


class MFCFC(nn.Module):
    def __init__(self, c_in, bottleneck=768):
        super(MFCFC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, c_in, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.norm1 = nn.InstanceNorm1d(c_in)
        self.fc2 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=False)
        )

    def forward(self, x, det_only=False):
        x = self.fc1(x)
        x = self.norm1(x)
        y = self.fc2(x)
        if det_only:
            return y
        z = self.fc3(x)
        return y, z


class MFCFC_v2(nn.Module):
    """
    Improved MFCFC adapter:
      1. LayerNorm instead of InstanceNorm (normalizes across channels, not tokens)
      2. Bottleneck hidden layer (compression forces discriminative features)
      3. No activation on output projections (full direction space for cosine sim)
      4. Dropout for regularization
      5. Residual skip connection with zero-init output (starts near identity)
    """
    def __init__(self, c_in, bottleneck=768, hidden=256, dropout=0.1):
        super(MFCFC_v2, self).__init__()
        self.norm = nn.LayerNorm(c_in)
        self.fc_down = nn.Linear(c_in, hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        # Output projections — no activation (goes into L2-norm + cosine sim)
        self.fc_det = nn.Linear(hidden, bottleneck)
        self.fc_seg = nn.Linear(hidden, bottleneck)
        # Residual: linear projection for dimension change (1024 -> 768)
        self.skip = nn.Linear(c_in, bottleneck, bias=False)
        # Zero-init output so adapter starts near identity (skip dominates early)
        nn.init.zeros_(self.fc_det.weight)
        nn.init.zeros_(self.fc_det.bias)
        nn.init.zeros_(self.fc_seg.weight)
        nn.init.zeros_(self.fc_seg.bias)

    def forward(self, x, det_only=False):
        residual = self.skip(x)
        h = self.drop(self.act(self.fc_down(self.norm(x))))
        y = self.fc_det(h) + residual
        if det_only:
            return y
        z = self.fc_seg(h) + residual
        return y, z


class LKA_Adapter(nn.Module):
    """
    Input:  [seq_len, batch_size, c_in]   (seq = 1 CLS + H*W patches)
    Output: [seq_len, batch_size, bottleneck]
    """

    def __init__(self, c_in, bottleneck=768, d_hat=8, kernel_size=7, dropout=0.1):
        super().__init__()
        self.d_hat = d_hat
        padding = kernel_size // 2

        # Skip connection for residual (handles 1024 → 768 dim change)
        self.skip = nn.Linear(c_in, bottleneck, bias=False)

        # Down projection — matching repo Seven.adapter_down
        self.adapter_down = nn.Linear(c_in, d_hat)

        # Channel-wise large kernel depthwise convolution — matching repo Seven.adapter_conv
        self.adapter_conv = nn.Conv2d(
            d_hat, d_hat, kernel_size=kernel_size, stride=1,
            padding=padding, groups=d_hat
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(dropout)

        # Up projections (det and seg branches) — matching repo Seven.adapter_up
        self.up_det = nn.Linear(d_hat, bottleneck)
        self.up_seg = nn.Linear(d_hat, bottleneck)

        # Initialization matching repo's _init_vit_weights:
        #   Linear: trunc_normal_(std=0.01), bias zeros
        #   Conv2d: kaiming_normal_(fan_out), bias zeros
        self._init_weights()

        # Zero-init output projections so adapter starts near identity
        nn.init.zeros_(self.up_det.weight)
        nn.init.zeros_(self.up_det.bias)
        nn.init.zeros_(self.up_seg.weight)
        nn.init.zeros_(self.up_seg.bias)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, det_only=False):
        # x: [seq, B, c_in]  where seq = 1 (CLS) + P (patches)
        residual = self.skip(x)  # [seq, B, bottleneck]

        # ---- Matches repo Seven.forward exactly ----
        # Step 1: adapter_down
        h = self.adapter_down(x)           # [seq, B, d_hat]

        # Step 2: GELU after down (repo: x_down = self.act(x_down))
        h = self.act(h)

        # Step 3: Separate CLS and patches, apply DWConv on patches
        cls_token = h[:1]   # [1, B, d_hat]
        patches = h[1:]     # [P, B, d_hat]

        P, B, D = patches.shape
        H = int(math.sqrt(P))
        assert H * H == P, f"Patch count {P} is not a perfect square"

        # Reshape [P, B, D] → [B, D, H, H] for Conv2d
        patches = patches.permute(1, 2, 0).view(B, D, H, H)
        patches = self.adapter_conv(patches)
        patches = patches.view(B, D, P).permute(2, 0, 1)  # → [P, B, D]

        # Rejoin CLS + patches
        h = torch.cat([cls_token, patches], dim=0)  # [seq, B, d_hat]

        # Step 4: GELU after conv (repo: x_down = self.act(x_down))
        h = self.act(h)

        # Step 5: Dropout (repo: x_down = self.dropout(x_down))
        h = self.dropout(h)

        # Step 6: adapter_up + residual
        y = self.up_det(h) + residual  # [seq, B, bottleneck]
        if det_only:
            return y
        z = self.up_seg(h) + residual
        return y, z


class MViTFC(nn.Module):
    def __init__(self, c_in, bottleneck, num_heads=8, dropout=0.1):
        super(MViTFC, self).__init__()

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=c_in,
            nhead=num_heads,
            dim_feedforward=c_in,
            dropout=dropout,
            activation='relu'  # Use LeakyReLU as the activation function
        )
        # Bottleneck fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=True)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(c_in, bottleneck, bias=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, det_only=False):
        # Input: [290, batch_size, 1024]
        x = x.permute(1, 0, 2)  # Permute to [batch_size, 290, 1024] for Transformer
        x = self.transformer_encoder(x)  # Apply Vision Transformer encoder
        x = x.permute(1, 0, 2)  # Back to [290, batch_size, 1024]
        y = self.fc(x)  # Apply bottleneck fully connected layers
        if det_only:
            return y
        z = self.fc1(x)
        return y, z

class CLIP_Inplanted(nn.Module):
    def __init__(self,args, clip_model):
        super().__init__()
        self.clipmodel = clip_model
        self.image_encoder = clip_model.visual
        self.features = args.features_list
        # self.res_mood = args.adapter_res_mood
        self.img_size = args.img_size
        ###################
        # adapters set up #
        ###################
        if args.visionA == "MCNNFC":
            self.normal_det_adapters = nn.ModuleList([MCNNFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MCNNFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MCNNFC")
        elif args.visionA == "MFCFC":
            self.normal_det_adapters = nn.ModuleList([MFCFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MFCFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MFCFC")
        elif args.visionA == "MFCFC_v2":
            hidden = getattr(args, "adapter_hidden", 256)
            adapter_dropout = getattr(args, "adapter_dropout", 0.1)
            self.normal_det_adapters = nn.ModuleList([MFCFC_v2(1024, bottleneck=768, hidden=hidden, dropout=adapter_dropout) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList([MFCFC_v2(1024, bottleneck=768, hidden=hidden, dropout=adapter_dropout) for i in range(len(self.features))])
            print(f"MFCFC_v2 (hidden={hidden}, dropout={adapter_dropout})")
        elif args.visionA == "LKA":
            lka_d_hat = getattr(args, "lka_bottleneck", 8)
            lka_ks = getattr(args, "lka_kernel_size", 7)
            lka_drop = getattr(args, "adapter_dropout", 0.1)
            self.normal_det_adapters = nn.ModuleList([LKA_Adapter(1024, bottleneck=768, d_hat=lka_d_hat, kernel_size=lka_ks, dropout=lka_drop) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList([LKA_Adapter(1024, bottleneck=768, d_hat=lka_d_hat, kernel_size=lka_ks, dropout=lka_drop) for i in range(len(self.features))])
            print(f"LKA (d_hat={lka_d_hat}, kernel_size={lka_ks}, dropout={lka_drop})")
        elif args.visionA == "MViTFC":
            self.normal_det_adapters = nn.ModuleList([MViTFC(1024, bottleneck=768) for i in range(len(self.features))])
            self.abnormal_det_adapters = nn.ModuleList( [MViTFC(1024, bottleneck=768) for i in range(len(self.features))] )
            print("MViTFC")
        #######################
        # temporal adapters   #
        #######################
        self.num_frames = getattr(args, "num_frames", 1)
        # DBT-Bleed uses the multi-scale temporal adapter: one MTA per CLIP depth {6,12,18,24}.
        self.temporal_per_layer = True
        self.temporal_adapter_type = "transf"
        embed_dim = self.clipmodel.text_projection.shape[1]
        context_length = self.clipmodel.positional_embedding.shape[0]
        num_temporal = len(self.features)

        # Multi-scale temporal adapter (MTA): one adapter per branch per CLIP depth.
        temporal_layers = getattr(args, "temporal_layers", 2)
        temporal_pool_mode = getattr(args, "temporal_pool_mode", "mean")
        transformer_heads = embed_dim // 64
        self.temporal_normal = nn.ModuleList([
            TemporalAdapter(
                embed_dim=embed_dim,
                num_frames=self.num_frames,
                layers=temporal_layers,
                heads=transformer_heads,
                context_length=context_length,
                pool_mode=temporal_pool_mode,
            )
            for _ in range(num_temporal)
        ])
        self.temporal_abnormal = nn.ModuleList([
            TemporalAdapter(
                embed_dim=embed_dim,
                num_frames=self.num_frames,
                layers=temporal_layers,
                heads=transformer_heads,
                context_length=context_length,
                pool_mode=temporal_pool_mode,
            )
            for _ in range(num_temporal)
        ])
        print(f"Temporal config (transf): {num_temporal} adapters per branch, "
              f"{temporal_layers} transformer layers each, pool_mode={temporal_pool_mode}")
        self.all_adapters_optimizer = torch.optim.Adam(
            [
                {'params': self.normal_det_adapters.parameters(), 'lr': args.learning_rate},
                {'params': self.abnormal_det_adapters.parameters(), 'lr': args.learning_rate},
                {'params': self.temporal_normal.parameters(), 'lr': args.learning_rate},
                {'params': self.temporal_abnormal.parameters(), 'lr': args.learning_rate},
            ],
            betas=(0.5, 0.999)
        )
        print(f"Adapter LR (spatial + temporal): {args.learning_rate}")

        ###################
        # contrast set up #
        ###################
        self.contrast_mood = args.contrast_mood
        if self.contrast_mood == "no":
            self.contrast = lambda a, b: (a)
        elif self.contrast_mood== "yes":
            self.contrast = lambda a, b: (a - b)
        else:
            print("ERROR, no such a contrast mood")


    def forward(self, x, text_features):
        is_video = x.dim() == 5
        if is_video:
            batch_size, num_frames, channels, height, width = x.shape
            x = x.view(batch_size * num_frames, channels, height, width)
            # CLIP-style text normalization and temperature scaling for stable similarities.
            text_features = F.normalize(text_features, dim=0, eps=1e-8)
            logit_scale = self.clipmodel.logit_scale.exp().clamp(max=100).to(x.dtype)
        else:
            batch_size = x.shape[0]
            num_frames = 1

        x = self.image_encoder.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.image_encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                          device=x.device),
             x], dim=1)
        x = x + self.image_encoder.positional_embedding.to(x.dtype)

        x = self.image_encoder.patch_dropout(x)
        x = self.image_encoder.ln_pre(x)

        x = x.permute(1, 0, 2)

        det_scores = []
        seg_scores = []
        for i in range(24):
            if i + 1 == 12:
                x, attn = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            else:
                x, attn_map = self.image_encoder.transformer.resblocks[i](x, attn_mask=None)
            if (i + 1) in self.features:
                adapter_idx = self.features.index(i + 1)
                if is_video:
                    # detection-only adapters (skip seg branch)
                    normal_f_det_i = self.normal_det_adapters[adapter_idx](x, det_only=True)
                    abnormal_f_det_i = self.abnormal_det_adapters[adapter_idx](x, det_only=True)

                    # [seq, B*T, C] -> [B*T, seq, C]
                    normal_f_det_i = normal_f_det_i.permute(1, 0, 2)
                    abnormal_f_det_i = abnormal_f_det_i.permute(1, 0, 2)

                    # drop CLS token [B*T, 290, C] -> [B*T, 289, C]
                    normal_f_det_i = normal_f_det_i[:, 1:, :].contiguous()
                    abnormal_f_det_i = abnormal_f_det_i[:, 1:, :].contiguous()

                    # reshape to [B, T, P, C]  (P = num patches, e.g. 289)
                    normal_f_det_i = normal_f_det_i.view(
                        batch_size, num_frames, normal_f_det_i.shape[1], normal_f_det_i.shape[2]
                    )
                    abnormal_f_det_i = abnormal_f_det_i.view(
                        batch_size, num_frames, abnormal_f_det_i.shape[1], abnormal_f_det_i.shape[2]
                    )

                    t_idx = adapter_idx if self.temporal_per_layer else 0

                    # Spatial mean-pool first, then temporal adapter
                    normal_f_det_i = normal_f_det_i.mean(dim=2)    # [B, T, C]
                    abnormal_f_det_i = abnormal_f_det_i.mean(dim=2)
                    v_normal = self.temporal_normal[t_idx](normal_f_det_i)
                    v_abnormal = self.temporal_abnormal[t_idx](abnormal_f_det_i)

                    # L2-normalize video embeddings.
                    v_normal = F.normalize(v_normal, dim=-1, eps=1e-8)
                    v_abnormal = F.normalize(v_abnormal, dim=-1, eps=1e-8)

                    # Dual contrast at video level.
                    sim_same_normal = (v_normal * text_features[:, 0]).sum(dim=-1)  # [B]
                    sim_cross_normal = (v_normal * text_features[:, 1]).sum(dim=-1)  # [B]
                    sim_det_normal = self.contrast(sim_same_normal, sim_cross_normal)  # [B]

                    sim_same_ab = (v_abnormal * text_features[:, 1]).sum(dim=-1)  # [B]
                    sim_cross_ab = (v_abnormal * text_features[:, 0]).sum(dim=-1)  # [B]
                    sim_det_abnormal = self.contrast(sim_same_ab, sim_cross_ab)  # [B]

                    det_scores_cur = torch.stack([sim_det_normal, sim_det_abnormal], dim=-1)  # [B,2]
                    det_scores_cur = det_scores_cur * logit_scale
                    det_scores.append(det_scores_cur)
                else:
                    normal_f_det_i, normal_f_seg_i = self.normal_det_adapters[adapter_idx](x)
                    abnormal_f_det_i, abnormal_f_seg_i = self.abnormal_det_adapters[adapter_idx](x)

                    # reshape from [290,B,768] to [B,290,768]
                    normal_f_det_i = normal_f_det_i.permute(1, 0, 2)
                    normal_f_seg_i = normal_f_seg_i.permute(1, 0, 2)
                    abnormal_f_det_i = abnormal_f_det_i.permute(1, 0, 2)
                    abnormal_f_seg_i = abnormal_f_seg_i.permute(1, 0, 2)

                    # remove CLS token [B,290,768] -> [B,289,768]
                    normal_f_det_i = normal_f_det_i[:,1:,:]
                    normal_f_seg_i = normal_f_seg_i[:,1:,:]
                    abnormal_f_det_i = abnormal_f_det_i[:,1:,:]
                    abnormal_f_seg_i = abnormal_f_seg_i[:,1:,:]

                    # normalizing adapted features
                    normal_f_det_i = normal_f_det_i /normal_f_det_i.norm(dim=-1, keepdim=True)
                    normal_f_seg_i = normal_f_seg_i /normal_f_seg_i.norm(dim=-1, keepdim=True)
                    abnormal_f_det_i = abnormal_f_det_i /abnormal_f_det_i.norm(dim=-1, keepdim=True)
                    abnormal_f_seg_i = abnormal_f_seg_i /abnormal_f_seg_i.norm(dim=-1, keepdim=True)

                    #####################################
                    # Dual branch on detection features #
                    #####################################
                    #  text features = [768,2]: [:,0] -> t_n , [:,1] -> t_ab
                    #   ---> output: S_n,i = (O_n,i * t_n) - (O_n.i * t_ab)
                    sim_det_normal = self.dual_contrast(normal_f_det_i, text_features[:, 0], text_features[:, 1])

                    # ---> output: S_ab,i = (O_ab,i * t_ab) - (O_ab.i * t_n)
                    sim_det_abnormal = self.dual_contrast(abnormal_f_det_i, text_features[:, 1], text_features[:, 0])

                    # ---> output: S_i = [S_n,i, S_ab,i]
                    det_scores_cur = torch.cat([sim_det_normal, sim_det_abnormal], dim=-1)
                    det_scores.append(det_scores_cur)

                    ##########################################
                    #  Dual branch on Segmentation features  #
                    ##########################################
                    # normality branch
                    sim_seg_normal = self.dual_contrast(normal_f_seg_i, text_features[:,0], text_features[:,1])

                    # abnormality branch
                    sim_seg_abnormal = self.dual_contrast(abnormal_f_seg_i, text_features[:,1], text_features[:,0])
                    seg_scores_cur = torch.cat([sim_seg_normal, sim_seg_abnormal], dim=-1)  # shape: [B, 2(channels), img_size, img_size]

                    seg_scores.append(seg_scores_cur)



        x = x.permute(1, 0, 2)

        pooled, tokens = self.image_encoder._global_pool(x)
        pooled = self.image_encoder.ln_post(pooled)

        if self.image_encoder.proj is not None:
            pooled = pooled @ self.image_encoder.proj

        if is_video:
            pooled = pooled.view(batch_size, num_frames, -1).mean(dim=1)

        return pooled, det_scores, seg_scores


    def dual_contrast(self, features, same_text, opposite_text):
        same_view = (features @ same_text.unsqueeze(-1)) #[batch, 289,1]
        cross_view = ( features @ opposite_text.unsqueeze(-1)) #[batch, 289,1]

        return self.contrast(same_view, cross_view)
