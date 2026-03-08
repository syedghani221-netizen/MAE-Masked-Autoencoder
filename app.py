import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import math
import os
import gdown

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="MAE Image Reconstruction",
    page_icon="MAE",
    layout="wide"
)

# ── Config ───────────────────────────────────────────────────────
class CFG:
    image_size  = 224
    patch_size  = 16
    num_patches = 196
    mask_ratio  = 0.75
    enc_dim     = 768
    enc_layers  = 12
    enc_heads   = 12
    dec_dim     = 384
    dec_layers  = 12
    dec_heads   = 6
    device      = torch.device('cpu')

# ── Google Drive Model File ID ───────────────────────────────────
GDRIVE_FILE_ID = "1Kjd3Kd3eKFfDlpV1DmwANm-XGEEcbmry"
MODEL_PATH     = "mae_best.pth"

# ── Building Blocks ──────────────────────────────────────────────
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=True)
        self.proj      = nn.Linear(dim, dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = MLP(dim, mlp_ratio, dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Patchify / Unpatchify / Masking ──────────────────────────────
def patchify(imgs, patch_size=16):
    B, C, H, W = imgs.shape
    p = patch_size
    h = H // p
    w = W // p
    x = imgs.reshape(B, C, h, p, w, p)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, h * w, C * p * p)
    return x


def unpatchify(patches, patch_size=16, img_size=224):
    B, N, D = patches.shape
    p = patch_size
    C = 3
    h = w = img_size // p
    x = patches.reshape(B, h, w, C, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, C, img_size, img_size)
    return x


def random_masking(x, mask_ratio=0.75):
    B, N, D     = x.shape
    keep_n      = int(N * (1 - mask_ratio))
    noise       = torch.rand(B, N, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep    = ids_shuffle[:, :keep_n]
    x_visible   = torch.gather(
        x, 1,
        ids_keep.unsqueeze(-1).expand(-1, -1, D))
    mask        = torch.ones(B, N, device=x.device)
    mask[:, :keep_n] = 0
    mask        = torch.gather(mask, 1, ids_restore)
    return x_visible, mask, ids_restore


# ── Encoder ──────────────────────────────────────────────────────
class MAEEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16,
                 dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim        = 3 * patch_size * patch_size
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, dim),
            requires_grad=False)
        self._init_pos_embed(dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self._init_weights()

    def _init_pos_embed(self, embed_dim):
        grid_size = int(self.num_patches ** 0.5)
        pos_embed = self._get_sincos_pos_embed(embed_dim, grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    @staticmethod
    def _get_sincos_pos_embed(embed_dim, grid_size):
        assert embed_dim % 4 == 0
        half_dim = embed_dim // 4
        omega    = np.arange(half_dim, dtype=np.float32) / half_dim
        omega    = 1.0 / (10000 ** omega)
        grid_h   = np.arange(grid_size, dtype=np.float32)
        grid_w   = np.arange(grid_size, dtype=np.float32)
        grid_h, grid_w = np.meshgrid(grid_h, grid_w, indexing='ij')
        grid_h   = grid_h.reshape(-1)
        grid_w   = grid_w.reshape(-1)
        emb_h    = grid_h[:, None] * omega[None, :]
        emb_w    = grid_w[:, None] * omega[None, :]
        emb      = np.concatenate([
            np.sin(emb_h), np.cos(emb_h),
            np.sin(emb_w), np.cos(emb_w),
        ], axis=1)
        cls_emb  = np.zeros((1, embed_dim), dtype=np.float32)
        return np.concatenate([cls_emb, emb], axis=0)

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, imgs):
        patches  = patchify(imgs, self.patch_size)
        x        = self.patch_embed(patches)
        x        = x + self.pos_embed[:, 1:, :]
        x_vis, mask, ids_restore = random_masking(x, CFG.mask_ratio)
        cls      = self.cls_token.expand(x_vis.shape[0], -1, -1)
        cls      = cls + self.pos_embed[:, :1, :]
        x_vis    = torch.cat([cls, x_vis], dim=1)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        x_vis    = self.norm(x_vis)
        return x_vis, mask, ids_restore


# ── Decoder ──────────────────────────────────────────────────────
class MAEDecoder(nn.Module):
    def __init__(self, num_patches=196, patch_size=16,
                 enc_dim=768, dec_dim=384, depth=12, num_heads=6):
        super().__init__()
        patch_dim       = 3 * patch_size * patch_size
        self.proj       = nn.Linear(enc_dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.pos_embed  = nn.Parameter(
            torch.zeros(1, num_patches + 1, dec_dim),
            requires_grad=False)
        self._init_pos_embed(dec_dim, num_patches)
        self.blocks = nn.ModuleList([
            TransformerBlock(dec_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, patch_dim)
        self._init_weights()

    def _init_pos_embed(self, embed_dim, num_patches):
        assert embed_dim % 4 == 0
        grid_size = int(num_patches ** 0.5)
        pos_embed = MAEEncoder._get_sincos_pos_embed(embed_dim, grid_size)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, enc_tokens, ids_restore):
        B           = enc_tokens.shape[0]
        N           = ids_restore.shape[1]
        x           = self.proj(enc_tokens)
        num_vis     = x.shape[1] - 1
        num_mask    = N - num_vis
        mask_tokens = self.mask_token.expand(B, num_mask, -1)
        x_no_cls    = x[:, 1:, :]
        x_          = torch.cat([x_no_cls, mask_tokens], dim=1)
        x_          = torch.gather(
            x_, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, x_.shape[-1]))
        x           = torch.cat([x[:, :1, :], x_], dim=1)
        x           = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x           = self.norm(x)
        x           = self.pred(x[:, 1:, :])
        return x


# ── Full MAE ─────────────────────────────────────────────────────
class MAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MAEEncoder(
            img_size=CFG.image_size,
            patch_size=CFG.patch_size,
            dim=CFG.enc_dim,
            depth=CFG.enc_layers,
            num_heads=CFG.enc_heads)
        self.decoder = MAEDecoder(
            num_patches=CFG.num_patches,
            patch_size=CFG.patch_size,
            enc_dim=CFG.enc_dim,
            dec_dim=CFG.dec_dim,
            depth=CFG.dec_layers,
            num_heads=CFG.dec_heads)

    def forward(self, imgs):
        enc_tokens, mask, ids_restore = self.encoder(imgs)
        pred = self.decoder(enc_tokens, ids_restore)
        return pred, mask


# ── Load Model (cached so only loads once) ───────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive... Please wait (411 MB)"):
            url = "https://drive.google.com/uc?id=" + GDRIVE_FILE_ID
            gdown.download(url, MODEL_PATH, quiet=False)
    m    = MAE()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    m.load_state_dict(ckpt)
    m.eval()
    return m


# ── Denormalize ──────────────────────────────────────────────────
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

def denormalize(tensor):
    t = tensor.cpu().clone().float()
    t = t * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    t = t.clamp(0, 1).permute(1, 2, 0).numpy()
    return t


# ── Inference ────────────────────────────────────────────────────
def reconstruct(model, pil_img, mask_ratio):
    transform = transforms.Compose([
        transforms.Resize((CFG.image_size, CFG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    img_tensor = transform(pil_img).unsqueeze(0)

    with torch.no_grad():
        patches = patchify(img_tensor, CFG.patch_size)
        x       = model.encoder.patch_embed(patches)
        x       = x + model.encoder.pos_embed[:, 1:, :]

        x_vis, mask, ids_restore = random_masking(x, mask_ratio)

        cls   = model.encoder.cls_token.expand(1, -1, -1)
        cls   = cls + model.encoder.pos_embed[:, :1, :]
        x_vis = torch.cat([cls, x_vis], dim=1)

        for blk in model.encoder.blocks:
            x_vis = blk(x_vis)
        x_vis = model.encoder.norm(x_vis)

        pred  = model.decoder(x_vis, ids_restore)
        pred  = pred.float()

    orig_patches   = patchify(img_tensor, CFG.patch_size)
    mean_          = orig_patches.mean(dim=-1, keepdim=True)
    var_           = orig_patches.var(dim=-1, keepdim=True)
    pred_denorm    = pred * (var_ + 1e-6).sqrt() + mean_

    masked_patches = orig_patches.clone()
    masked_patches[mask.bool()] = 0.5

    orig_img   = unpatchify(orig_patches,   CFG.patch_size, CFG.image_size)[0]
    recon_img  = unpatchify(pred_denorm,    CFG.patch_size, CFG.image_size)[0]
    masked_img = unpatchify(masked_patches, CFG.patch_size, CFG.image_size)[0]

    def to_pil(t):
        arr = denormalize(t)
        arr = (arr * 255).astype(np.uint8)
        return Image.fromarray(arr)

    return to_pil(masked_img), to_pil(recon_img), to_pil(orig_img)


# ── Streamlit UI ─────────────────────────────────────────────────
st.title("Masked Autoencoder (MAE) - Image Reconstruction")
st.markdown(
    "This app uses a self-supervised Masked Autoencoder trained on TinyImageNet. "
    "Upload any image, choose how much to mask, and see the MAE reconstruct it!"
)
st.markdown("---")

# Load model
model = load_model()
st.success("Model loaded and ready!")

st.markdown("---")

# Input section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    uploaded = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        st.image(uploaded, caption="Uploaded Image", use_column_width=True)

with col2:
    st.subheader("Settings")
    mask_ratio = st.slider(
        "Masking Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.75,
        step=0.05,
        help="0.75 means 75% of patches are masked"
    )
    st.info(
        "Masking Ratio: " + str(int(mask_ratio * 100)) + "% masked | " +
        str(int((1 - mask_ratio) * 100)) + "% visible"
    )
    btn = st.button("Reconstruct Image", type="primary", use_container_width=True)

st.markdown("---")

# Output section
if btn and uploaded is not None:
    pil_img = Image.open(uploaded).convert("RGB")

    with st.spinner("Reconstructing image..."):
        masked, recon, original = reconstruct(model, pil_img, mask_ratio)

    st.subheader("Results")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Masked Input**")
        st.caption(str(int(mask_ratio * 100)) + "% of patches removed")
        st.image(masked, use_column_width=True)

    with c2:
        st.markdown("**Reconstruction**")
        st.caption("MAE predicted missing patches")
        st.image(recon, use_column_width=True)

    with c3:
        st.markdown("**Ground Truth**")
        st.caption("Original image")
        st.image(original, use_column_width=True)

elif btn and uploaded is None:
    st.warning("Please upload an image first!")

# Footer
st.markdown("---")
st.markdown(
    "Built with PyTorch | "
    "Trained on TinyImageNet | "
    "Course: Generative AI (AI4009) | "
    "NUCES Spring 2026"
)
