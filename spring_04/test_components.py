import torch.nn as nn
import torch
import math
import sys

# allow import from the same dir
sys.path.insert(0, '.')
from model import (DHNNCELoss, SoftPatchContrastiveLoss, SegmentationLoss, ProbabilisticCrossAttention, PVLAdapter, SegmentationHead, HybridLoss)

DEVICE = torch.device('cpu')
B, D = 4, 768 # batch, CLIP dim
Tv, Tt = 197, 32 # ViT tokens (1 CLS + 196 patches), text tokens
Ds, Da = 256, 64 # PVL dims

def test_dhn_nce():
    # DHN-NCE loss
    loss_fn = DHNNCELoss(tau=0.6, beta1=0.15, beta2=0.15)
    img = torch.randn(B, D); img = img / img.norm(dim=-1, keepdim=True)
    txt = torch.randn(B, D); txt = txt / txt.norm(dim=-1, keepdim=True)
    loss = loss_fn(img, txt)
    assert loss.isfinite(), f"loss not finite: {loss}"
    print(f"DHN-NCE loss = {loss.item():.4f}  ")

def test_soft_contrastive():
    # soft patch contrastive loss
    loss_fn = SoftPatchContrastiveLoss(tau=0.2)
    patch = torch.randn(B, D); patch = patch / patch.norm(dim=-1, keepdim=True)
    text = torch.randn(B, D); text  = text  / text.norm( dim=-1, keepdim=True)
    loss = loss_fn(patch, text)
    assert loss.isfinite()
    print(f"soft contrastive loss = {loss.item():.4f}  ")

def test_seg_loss():
    # segmentation loss (Dice + BCE)
    loss_fn = SegmentationLoss()
    pred = torch.randn(B, 1, 224, 224)
    target = (torch.rand(B, 1, 224, 224) > 0.5).float()
    loss = loss_fn(pred, target)
    assert loss.isfinite()
    print(f"seg loss = {loss.item():.4f}  ")

def test_prob_cross_attn():
    # probabilistic cross attention
    attn = ProbabilisticCrossAttention(d_s=Ds, d_a=Da).to(DEVICE)
    Q = torch.randn(B, Tv, Ds)
    K = torch.randn(B, Tt, Ds)
    out_train = attn(Q, K, training=True)
    out_eval = attn(Q, K, training=False)
    assert out_train.shape == Q.shape, f"Shape mismatch: {out_train.shape}"
    assert out_eval.shape  == Q.shape
    print(f"output shape: {tuple(out_train.shape)}  ")

def test_pvl_adapter():
    # PVL adapter (bidirectional)
    adapter = PVLAdapter(d_v=D, d_t=D, d_s=Ds, d_a=Da).to(DEVICE)
    vis = torch.randn(B, Tv, D)
    txt = torch.randn(B, Tt, D)
    vis_out, txt_out = adapter(vis, txt, training=True)
    assert vis_out.shape == vis.shape, f"vis shape: {vis_out.shape}"
    assert txt_out.shape == txt.shape, f"txt shape: {txt_out.shape}"
    print(f"vis_out: {tuple(vis_out.shape)}  txt_out: {tuple(txt_out.shape)}  ")

def test_seg_head():
    # segmentation head
    head = SegmentationHead(d_model=D, patch_grid=14, n_upscale=2).to(DEVICE)
    patch = torch.randn(B, 196, D)   # 14×14 = 196 patches
    text = torch.randn(B, D)
    out = head(patch, text)
    # 2 upscales from 14 -> 56
    assert out.shape == (B, 1, 56, 56), f"Unexpected shape: {out.shape}"
    print(f"seg-head output: {tuple(out.shape)}  ")

def test_hybrid_loss():
    # hybrid loss (combined)
    loss_fn = HybridLoss()
    seg_logits = torch.randn(B, 1, 224, 224)
    masks = (torch.rand(B, 1, 224, 224) > 0.5).float()
    image_feat = torch.randn(B, D); image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
    text_feat = torch.randn(B, D); text_feat = text_feat / text_feat.norm( dim=-1, keepdim=True)
    avg_patch = torch.randn(B, D); avg_patch = avg_patch / avg_patch.norm( dim=-1, keepdim=True)

    total, breakdown = loss_fn(seg_logits, masks, image_feat, text_feat, avg_patch)
    assert total.isfinite()
    print(f"total={breakdown['total']:.4f}, seg={breakdown['seg']:.4f}, dhn={breakdown['dhn']:.4f}, scon={breakdown['scon']:.4f}")

def test_pvl_count():
    # PVL adapter param count
    adapter = PVLAdapter(d_v=D, d_t=D, d_s=Ds, d_a=Da)
    n = sum(p.numel() for p in adapter.parameters())
    print(f"params per PVL adapter: {n:,}")
    print(f"total for 12 vis + 12 txt adapters: {24*n:,}")

def test_gradient_flow():
    # gradient flow through PVL + SegHead
    adapter = PVLAdapter(d_v=D, d_t=D, d_s=Ds, d_a=Da)
    head = SegmentationHead(d_model=D, patch_grid=14, n_upscale=2)

    vis = torch.randn(B, Tv, D, requires_grad=False)
    txt = torch.randn(B, Tt, D, requires_grad=False)
    msk = (torch.rand(B, 1, 56, 56) > 0.5).float()

    vis_out, txt_out = adapter(vis, txt, training=True)
    logits = head(vis_out[:, 1:, :], txt_out[:, 0, :])  # skip CLS, take EOS
    loss = nn.functional.binary_cross_entropy_with_logits(logits, msk)
    loss.backward()

    grads_ok = all(p.grad is not None and p.grad.isfinite().all() for p in list(adapter.parameters()) + list(head.parameters()))
    assert grads_ok, "some gradients are none or NaN"
    print(f"all gradients finite (loss={loss.item():.4f})")

if __name__ == '__main__':
    tests = [test_dhn_nce, test_soft_contrastive, test_seg_loss, test_prob_cross_attn, test_pvl_adapter, test_seg_head, test_hybrid_loss, test_pvl_count, test_gradient_flow]

    passed = 0
    for fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"error: {e}")

    print(f"{passed}/{len(tests)} tests passed")