"""
param_counter.py  —  Parameter count verification
==================================================
Reference values:
    MaskDP paper baseline : 4,094,487
    RLModal current       : 6,465,303  (enc=256, neck=256, dec=256)
    R2 (enc_180)          : 4,964,350  (enc=180, neck=256, dec=256)
"""

import sys, io
from contextlib import redirect_stdout
import torch
import torch.nn as nn
from omegaconf import OmegaConf

try:
    from agent.mdp import MaskedDPMultimodal
except ImportError as e:
    print(f"[ERROR] Could not import MaskedDPMultimodal: {e}")
    sys.exit(1)

OBS_DIM    = 24
ACTION_DIM = 6

LOG_REFERENCE = {
    "RLModal_current": 6_465_303,
    "MaskDP_paper":    4_094_487,
    "R2_enc180":       4_964_350,
}

def make_cfg(**kwargs):
    defaults = dict(
        traj_length=64, traj_lengths=None, jitter_strategy="mix50",
        n_embd=256, enc_n_embd=256, dec_n_embd=256,
        n_head=4, n_enc_layer=2, n_dec_layer=2,
        n_fuse_layer=1, use_fusion_type_embed=True, fusion_type="cross",
        modality_dropout=False, modality_dropout_prob=0.0,
        state_dropout_prob=0.0, action_dropout_prob=0.0,
        min_keep_states=1, min_keep_actions=0, mlp_ratio=4,
        use_early_fusion=False, use_adapter_mlp=False,
        adapter_mlp_ratio=2, adapter_mlp_layers=2,
        adapter_mlp_norm=True, adapter_mlp_residual=True,
        embd_pdrop=0, resid_pdrop=0, attn_pdrop=0,
        pe="fixed", loss="total", norm="l2",
        use_pixel_obs=False, pixel_obs_shape=[64, 64, 3], pad_value=1e9,
    )
    defaults.update(kwargs)
    return OmegaConf.create(defaults)


def param_breakdown(model, label, ref_key=None):
    child_ptrs = set()
    child_counts = {}
    for name, module in model.named_children():
        count = sum(p.numel() for p in module.parameters())
        child_counts[name] = count
        for p in module.parameters():
            child_ptrs.add(p.data_ptr())
    loose = {n: p for n, p in model.named_parameters(recurse=False)
             if p.data_ptr() not in child_ptrs}
    total = sum(p.numel() for p in model.parameters())

    print(f"{chr(61)*68}")
    print(f"  {label}")
    print(f"  {chr(45)*66}")
    print(f"  {chr(39)*0}{'Module':<50}{'Params':>10}  {'%':>5}")
    print(f"  {chr(45)*66}")
    for name, count in child_counts.items():
        print(f"  {name:<50} {count:>10,}  {100*count/total:>4.1f}%")
    for name, p in loose.items():
        print(f"  {name} [Parameter]{chr(39)*0:>29} {p.numel():>10,}  {100*p.numel()/total:>4.1f}%")
    print(f"  {chr(45)*66}")
    print(f"  {'TOTAL':<50} {total:>10,}")
    if ref_key and ref_key in LOG_REFERENCE:
        ref  = LOG_REFERENCE[ref_key]
        diff = total - ref
        mark = "EXACT MATCH" if diff == 0 else f"DIFF = {diff:+,}"
        print(f"  {chr(45)*66}")
        print(f"  Ref log [{ref_key}]: {ref:,}  =>  {mark}")
    return total


# ── Analytic helpers ─────────────────────────────────────────────────────────
def lin(i,o):  return i*o + o
def ln(d):     return 2*d
def blk(d, r=4):
    return 4*lin(d,d) + 2*ln(d) + lin(d,r*d) + lin(r*d,d)
def coattn(d, r=4):
    stream = ln(d) + 4*lin(d,d) + ln(d) + lin(d,r*d) + lin(r*d,d)
    return 2*stream

def analytic_estimate(De, Dn, Dd, label, n_enc=2, n_dec=2, n_fuse=1):
    """Full analytic param count for (enc_embd, neck_embd, dec_embd)."""
    parts = {
        f"state_embed ({OBS_DIM}->{De})": lin(OBS_DIM, De),
        f"action_embed ({ACTION_DIM}->{De})": lin(ACTION_DIM, De),
        f"state_enc ({n_enc}xBlk({De}))": n_enc*blk(De),
        f"action_enc ({n_enc}xBlk({De}))": n_enc*blk(De),
        f"enc_norms (2xLN({De}))": 2*ln(De),
        f"proj_s ({De}->{Dn})": lin(De, Dn) if De != Dn else 0,
        f"proj_a ({De}->{Dn})": lin(De, Dn) if De != Dn else 0,
        f"fusion_type_embed Emb(2,{Dn})": 2*Dn,
        f"fusion ({n_fuse}xCoAttn({Dn}))": n_fuse*coattn(Dn),
        f"fusion_norm LN({Dn})": ln(Dn),
        "mask_token": Dn,
        f"dec_embeds (2xLin({Dn},{Dd}))": 2*lin(Dn, Dd),
        f"dec_blocks ({n_dec}xBlk({Dd}))": n_dec*blk(Dd),
        f"state_head LN+Lin({Dd},{OBS_DIM})": ln(Dd)+lin(Dd, OBS_DIM),
        f"action_head LN+Lin({Dd},{ACTION_DIM})": ln(Dd)+lin(Dd, ACTION_DIM),
    }
    # Remove zero entries (identity proj when De==Dn)
    parts = {k: v for k, v in parts.items() if v > 0}
    total = sum(parts.values())

    print(f"{chr(61)*68}")
    print(f"  {label}  [ANALYTIC]")
    print(f"  enc={De} / neck={Dn} / dec={Dd}")
    print(f"  {chr(45)*66}")
    print(f"  {'Module':<50} {'Params':>10}  {'%':>5}")
    print(f"  {chr(45)*66}")
    for name, count in parts.items():
        print(f"  {name:<50} {count:>10,}  {100*count/total:>4.1f}%")
    print(f"  {chr(45)*66}")
    print(f"  {'TOTAL (analytic)':<50} {total:>10,}")
    ref_p = LOG_REFERENCE["MaskDP_paper"]
    ref_r = LOG_REFERENCE["RLModal_current"]
    print(f"  vs MaskDP paper  ({ref_p:,}): {total-ref_p:+,}")
    print(f"  vs RLModal curr  ({ref_r:,}): {total-ref_r:+,}")
    return total


# ── Configs to instantiate ────────────────────────────────────────────────────
configs = [
    # label, ref_key, cfg_kwargs
    ("Base [enc=256, neck=256, dec=256]",
     "RLModal_current", {}),
    ("R2_enc180 [enc=180, neck=256, dec=256]",
     "R2_enc180", dict(enc_n_embd=180)),
    ("OptionA [enc=180, neck=256, dec=180]",
     None, dict(enc_n_embd=180, dec_n_embd=180)),
    ("OptionB [enc=128, neck=256, dec=256]",
     None, dict(enc_n_embd=128)),
    ("MaskDP_approx [fusion=none, no type_embed]",
     "MaskDP_paper",
     dict(fusion_type="none", n_fuse_layer=0, use_fusion_type_embed=False)),
]

print("" + "="*68)
print("  PARAM COUNTER -- RLModal vs MaskDP")
print(f"  obs_dim={OBS_DIM}  action_dim={ACTION_DIM}")
print("="*68)

summary = []
for label, ref_key, kw in configs:
    cfg = make_cfg(**kw)
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            model = MaskedDPMultimodal(OBS_DIM, ACTION_DIM, cfg)
        n = param_breakdown(model, label, ref_key)
        summary.append((label, n, LOG_REFERENCE.get(ref_key)))
    except Exception as e:
        print(f"[ERROR] {label}: {e}")
        summary.append((label, None, None))

# Analytic estimates for sweep candidates
an_A  = analytic_estimate(180, 256, 180, "Option A")
an_B  = analytic_estimate(128, 256, 256, "Option B")
an_AB = analytic_estimate(128, 256, 180, "Option A+B (future)")
summary += [
    ("OptionA analytic [180/256/180]", an_A,  None),
    ("OptionB analytic [128/256/256]", an_B,  None),
    ("Future  analytic [128/256/180]", an_AB, None),
]

# ── Summary ──────────────────────────────────────────────────────────────────
base  = LOG_REFERENCE["RLModal_current"]
paper = LOG_REFERENCE["MaskDP_paper"]

print("" + "="*68)
print("  SUMMARY")
print("="*68)
print(f"  {'Config':<44} {'Params':>10}  {'vs paper':>10}  {'vs RLMod':>10}")
print(f"  {chr(45)*44} {chr(45)*10}  {chr(45)*10}  {chr(45)*10}")
for lbl, n, ref in summary:
    ns = f"{n:>10,}" if n is not None else f"{'ERROR':>10}"
    dp = f"{n-paper:>+10,}" if n is not None else chr(45)*10
    dr = f"{n-base:>+10,}"  if n is not None else chr(45)*10
    print(f"  {lbl[:44]:<44} {ns}  {dp}  {dr}")

print(f"MaskDP paper  = {paper:>10,}  (target)")
print(f"  RLModal base  = {base:>10,}  ({base-paper:+,} overhead)")
print(f"  R2 enc_180    = {LOG_REFERENCE[chr(82)+chr(50)+chr(95)+chr(101)+chr(110)+chr(99)+chr(49)+chr(56)+chr(48)]:>10,}  ({LOG_REFERENCE[chr(82)+chr(50)+chr(95)+chr(101)+chr(110)+chr(99)+chr(49)+chr(56)+chr(48)]-paper:+,} overhead)")
print()