"""
param_counter.py  —  Paramaters count Verification
==========================================================
Reference values:
    MaskDP paper baseline : 4,094,487
    RLModal current       : 6,465,303
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
}

def make_cfg(**kwargs):
    defaults = dict(
        traj_length=64, traj_lengths=None, jitter_strategy="mix50",
        n_embd=256, n_head=4, n_enc_layer=2, n_dec_layer=2,
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

    print(f"\n{chr(61)*68}")
    print(f"  {label}")
    print(f"  {chr(45)*66}")
    print(f"  {'Module':<50} {'Params':>10}  {'%':>5}")
    print(f"  {chr(45)*66}")
    for name, count in child_counts.items():
        print(f"  {name:<50} {count:>10,}  {100*count/total:>4.1f}%")
    for name, p in loose.items():
        print(f"  {name} [Parameter]{'':>29} {p.numel():>10,}  {100*p.numel()/total:>4.1f}%")
    print(f"  {chr(45)*66}")
    print(f"  {'TOTAL':<50} {total:>10,}")
    print(f"  {chr(45)*66}")
    if ref_key and ref_key in LOG_REFERENCE:
        ref = LOG_REFERENCE[ref_key]
        diff = total - ref
        mark = "EXACT MATCH" if diff == 0 else f"DIFF = {diff:+,}"
        print(f"  Ref log [{ref_key}]: {ref:,}  =>  {mark}")
    return total


def lin(i,o): return i*o+o
def ln(d):    return 2*d
def block(d, r=4):
    return 4*lin(d,d) + ln(d) + lin(d,r*d) + lin(r*d,d) + ln(d)
def coattn(d, r=4):
    stream = ln(d) + 4*lin(d,d) + ln(d) + lin(d,r*d) + lin(r*d,d)
    return 2*stream

def reduction2_estimate():
    D_enc, D_full = 180, 256
    n_enc, n_dec, n_fuse = 2, 2, 1
    parts = {
        f"state_embed ({OBS_DIM}->{D_enc})":      lin(OBS_DIM, D_enc),
        f"action_embed ({ACTION_DIM}->{D_enc})":  lin(ACTION_DIM, D_enc),
        f"state_enc ({n_enc}xBlock({D_enc}))":    n_enc*block(D_enc),
        f"action_enc ({n_enc}xBlock({D_enc}))":   n_enc*block(D_enc),
        f"enc_norms (2xLN({D_enc}))":             2*ln(D_enc),
        f"proj_s ({D_enc}->{D_full})":            lin(D_enc, D_full),
        f"proj_a ({D_enc}->{D_full})":            lin(D_enc, D_full),
        f"fusion_type_embed Emb(2,{D_full})":     2*D_full,
        f"fusion ({n_fuse}xCoAttn({D_full}))":    n_fuse*coattn(D_full),
        f"fusion_norm LN({D_full})":              ln(D_full),
        "mask_token":                              D_full,
        f"dec_embeds (2xLin({D_full}))":          2*lin(D_full, D_full),
        f"dec_blocks ({n_dec}xBlock({D_full}))":  n_dec*block(D_full),
        "state_head":                             ln(D_full)+lin(D_full, OBS_DIM),
        "action_head":                            ln(D_full)+lin(D_full, ACTION_DIM),
    }
    total = sum(parts.values())
    print(f"\n{chr(61)*68}")
    print(f"  Reduction2 [enc={D_enc}->proj->{D_full}]  ANALITIC ESTIMATION")
    print(f"  {chr(45)*66}")
    print(f"  {'Module':<50} {'Params':>10}  {'%':>5}")
    print(f"  {chr(45)*66}")
    for name, count in parts.items():
        print(f"  {name:<50} {count:>10,}  {100*count/total:>4.1f}%")
    print(f"  {chr(45)*66}")
    print(f"  {'TOTAL (estimated)':<50} {total:>10,}")
    ref = LOG_REFERENCE["MaskDP_paper"]
    print(f"  vs MaskDP paper (4,094,487): {total-ref:+,}")
    return total


configs = [
    ("RLModal_current  [n_embd=256, n_head=4, cross, n_fuse=1]",
     "RLModal_current", {}),
    ("MaskDP_approx  [fusion=none, n_fuse=0, no type_embed]",
     "MaskDP_paper",
     dict(fusion_type="none", n_fuse_layer=0, use_fusion_type_embed=False)),
    ("Reduction1  [n_embd=192, n_head=4 -- all transformers]",
     None, dict(n_embd=192, n_head=4)),
    ("Reduction3  [n_embd=256, n_head=2 -- all blocks]",
     None, dict(n_embd=256, n_head=2)),
    ("Reduction4  [n_embd=255, n_head=3 -- all blocks]",
     None, dict(n_embd=255, n_head=3)),
]

print("\n" + "="*68)
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
        print(f"\n[ERROR] {label}: {e}")
        summary.append((label, None, None))

n_r2 = reduction2_estimate()
summary.append(("Reduction2 [enc=180->proj->256, ANALITIC]", n_r2, None))

base  = LOG_REFERENCE["RLModal_current"]
paper = LOG_REFERENCE["MaskDP_paper"]

print("\n\n" + "="*68)
print("  SUMMARY")
print("="*68)
print(f"  {'Config':<42} {'Params':>10}  {'Dvs paper':>10}  {'Dvs RLMod':>10}")
print(f"  {chr(45)*42} {chr(45)*10}  {chr(45)*10}  {chr(45)*10}")
for lbl, n, ref in summary:
    ns = f"{n:>10,}" if n is not None else f"{'ERROR':>10}"
    dp = f"{n-paper:>+10,}" if n is not None else chr(45)*10
    dr = f"{n-base:>+10,}"  if n is not None else chr(45)*10
    print(f"  {lbl[:42]:<42} {ns}  {dp}  {dr}")
print(f"\n  MaskDP paper  = {paper:,}")
print(f"  RLModal curr  = {base:,}  ({base-paper:+,} overhead multimodal)")
print()