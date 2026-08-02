"""
Self-Attention Attribution (ATTATTR) over the cross-attention layer.

Applies Integrated Gradients to the attention matrix, adapted from Hao et al. 2021 
(https://github.com/YRdddream/attattr). This method operates on an already-loaded 
model in evaluation mode and processes only a single fixed input at a time 
(batch size of 1).
"""
import numpy as np
import torch


def attn_attr_fusion(agent, tar_layer: int, action_idx: int, m: int = 20):
    """
    Returns (attr_s, attr_a): attribution maps [n_head, T_s, T_a] and [n_head, T_a, T_s]
    for both state and action cross-attention.

    Calculated at the target fusion layer with respect to the current action logit. 
    Requires the agent's observation and action buffers to be already populated.
    """
    agent.eval()  # critical: dropout must be identity

    n_s = len(agent._obs_buffer)
    assert n_s > 0, "attn_attr_fusion() requires a non-empty obs buffer."

    with torch.no_grad():
        _, att_s, att_a = agent._forward_for_attr(tar_layer, capture_att=True)
    att_s = att_s.detach()
    att_a = att_a.detach()

    grad_accum_s = torch.zeros_like(att_s)
    grad_accum_a = torch.zeros_like(att_a)

    for k in range(1, m + 1):
        alpha = k / m
        tmp_att_s = (alpha * att_s).clone().requires_grad_(True)
        tmp_att_a = (alpha * att_a).clone().requires_grad_(True)

        pred_a, _, _ = agent._forward_for_attr(tar_layer, tmp_att_s=tmp_att_s, tmp_att_a=tmp_att_a)
        target_logit = pred_a[n_s - 1, action_idx]

        agent.mdp.zero_grad(set_to_none=True)
        target_logit.backward()

        grad_accum_s += tmp_att_s.grad
        grad_accum_a += tmp_att_a.grad

    attr_s = att_s * (grad_accum_s / m)
    attr_a = att_a * (grad_accum_a / m)
    return attr_s.squeeze(0), attr_a.squeeze(0)


def attn_attr_encoder(agent, stream: str, tar_layer: int, action_idx: int, m: int = 20):
    """
    Same IG mechanism as `attn_attr_fusion`, but targets a self-attention layer of
    `state_encoder_blocks` ('s') or `action_encoder_blocks` ('a') -- the pre-fusion
    unimodal streams -- instead of the fusion cross-attention. Fusion runs with its
    real (unhooked) attention throughout.

    Returns attr_enc: [n_head, T, T] (T = n_s for 's', n_a for 'a').
    """
    agent.eval()

    n_s = len(agent._obs_buffer)
    assert n_s > 0, "attn_attr_encoder() requires a non-empty obs buffer."

    with torch.no_grad():
        _, att = agent._forward_for_attr_encoder(stream, tar_layer, capture_att=True)
    att = att.detach()

    grad_accum = torch.zeros_like(att)

    for k in range(1, m + 1):
        alpha = k / m
        tmp_att = (alpha * att).clone().requires_grad_(True)

        pred_a, _ = agent._forward_for_attr_encoder(stream, tar_layer, tmp_att=tmp_att)
        target_logit = pred_a[n_s - 1, action_idx]

        agent.mdp.zero_grad(set_to_none=True)
        target_logit.backward()

        grad_accum += tmp_att.grad

    attr = att * (grad_accum / m)
    return attr.squeeze(0)


def token_timesteps(t_target: int, n_s: int, n_a: int):
    """
    Maps attribution row/column indices to real environment timesteps. Returns
    (ts_state, ts_action); state token i and action token j both sit at
    `t_target - (n_s - 1) + index`, so index n_s-1 is the frame being decided on.
    """
    assert n_a == n_s - 1, (
        f"expected n_a == n_s - 1 from the interleaved eval buffers, got n_s={n_s}, "
        f"n_a={n_a}; the timestep mapping does not hold otherwise"
    )
    start = t_target - (n_s - 1)
    return np.arange(start, start + n_s), np.arange(start, start + n_a)


def attribution_range_diagnostics(attr_s: torch.Tensor, attr_a: torch.Tensor):
    """
    Range of the head-summed attribution matrices, used to sanity-check a threshold `tau` 
    before trusting it. 

    The threshold (e.g., 0.4) might need empirical adjustment if the resulting tree is 
    empty or too large for this specific architecture.
    """
    a_s = attr_s.sum(dim=0)
    a_a = attr_a.sum(dim=0)
    flat = torch.cat([a_s.flatten(), a_a.flatten()])
    return {
        "min": flat.min().item(),
        "max": flat.max().item(),
        "mean": flat.mean().item(),
        "p50": flat.median().item(),
        "p90": flat.quantile(0.90).item(),
        "p99": flat.quantile(0.99).item(),
    }


def build_attribution_tree(
    attr_s: torch.Tensor,   # [n_head, T_s, T_a] -- attribution matrix for state
    attr_a: torch.Tensor,   # [n_head, T_a, T_s] -- attribution matrix for action
    tau: float = 0.4,       # edge threshold (fusion layer)
    label_fn=None,          # optional: label_fn(stream: 's'|'a', kept_idx: int) -> str
                            # default: f"{stream}_{kept_idx}"
    attr_enc_s=None,        # optional [n_head, T_s, T_s] -- state_encoder_blocks self-attn
    attr_enc_a=None,        # optional [n_head, T_a, T_a] -- action_encoder_blocks self-attn
    tau_enc=None,           # edge threshold for the encoder phase; defaults to `tau`
    exclude_self_loops=True,  # drop i==j edges in the encoder phase (self-attention only)
):
    """
    Constructs an attribution tree adapted from Hao et al. 2021, Algorithm 1.
    Returns (V, E, diagnostics): V is a list of node labels, E is a list of
    (src, dst, kind) with kind in {'real', 'real_encoder', 'terminal'}.

    The 'TARGET' node is virtual (not a real model token) and represents
    the target prediction logit.

    Phase 1 (fusion, always runs) gives depth 1: root + direct cross-attention
    neighbors. Phase 2 (optional, `attr_enc_s`/`attr_enc_a`) extends any node that
    survived phase 1 into ITS OWN pre-fusion self-attention neighbors (same
    stream, same token index space -- state_encoder/action_encoder tokens map
    1:1 onto the 's'/'a' tokens fusion already saw), giving depth 2. This mirrors
    the paper's per-layer pass from the target backward: process the layer
    closest to the target first, then expand only the nodes that already
    survived using the next layer back -- never a fresh top-node search.

    With `attr_enc_s=attr_enc_a=None` (default), phase 2 is skipped and the tree
    stays flat (depth 1).
    """
    if label_fn is None:
        label_fn = lambda stream, idx: f"{stream}_{idx}"
    if tau_enc is None:
        tau_enc = tau

    a_s = attr_s.sum(dim=0)   # [T_s, T_a], summed over heads
    a_a = attr_a.sum(dim=0)   # [T_a, T_s]
    T_s, T_a = a_s.shape

    def node_id(stream, idx):
        return (stream, idx)  # internal id; label_fn applied only when returning

    # 1. Candidate token state: 'NotAppear' | 'Appear' | 'Fixed'
    state = {}
    for i in range(T_s):
        state[node_id('s', i)] = 'NotAppear'
    for i in range(T_a):
        state[node_id('a', i)] = 'NotAppear'

    # 2. TopNode: highest Sum_j a[i,j] (attribution mass the token "sends")
    attr_all = {}
    for i in range(T_s):
        attr_all[node_id('s', i)] = a_s[i].sum().item()
    for i in range(T_a):
        attr_all[node_id('a', i)] = a_a[i].sum().item()
    top_node = max(attr_all, key=attr_all.get)

    V = [top_node]
    state[top_node] = 'Appear'
    E = []

    # 3. Fusion layer: build downward
    edges_s = [(('s', i), ('a', j), a_s[i, j].item())
               for i in range(T_s) for j in range(T_a) if a_s[i, j].item() > tau]
    edges_a = [(('a', i), ('s', j), a_a[i, j].item())
               for i in range(T_a) for j in range(T_s) if a_a[i, j].item() > tau]

    for u, v, w in edges_s + edges_a:
        if u not in state or v not in state:
            continue
        if state[u] == 'Appear' and state[v] == 'NotAppear':
            E.append((u, v, 'real'))
            V.append(v)
            state[u] = 'Fixed'
            state[v] = 'Appear'
        elif state[u] == 'Fixed' and state[v] == 'NotAppear':
            E.append((u, v, 'real'))
            V.append(v)
            state[v] = 'Appear'

    # 3b. Encoder layer (optional): expand nodes that survived fusion into their
    #     own pre-fusion self-attention neighbors, one layer further from the target.
    n_nodes_before_encoder = len(V)
    if attr_enc_s is not None or attr_enc_a is not None:
        edges_enc = []
        if attr_enc_s is not None:
            a_enc_s = attr_enc_s.sum(dim=0)   # [T_s, T_s]
            assert a_enc_s.shape == (T_s, T_s), (
                f"attr_enc_s shape {tuple(a_enc_s.shape)} does not match T_s={T_s} from attr_s"
            )
            edges_enc += [(('s', i), ('s', j), a_enc_s[i, j].item())
                          for i in range(T_s) for j in range(T_s)
                          if (not exclude_self_loops or i != j) and a_enc_s[i, j].item() > tau_enc]
        if attr_enc_a is not None:
            a_enc_a = attr_enc_a.sum(dim=0)   # [T_a, T_a]
            assert a_enc_a.shape == (T_a, T_a), (
                f"attr_enc_a shape {tuple(a_enc_a.shape)} does not match T_a={T_a} from attr_a"
            )
            edges_enc += [(('a', i), ('a', j), a_enc_a[i, j].item())
                          for i in range(T_a) for j in range(T_a)
                          if (not exclude_self_loops or i != j) and a_enc_a[i, j].item() > tau_enc]

        for u, v, w in edges_enc:
            if u not in state or v not in state:
                continue
            if state[u] == 'Appear' and state[v] == 'NotAppear':
                E.append((u, v, 'real_encoder'))
                V.append(v)
                state[u] = 'Fixed'
                state[v] = 'Appear'
            elif state[u] == 'Fixed' and state[v] == 'NotAppear':
                E.append((u, v, 'real_encoder'))
                V.append(v)
                state[v] = 'Appear'
    n_nodes_from_encoder = len(V) - n_nodes_before_encoder

    # 4. Virtual terminal: TARGET attaches to the roots of the attribution
    #    structure -- the nodes no real edge points at, normally just the TopNode.
    #    Connecting it to every node instead is a fan of unscored edges that
    #    carries no information.
    target = ('TARGET', 0)
    has_parent = {dst for _, dst, _ in E}
    roots = [n for n in V if n not in has_parent and state.get(n) in ('Appear', 'Fixed')]
    V.append(target)
    for node in roots:
        E.append((target, node, 'terminal'))

    # 5. Apply labels only on return (does not affect tree construction)
    V_labeled = [
        'TARGET' if n == target else label_fn(n[0], n[1])
        for n in V
    ]
    E_labeled = [
        ('TARGET' if u == target else label_fn(u[0], u[1]),
         'TARGET' if v == target else label_fn(v[0], v[1]),
         kind)
        for u, v, kind in E
    ]

    n_orphans = sum(1 for k, v in state.items() if v == 'NotAppear')
    n_total = T_s + T_a
    diagnostics = {
        "n_nodes": len(V_labeled),
        "n_edges": len(E_labeled),
        "n_orphans": n_orphans,
        "n_candidate_tokens": n_total,
        "n_nodes_from_encoder": n_nodes_from_encoder,
    }
    return V_labeled, E_labeled, diagnostics
