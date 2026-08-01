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
    tau: float = 0.4,       # edge threshold
    label_fn=None,          # optional: label_fn(stream: 's'|'a', kept_idx: int) -> str
                            # default: f"{stream}_{kept_idx}"
):
    """
    Constructs an attribution tree adapted from Hao et al. 2021. 
    Returns V (list of node labels) and E (list of (src, dst, kind)), 
    where kind is 'real' or 'terminal' for differentiated rendering.

    The 'TARGET' node is virtual (not a real model token) and represents 
    the target prediction logit.

    Since this operates over a single layer, the resulting tree has 
    depth 1 by construction.
    """
    if label_fn is None:
        label_fn = lambda stream, idx: f"{stream}_{idx}"

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

    # 3. Build downward
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

    # 4. Virtual terminal: TARGET connects to everything left in the tree
    #    (automatic connection to all relevant nodes, not score-based)
    target = ('TARGET', 0)
    V.append(target)
    for node in V:
        if node == target:
            continue
        if state.get(node) in ('Appear', 'Fixed'):
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
    }
    return V_labeled, E_labeled, diagnostics
