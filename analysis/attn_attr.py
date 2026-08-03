"""
Self-Attention Attribution (ATTATTR) over the single-stream encoder stack.

Applies Integrated Gradients to the attention matrix, adapted from Hao et al. 2021
(https://github.com/YRdddream/attattr). This method operates on an already-loaded
model in evaluation mode and processes only a single fixed input at a time
(batch size of 1).

Port of the multi-stream version (branch hier-procgen-attattr), where the four blocks of
the attribution matrix come from different modules: the cross-attention fusion layer
(state<->action) and the two pre-fusion unimodal encoders (state->state, action->action).
Here there is a single homogeneous self-attention stack, so all four blocks are parity
slices of ONE layer's attention: tokens interleave as s_0 a_0 s_1 a_1..., states at even
positions and actions at odd ones.

To keep the two architectures comparable the blocks are read at the same depth from the
decoder as their multi-stream counterparts: cross blocks from the last encoder layer
(fusion sits one layer from the decoder there), same-stream blocks from the layer behind
it (where the unimodal encoders sit there). That also keeps the tree's depth ceiling the
same, since each edge list is one pass of the Appear/Fixed cascade.
"""
import numpy as np
import torch


def attn_attr_layer(agent, tar_layer: int, action_idx: int, m: int = 20):
    """
    Returns attr: [n_head, L, L], the attribution map of one `encoder_blocks` layer with
    respect to the current action logit (L = n_s + n_a, the interleaved real-token prefix).

    Requires the agent's observation and action buffers to be already populated.
    """
    agent.eval()  # critical: dropout must be identity

    n_s = len(agent._obs_buffer)
    assert n_s > 0, "attn_attr_layer() requires a non-empty obs buffer."

    with torch.no_grad():
        _, att = agent._forward_for_attr(tar_layer, capture_att=True)
    att = att.detach()

    grad_accum = torch.zeros_like(att)

    for k in range(1, m + 1):
        alpha = k / m
        tmp_att = (alpha * att).clone().requires_grad_(True)

        pred_a, _ = agent._forward_for_attr(tar_layer, tmp_att=tmp_att)
        target_logit = pred_a[n_s - 1, action_idx]

        agent.mdp.zero_grad(set_to_none=True)
        target_logit.backward()

        grad_accum += tmp_att.grad

    attr = att * (grad_accum / m)
    return attr.squeeze(0)


def split_by_parity(attr, n_s: int, n_a: int):
    """
    Splits a full [n_head, L, L] attribution map into the four blocks the multi-stream
    figures expect: (attr_s, attr_a, attr_enc_s, attr_enc_a) with shapes
    [n_head, n_s, n_a], [n_head, n_a, n_s], [n_head, n_s, n_s], [n_head, n_a, n_a].

    Rows are queries and columns keys, so attr_s is state -> action and attr_a is
    action -> state, matching the fusion naming on the multi-stream side.
    """
    assert n_a == n_s - 1, (
        f"expected n_a == n_s - 1 from the interleaved eval buffers, got n_s={n_s}, "
        f"n_a={n_a}; the parity split does not hold otherwise"
    )
    L = n_s + n_a
    assert attr.shape[-2:] == (L, L), (
        f"attribution map is {tuple(attr.shape)}, expected [..., {L}, {L}] for "
        f"n_s={n_s}, n_a={n_a}"
    )

    attr_s = attr[:, 0::2, 1::2]
    attr_a = attr[:, 1::2, 0::2]
    attr_enc_s = attr[:, 0::2, 0::2]
    attr_enc_a = attr[:, 1::2, 1::2]

    assert attr_s.shape[1:] == (n_s, n_a), tuple(attr_s.shape)
    assert attr_a.shape[1:] == (n_a, n_s), tuple(attr_a.shape)
    assert attr_enc_s.shape[1:] == (n_s, n_s), tuple(attr_enc_s.shape)
    assert attr_enc_a.shape[1:] == (n_a, n_a), tuple(attr_enc_a.shape)
    return attr_s, attr_a, attr_enc_s, attr_enc_a


def token_timesteps(t_target: int, n_s: int, n_a: int):
    """
    Maps attribution row/column indices to real environment timesteps. Returns
    (ts_state, ts_action); state token i and action token j both sit at
    `t_target - (n_s - 1) + index`, so index n_s-1 is the frame being decided on.

    Same mapping as the multi-stream version: the parity slice index of the interleaved
    sequence is the per-stream index (full position 2i or 2i+1 -> stream index i).
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
    attr_s: torch.Tensor,   # [n_head, T_s, T_a] -- state -> action, last encoder layer
    attr_a: torch.Tensor,   # [n_head, T_a, T_s] -- action -> state, last encoder layer
    tau: float = 0.4,       # edge threshold (cross-stream phase)
    label_fn=None,          # optional: label_fn(stream: 's'|'a', kept_idx: int) -> str
                            # default: f"{stream}_{kept_idx}"
    attr_enc_s=None,        # optional [n_head, T_s, T_s] -- state -> state, layer behind
    attr_enc_a=None,        # optional [n_head, T_a, T_a] -- action -> action, layer behind
    tau_enc=None,           # edge threshold for the same-stream phase; defaults to `tau`
    exclude_self_loops=True,  # drop i==j edges in the same-stream phase
):
    """
    Constructs an attribution tree adapted from Hao et al. 2021, Algorithm 1.
    Returns (V, E, diagnostics): V is a list of node labels, E is a list of
    (src, dst, kind) with kind in {'real', 'real_encoder', 'terminal'}.

    The 'TARGET' node is virtual (not a real model token) and represents
    the target prediction logit.

    Phase 1 (cross-stream, always runs) gives depth 1: root + direct state<->action
    neighbors. Phase 2 (optional, `attr_enc_s`/`attr_enc_a`) extends any node that
    survived phase 1 into ITS OWN same-stream neighbors, one layer further from the
    target, giving depth 2. This mirrors the paper's per-layer pass from the target
    backward: process the layer closest to the target first, then expand only the nodes
    that already survived using the next layer back -- never a fresh top-node search.

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

    # 3. Cross-stream phase: build downward
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

    # 3b. Same-stream phase (optional): expand nodes that survived the cross-stream phase
    #     into their own same-stream neighbors, one layer further from the target.
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
