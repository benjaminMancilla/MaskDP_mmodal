import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
from pathlib import Path

import hydra
import numpy as np
import omegaconf
import torch

import utils
import wandb
from agent.mdp_bct import BCTEvalAgent
from analysis.perturbation import occlude, perturb_batch
from analysis.trajectory_recorder import (
    load_buffers,
    load_trajectory_states,
    record_reference_trajectory,
    save_trajectory_states,
    select_quantile_states,
)
from third_party.sarfa_saliency import computeSaliencyUsingSarfa

torch.backends.cudnn.benchmark = True


def parse_checkpoint_step(path):
    try:
        return int(Path(path).stem.split("_")[-1])
    except Exception:
        return -1


def _sarfa_triplet(a_hat, dict_before, dict_after):
    answer, dP, K, _QmaxAnswer, _gap_before, _gap_after = computeSaliencyUsingSarfa(
        a_hat, dict_before, dict_after
    )
    return answer, dP, K


def compute_state_saliency(agent, state, perturb_cfg, K_max, temporal_blur_sigma=None):
    """
    Computes spatial and temporal saliency for a single state. 
    Performs a single unperturbed forward pass (reused by both phases), then:

      - Spatial: Batched forward pass over the grid of perturbations on the 
        CURRENT frame. Returns a raw (H/d, W/d) map without interpolation.
      - Temporal: Uses the same saliency logic, but perturbs the ENTIRE frame 
        at each buffer position `j` using a global blur, rather than a spatial patch. 
        Returns a vector of length `n_s` indicating the importance of past timesteps 
        for the current action. It is padded with NaNs up to `K_max` to ensure 
        fixed shapes for stacking.

    `temporal_blur_sigma`: If None, reuses `perturb_cfg.blur_sigma` (same as spatial).
    """
    if temporal_blur_sigma is None:
        temporal_blur_sigma = perturb_cfg.blur_sigma

    load_buffers(agent, state["obs_buffer"], state["action_buffer"])

    with torch.no_grad():
        base_logits = agent.logits_for()  # (num_actions,) raw logits, unperturbed
    a_hat = int(torch.argmax(base_logits).item())
    dict_before = {a: float(v) for a, v in enumerate(base_logits.tolist())}

    # Spatial saliency on the current frame
    current_frame = agent._obs_buffer[-1]
    perturbed, centers = perturb_batch(
        current_frame,
        d=perturb_cfg.d,
        radius=perturb_cfg.radius,
        blur_sigma=perturb_cfg.blur_sigma,
        mode=perturb_cfg.mode,
    )
    with torch.no_grad():
        pert_logits = agent.logits_for(obs_override=perturbed, override_index=-1)  # (N, num_actions)

    rows = sorted({r for r, c in centers})
    cols = sorted({c for r, c in centers})
    grid_h, grid_w = len(rows), len(cols)

    answer_flat = np.zeros(len(centers), dtype=np.float32)
    dP_flat = np.zeros(len(centers), dtype=np.float32)
    K_flat = np.zeros(len(centers), dtype=np.float32)
    for i in range(len(centers)):
        dict_after = {a: float(v) for a, v in enumerate(pert_logits[i].tolist())}
        answer, dP, K = _sarfa_triplet(a_hat, dict_before, dict_after)
        answer_flat[i] = answer
        dP_flat[i] = dP
        K_flat[i] = K

    # Temporal saliency, global perturbation per position
    n_s = len(agent._obs_buffer)
    full_mask = np.ones(current_frame.shape[:2], dtype=np.float32)

    t_answer = np.full(K_max, np.nan, dtype=np.float32)
    t_dP = np.full(K_max, np.nan, dtype=np.float32)
    t_K = np.full(K_max, np.nan, dtype=np.float32)
    for j in range(n_s):
        blurred = occlude(agent._obs_buffer[j], full_mask, blur_sigma=temporal_blur_sigma)
        with torch.no_grad():
            logits_j = agent.logits_for(obs_override=blurred, override_index=j)
        dict_after = {a: float(v) for a, v in enumerate(logits_j.tolist())}
        answer, dP, K = _sarfa_triplet(a_hat, dict_before, dict_after)
        t_answer[j] = answer
        t_dP[j] = dP
        t_K[j] = K

    return {
        "a_hat": a_hat,
        "answer_map": answer_flat.reshape(grid_h, grid_w),
        "dP_map": dP_flat.reshape(grid_h, grid_w),
        "K_map": K_flat.reshape(grid_h, grid_w),
        "temporal_answer": t_answer,
        "temporal_dP": t_dP,
        "temporal_K": t_K,
        "n_s": n_s,
    }


@hydra.main(config_path=".", config_name="eval_sarfa")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    assert cfg.distribution_mode == "easy", (
        f"distribution_mode='{cfg.distribution_mode}' -- 'easy' is explicitly required. "
        "It is the mode used to train the shared frozen IMPALA encoder. "
        "'hard' shifts states out of distribution."
    )
    assert len(cfg.models) >= 2, "At least 2 models are required in `models` for comparison."
    assert cfg.phase in ("record", "evaluate", "both"), f"phase must be record|evaluate|both, got '{cfg.phase}'"
    assert cfg.get("output_dir", None), (
        "`output_dir` is required (no default derived from a single model's path) -- "
        "hier and uni snapshots live under different directories, so this run needs "
        "an explicit shared location both checkouts can read/write."
    )

    only_arch = set(cfg.only_arch) if cfg.get("only_arch", None) else None
    if cfg.phase != "record":
        assert len(cfg.levels) >= 1, (
            "`levels` cannot be empty. Set an EXPLICIT list of levels so "
            "all models visit the same levels."
        )

    obs_shape = (64, 64, 3)  # procgen; generalize if another env_name is used
    action_shape = (1,)

    agents = {}
    model_meta = {}
    for m in cfg.models:
        model_meta[m.name] = {
            "arch": str(m.arch),
            "seed": int(m.seed),
            "checkpoint_step": parse_checkpoint_step(m.path),
        }
        if only_arch is not None and str(m.arch) not in only_arch:
            continue
        print(f"\n[SARFA] Loading model '{m.name}' (arch={m.arch}, seed={m.seed}): {m.path}")
        agent = BCTEvalAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            temperature=1.0,
            sample=False,  # argmax -- SARFA assumes exploitation phase!!
            path=m.path,
        )
        agents[m.name] = agent

    assert agents, f"only_arch={only_arch} matched none of the models in `models`."

    K_values = {name: agent.K for name, agent in agents.items()}
    assert len(set(K_values.values())) == 1, (
        f"Models have different K values: {K_values}. They must share "
        "the same K to be comparable. Set `K` explicitly in each agent if needed."
    )
    expected_K = cfg.get("expected_K", None)
    if expected_K is not None:
        for name, k in K_values.items():
            assert k == expected_K, (
                f"Model '{name}' has K={k}, expected K={expected_K} (cross-checkout "
                "consistency check -- the other architecture's models must match too)."
            )

    exp_name = str(cfg.exp_name)
    wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.project,
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(start_method="thread", _disable_stats=True),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )

    perturb_cfg = cfg.perturbation
    temporal_blur_sigma = cfg.get("temporal_blur_sigma", None)  # None -> reuses perturb_cfg.blur_sigma
    K_shared = next(iter(K_values.values()))
    out_root = Path(cfg.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if cfg.phase in ("record", "both"):
        trajectories = {name: {} for name in agents}
        for level in cfg.levels:
            for name, agent in agents.items():
                print(f"[SARFA] Recording reference trajectory: model={name} level={level}")
                steps = record_reference_trajectory(agent, cfg.env_name, level, cfg.distribution_mode)
                selected = select_quantile_states(steps, cfg.states_per_episode)
                trajectories[name][level] = selected
                print(f"    episode of {len(steps)} steps -> {len(selected)} sampled states")

        for name in agents:
            states_path = save_trajectory_states(
                out_root, name,
                model_meta[name]["arch"], model_meta[name]["seed"], model_meta[name]["checkpoint_step"],
                trajectories[name],  # {level: [state, ...]}
            )
            print(f"[SARFA] Raw states of '{name}' -> {states_path}")

    if cfg.phase == "record":
        print("\n[SARFA] phase=record done. Run phase=evaluate (from every checkout, "
              "after every architecture's phase=record has completed) to build the "
              "M x M matrix.")
        if cfg.use_wandb:
            wandb.finish()
        return

    # Evaluate
    all_trajectories = {m.name: load_trajectory_states(out_root, m.name) for m in cfg.models}

    n_cells = 0
    for traj_name, traj_states in all_trajectories.items():
        for eval_name, eval_agent in agents.items():
            records = []
            for level, states in traj_states.items():
                for state in states:
                    result = compute_state_saliency(
                        eval_agent, state, perturb_cfg, K_max=K_shared,
                        temporal_blur_sigma=temporal_blur_sigma,
                    )
                    records.append({
                        "level": level,
                        "t": state["t"],
                        "traj_action": state["traj_action"],
                        **result,
                    })

            cell_path = out_root / f"traj_{traj_name}__eval_{eval_name}.npz"
            np.savez_compressed(
                cell_path,
                levels=np.array([r["level"] for r in records]),
                timesteps=np.array([r["t"] for r in records]),
                traj_action=np.array([r["traj_action"] for r in records]),
                a_hat=np.array([r["a_hat"] for r in records]),
                n_s=np.array([r["n_s"] for r in records]),
                answer_maps=np.stack([r["answer_map"] for r in records]),
                dP_maps=np.stack([r["dP_map"] for r in records]),
                K_maps=np.stack([r["K_map"] for r in records]),
                temporal_answer=np.stack([r["temporal_answer"] for r in records]),  # (N, K_shared), NaN-padded
                temporal_dP=np.stack([r["temporal_dP"] for r in records]),
                temporal_K=np.stack([r["temporal_K"] for r in records]),
                traj_source=traj_name,
                eval_model=eval_name,
                traj_arch=model_meta[traj_name]["arch"],
                traj_seed=model_meta[traj_name]["seed"],
                eval_arch=model_meta[eval_name]["arch"],
                eval_seed=model_meta[eval_name]["seed"],
                eval_checkpoint_step=model_meta[eval_name]["checkpoint_step"],
                perturbation_d=perturb_cfg.d,
                perturbation_radius=perturb_cfg.radius,
                perturbation_blur_sigma=perturb_cfg.blur_sigma,
                perturbation_mode=perturb_cfg.mode,
                temporal_blur_sigma=(temporal_blur_sigma if temporal_blur_sigma is not None else perturb_cfg.blur_sigma),
            )
            n_cells += 1
            print(f"[SARFA] traj={traj_name} eval={eval_name}: {len(records)} states -> {cell_path}")

            if cfg.use_wandb:
                wandb.log({
                    f"eval_sarfa/{traj_name}__{eval_name}/n_states": len(records),
                    "eval_sarfa/cells_done": n_cells,
                })

    print(f"\n[SARFA] {n_cells} cells written to {out_root}")
    print("[SARFA] Run analysis/compare_saliency.py on this output folder for "
          "comparison metrics (Spearman, IoU, noise floor, etc.).")
    print("[SARFA] Qualitative cases: run "
          "analysis/select_qualitative_states.py on this output folder "
          "to filter disagreeing states; counterfactual branching and "
          "human recording are not yet implemented.")

    if cfg.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
