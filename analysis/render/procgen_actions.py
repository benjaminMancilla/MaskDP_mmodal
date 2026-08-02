"""
Procgen's canonical action table. The 15 combos are the same for every game;
each game simply leaves some of them without effect (in coinrun, 3, 4 and 9-14
are all no-ops). Combos are kept distinct here rather than collapsed, so a
figure never implies two different action ids were the same decision.
Reference: https://github.com/seanpm2001/OpenAI_ProcGen/blob/master/procgen/env.py
"""

ACTION_COMBOS = (
    "LEFT+DOWN",   # 0
    "LEFT",        # 1
    "LEFT+UP",     # 2
    "DOWN",        # 3
    "(none)",      # 4  empty combo
    "UP",          # 5
    "RIGHT+DOWN",  # 6
    "RIGHT",       # 7
    "RIGHT+UP",    # 8
    "D",           # 9
    "A",           # 10
    "W",           # 11
    "S",           # 12
    "Q",           # 13
    "E",           # 14
)

NUM_ACTIONS = len(ACTION_COMBOS)

# Arrows for the movement combos, the bare button letter for the rest. Every
# figure also prints the action id next to the glyph, which is what keeps 3, 4
# and 9-14 distinguishable once they render as no-ops.
ACTION_GLYPHS = (
    "↙",    # 0
    "←",    # 1
    "↖",    # 2
    "↓",    # 3
    "·",    # 4
    "↑",    # 5
    "↘",    # 6
    "→",    # 7
    "↗",    # 8
    "D",    # 9
    "A",    # 10
    "W",    # 11
    "S",    # 12
    "Q",    # 13
    "E",    # 14
)

assert len(ACTION_GLYPHS) == NUM_ACTIONS


def _checked(action_id):
    action_id = int(action_id)
    assert 0 <= action_id < NUM_ACTIONS, (
        f"action id {action_id} outside procgen's 0..{NUM_ACTIONS - 1}"
    )
    return action_id


def action_glyph(action_id):
    """Compact glyph for a token box or an axis tick."""
    return ACTION_GLYPHS[_checked(action_id)]


def action_label(action_id):
    """Literal combo name, for legends and captions."""
    return ACTION_COMBOS[_checked(action_id)]
