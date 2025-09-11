
#!/usr/bin/env python3
"""
compare_engines.py — lockstep differential tester for two NLHEngine implementations.

It runs the pure-Python engine and the Rust-backed Python wrapper side-by-side,
chooses actions from the intersection of their legal action sets, and reports the
first point of divergence (or all divergences).

Usage (from your project root, where `nlhe` is importable and the Rust extension is built):

  python compare_engines.py --episodes 50 --max-steps 400 --seed 123 \
    --sb 1 --bb 2 --stack 100 --num-players 6 --stop-on-first

Tips:
- Place this script anywhere, but ensure `python -c "import nlhe"` works.
- If you can’t import via the package, see the fallback import notes below.
"""

import argparse
import json
import os
import sys
import traceback
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
# ---------- Imports ----------
# Preferred: import via your package (recommended).
try:
    from nlhe.core.types import Action, ActionType
    from nlhe.core.engine import NLHEngine as PyEngine
    from nlhe.core.rs_engine import NLHEngine as RsEngine
    PACKAGE_MODE = "package"
except Exception as e:
    # Fallback: best-effort local file loading. This requires that the relative imports
    # inside engine/rs_engine resolve (i.e., you're running from within the package).
    PACKAGE_MODE = f"fallback ({e})"
    import importlib.util

    def _load_mod(name: str, rel_path: str):
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, rel_path)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod

    # NOTE: This fallback only works if you're running from the package tree so that
    # the relative imports (e.g., ".types") used by those modules can resolve.
    engine_mod = _load_mod("engine", "engine.py")
    rs_engine_mod = _load_mod("rs_engine", "rs_engine.py")
    PyEngine = engine_mod.NLHEngine
    RsEngine = rs_engine_mod.NLHEngine
    # Action/ActionType must come from your package for equality; try to load them too.
    try:
        from nlhe.core.types import Action, ActionType  # type: ignore
    except Exception:
        # Try sibling file "types.py"
        types_mod = _load_mod("types", "types.py")
        Action = types_mod.Action
        ActionType = types_mod.ActionType

import random

# ---------- Utilities ----------

def pretty_action(a: Action) -> str:
    if a.kind == ActionType.RAISE_TO:
        return f"{a.kind.name}(to={a.amount})"
    return a.kind.name

def canonical_legal(li) -> Dict[str, Any]:
    """
    Convert LegalActionInfo into a canonical dict for equality & printing.
    - actions: list of action kind names in stable order, 'RAISE_TO' included
    - For raises, include min/max and has_raise_right flags (can be None).
    """
    kinds = [a.kind.name for a in li.actions]
    return dict(
        actions=kinds,
        min_raise_to=li.min_raise_to,
        max_raise_to=li.max_raise_to,
        has_raise_right=li.has_raise_right,
    )

def choose_action(li_py, li_rs, rng: random.Random) -> Tuple[Optional[Action], Optional[str]]:
    """
    Pick an action from the intersection of legal sets.
    Returns (Action or None, reason_if_none).
    Preference order: CHECK > CALL > RAISE_TO > FOLD.
    For raises, if both allow raises, choose either min or max of the INTERSECTION range.
    """
    kinds_py = {a.kind for a in li_py.actions}
    kinds_rs = {a.kind for a in li_rs.actions}
    common = kinds_py & kinds_rs
    if not common:
        return None, "no_common_actions"

    order = [ActionType.CHECK, ActionType.CALL, ActionType.RAISE_TO, ActionType.FOLD]
    choice_kind = None
    for k in order:
        if k in common:
            choice_kind = k
            break

    assert choice_kind is not None

    if choice_kind != ActionType.RAISE_TO:
        return Action(choice_kind), None

    # RAISE_TO: compute intersection [lo, hi]
    # Guard against None values.
    mins = [x for x in (li_py.min_raise_to, li_rs.min_raise_to) if x is not None]
    maxs = [x for x in (li_py.max_raise_to, li_rs.max_raise_to) if x is not None]
    if len(mins) < 2 or len(maxs) < 2:
        return None, "raise_bounds_missing"
    lo = max(mins)
    hi = min(maxs)
    if lo > hi:
        return None, "raise_range_disjoint"

    # If raise rights disagree, still attempt an all-in if both allow hi == max.
    if (li_py.has_raise_right is not None and li_rs.has_raise_right is not None
        and (li_py.has_raise_right != li_rs.has_raise_right)):
        # If both allow max (all-in), pick hi; otherwise report mismatch.
        if hi is None:
            return None, "has_raise_right_mismatch"
        return Action(ActionType.RAISE_TO, hi), None

    # Choose deterministically: 70% min raise, 30% max raise (mix coverage)
    amount = lo if rng.random() < 0.7 else hi
    return Action(ActionType.RAISE_TO, amount), None

@dataclass
class PlayerSnap:
    stack: int
    bet: int
    cont: int
    status: str
    rho: int

@dataclass
class StateSnap:
    round_label: str
    next_to_act: Optional[int]
    current_bet: int
    min_raise: int
    tau: int
    step_idx: int
    pot: int
    players: List[PlayerSnap]

def snapshot(s) -> StateSnap:
    return StateSnap(
        round_label=s.round_label,
        next_to_act=s.next_to_act,
        current_bet=int(s.current_bet),
        min_raise=int(s.min_raise),
        tau=int(s.tau),
        step_idx=int(s.step_idx),
        pot=int(s.pot),
        players=[PlayerSnap(int(p.stack), int(p.bet), int(p.cont), str(p.status), int(p.rho)) for p in s.players],
    )

def diff_states(a: StateSnap, b: StateSnap) -> Dict[str, Any]:
    diffs: Dict[str, Any] = {}
    for field in ["round_label","next_to_act","current_bet","min_raise","tau","step_idx","pot"]:
        va, vb = getattr(a, field), getattr(b, field)
        if va != vb:
            diffs[field] = (va, vb)
    # Per-player
    plist = []
    for i, (pa, pb) in enumerate(zip(a.players, b.players)):
        pdelta = {}
        for f in ["stack","bet","cont","status","rho"]:
            va, vb = getattr(pa, f), getattr(pb, f)
            if va != vb:
                pdelta[f] = (va, vb)
        if pdelta:
            plist.append((i, pdelta))
    if plist:
        diffs["players"] = plist
    return diffs

def check_invariants(label: str, s) -> List[str]:
    errs = []
    # pot equals sum(cont)
    pot_calc = sum(int(p.cont) for p in s.players)
    if int(s.pot) != pot_calc:
        errs.append(f"{label}: pot({s.pot}) != sum(cont)({pot_calc})")
    # current_bet equals max(bet)
    max_bet = max(int(p.bet) for p in s.players)
    if int(s.current_bet) != max_bet:
        errs.append(f"{label}: current_bet({s.current_bet}) != max(bet)({max_bet})")
    # owed >= 0 for all
    # owed uses engine's owed() to include its logic
    return errs

@dataclass
class StepLog:
    step: int
    actor: Optional[int]
    round_label: str
    legal_py: Dict[str, Any]
    legal_rs: Dict[str, Any]
    chosen: Optional[str]
    reason_if_none: Optional[str]
    state_py: StateSnap
    state_rs: StateSnap
    state_diff: Dict[str, Any]

def run_episode(cfg) -> Dict[str, Any]:
    rng = random.Random(cfg.seed)
    # Build engines with (optionally) the SAME RNG instance passed to both.
    # They will use it differently, but we keep determinism for action selection.
    shared_rng = random.Random(rng.getrandbits(64))

    py = PyEngine(sb=cfg.sb, bb=cfg.bb, start_stack=cfg.stack, num_players=cfg.num_players, rng=shared_rng)
    rs = RsEngine(sb=cfg.sb, bb=cfg.bb, start_stack=cfg.stack, num_players=cfg.num_players, rng=shared_rng)

    s_py = py.reset_hand(button=cfg.button)
    s_rs = rs.reset_hand(button=cfg.button)

    logs: List[StepLog] = []
    divergence: Optional[str] = None
    term_info: Optional[Dict[str, Any]] = None

    for t in range(cfg.max_steps):
        # Compare legal sets
        li_py = py.legal_actions(s_py)
        li_rs = rs.legal_actions(s_rs)

        legal_py = canonical_legal(li_py)
        legal_rs = canonical_legal(li_rs)

        chosen_action: Optional[Action] = None
        reason: Optional[str] = None

        if legal_py != legal_rs:
            divergence = "legal_actions_mismatch"
            # Try to choose from intersection anyway (to continue and see where it collapses).
            chosen_action, reason = choose_action(li_py, li_rs, rng)
        else:
            chosen_action, reason = choose_action(li_py, li_rs, rng)

        st_py = snapshot(s_py); st_rs = snapshot(s_rs)
        diff_pre = diff_states(st_py, st_rs)
        if diff_pre and divergence is None:
            divergence = "state_mismatch_before_step"

        # If no common action can be chosen, stop here.
        if chosen_action is None:
            logs.append(StepLog(t, s_py.next_to_act, s_py.round_label, legal_py, legal_rs, None, reason, st_py, st_rs, diff_pre))
            term_info = dict(reason=reason, where="before_step", step=t)
            break

        # Apply to both engines; capture exceptions independently.
        exc_py = exc_rs = None
        done_py = done_rs = False
        rew_py = rew_rs = None

        try:
            s_py, done_py, rew_py, _ = py.step(s_py, chosen_action)
        except Exception as e:
            exc_py = "".join(traceback.format_exception_only(type(e), e)).strip()

        try:
            s_rs, done_rs, rew_rs, _ = rs.step(s_rs, chosen_action)
        except Exception as e:
            exc_rs = "".join(traceback.format_exception_only(type(e), e)).strip()

        st_py2 = snapshot(s_py); st_rs2 = snapshot(s_rs)
        diff_post = diff_states(st_py2, st_rs2)

        # Log step
        logs.append(StepLog(
            t,
            actor=st_py.next_to_act,
            round_label=st_py.round_label,
            legal_py=legal_py,
            legal_rs=legal_rs,
            chosen=pretty_action(chosen_action),
            reason_if_none=reason,
            state_py=st_py2,
            state_rs=st_rs2,
            state_diff=diff_post,
        ))

        # Check for exceptions divergence
        if exc_py or exc_rs:
            divergence = divergence or "exception"
            term_info = dict(exc_py=exc_py, exc_rs=exc_rs, step=t)
            break

        # Check for state divergence after step
        if diff_post:
            divergence = divergence or "state_mismatch_after_step"

        # Check invariants on each engine
        inv_py = check_invariants("py", s_py)
        inv_rs = check_invariants("rs", s_rs)
        if inv_py or inv_rs:
            divergence = divergence or "invariant_violation"
            term_info = dict(inv_py=inv_py, inv_rs=inv_rs, step=t)
            break

        # Terminal discrepancies
        if done_py != done_rs:
            divergence = divergence or "termination_flag_mismatch"
            term_info = dict(done_py=done_py, done_rs=done_rs, step=t)
            break

        if done_py and done_rs:
            # Compare rewards if both ended
            if (rew_py is None) != (rew_rs is None) or (rew_py is not None and list(rew_py) != list(rew_rs)):
                divergence = divergence or "rewards_mismatch"
                term_info = dict(rew_py=rew_py, rew_rs=rew_rs, step=t)
            break

        if cfg.stop_on_first and divergence:
            term_info = dict(step=t)
            break

    return dict(
        divergence=divergence,
        term_info=term_info,
        steps=[asdict(x) for x in logs],
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sb", type=int, default=1)
    ap.add_argument("--bb", type=int, default=2)
    ap.add_argument("--stack", type=int, default=100)
    ap.add_argument("--num-players", type=int, default=6)
    ap.add_argument("--stop-on-first", action="store_true")
    ap.add_argument("--buttons", type=str, default="0")  # e.g., "0,1,2,3,4,5" or "all"
    ap.add_argument("--json-out", type=str, default=None)
    cfg = ap.parse_args()

    # Expand buttons
    if cfg.buttons.strip().lower() == "all":
        buttons = list(range(cfg.num_players))
    else:
        buttons = [int(x) for x in cfg.buttons.split(",") if x.strip() != ""]

    rng = random.Random(cfg.seed)
    results = []
    print(f"[import mode] {PACKAGE_MODE}")
    for ep in range(cfg.episodes):
        button = buttons[ep % len(buttons)]
        ep_seed = rng.getrandbits(64)
        class EpCfg:
            pass
        epcfg = EpCfg()
        epcfg.sb = cfg.sb; epcfg.bb = cfg.bb; epcfg.stack = cfg.stack
        epcfg.num_players = cfg.num_players; epcfg.max_steps = cfg.max_steps
        epcfg.stop_on_first = cfg.stop_on_first; epcfg.seed = ep_seed; epcfg.button = button

        out = run_episode(epcfg)
        results.append(dict(episode=ep, button=button, seed=ep_seed, **out))

        div = out["divergence"]
        if div:
            print(f"[EP {ep}] divergence: {div} @ button={button} seed={ep_seed}")
            if cfg.stop_on_first:
                break
        else:
            print(f"[EP {ep}] OK")

    if cfg.json_out:
        with open(cfg.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Wrote {cfg.json_out}")

    # Print a short human summary of first divergence, if any
    for r in results:
        if r["divergence"]:
            print("\n=== First divergence summary ===")
            print(json.dumps(r, ensure_ascii=False, indent=2))
            break

if __name__ == "__main__":
    main()
