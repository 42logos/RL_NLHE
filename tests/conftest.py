import sys
from pathlib import Path

# Ensure the repository root is on sys.path so `nlhe` can be imported when the
# project is not installed as a package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Provide a lightweight fallback evaluator so `nlhe.core.engine` can be
# imported without the optional compiled extension.
import types
import itertools
from nlhe.core.cards import rank_of, suit_of


class HandCategory:
    STRAIGHT_FLUSH = 8
    FOUR = 7
    FULL_HOUSE = 6
    FLUSH = 5
    STRAIGHT = 4
    TRIPS = 3
    TWO_PAIR = 2
    ONE_PAIR = 1
    HIGH = 0


def _hand_rank_5(cards5):
    ranks = sorted([rank_of(c) for c in cards5], reverse=True)
    suits = [suit_of(c) for c in cards5]
    cnt = {}
    for r in ranks:
        cnt[r] = cnt.get(r, 0) + 1
    bycnt = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)

    def straight_high(uniq_ranks):
        if {14, 5, 4, 3, 2}.issubset(set(uniq_ranks)):
            return 5
        for i in range(len(uniq_ranks) - 4):
            window = uniq_ranks[i : i + 5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return window[0]
        return None

    s_high = straight_high(uniq)
    if is_flush and s_high is not None:
        return (HandCategory.STRAIGHT_FLUSH, (s_high,))
    if bycnt[0][1] == 4:
        quad = bycnt[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (HandCategory.FOUR, (quad, kicker))
    if bycnt[0][1] == 3 and bycnt[1][1] == 2:
        trips = bycnt[0][0]
        pair = bycnt[1][0]
        return (HandCategory.FULL_HOUSE, (trips, pair))
    if is_flush:
        return (HandCategory.FLUSH, tuple(ranks))
    if s_high is not None:
        return (HandCategory.STRAIGHT, (s_high,))
    if bycnt[0][1] == 3:
        trips = bycnt[0][0]
        kickers = [r for r in ranks if r != trips][:2]
        return (HandCategory.TRIPS, (trips, *kickers))
    if bycnt[0][1] == 2 and bycnt[1][1] == 2:
        hp = max(bycnt[0][0], bycnt[1][0])
        lp = min(bycnt[0][0], bycnt[1][0])
        kicker = max(r for r in ranks if r not in (hp, lp))
        return (HandCategory.TWO_PAIR, (hp, lp, kicker))
    if bycnt[0][1] == 2:
        pair = bycnt[0][0]
        kickers = [r for r in ranks if r != pair][:3]
        return (HandCategory.ONE_PAIR, (pair, *kickers))
    return (HandCategory.HIGH, tuple(ranks))


def _best5_rank_from_7_py(cards7):
    best = None
    for combo in itertools.combinations(cards7, 5):
        val = _hand_rank_5(combo)
        if best is None or val > best:
            best = val
    cat, tb = best
    return cat, list(tb)


stub = types.ModuleType("nlhe_engine")
stub.best5_rank_from_7_py = _best5_rank_from_7_py
sys.modules.setdefault("nlhe_engine", stub)
