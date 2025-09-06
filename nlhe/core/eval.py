# nlhe/core/eval.py
import datetime
from typing import Dict, List, Optional, Tuple
from .cards import rank_of, suit_of

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

def hand_rank_5(cards5: Tuple[int, int, int, int, int]) -> Tuple[int, Tuple[int, ...]]:
    """
    Evaluate the rank of a 5-card poker hand.
    
    Args:
        cards5: A tuple of exactly 5 integers representing cards.
                Each card is encoded as an integer where the rank and suit
                can be extracted using rank_of() and suit_of() functions.
                
    Returns:
        A tuple containing:
        - Hand category (int): The type of poker hand (e.g., straight flush, four of a kind, etc.)
        - Rank values (tuple): The key cards that determine hand strength, ordered by importance.
          For example:
          - Straight flush: (high_card,)
          - Four of a kind: (quad_rank, kicker)
          - Full house: (trips_rank, pair_rank)
          - Flush: (card1, card2, card3, card4, card5) in descending order
          - Straight: (high_card,)
          - Three of a kind: (trips_rank, kicker1, kicker2)
          - Two pair: (high_pair, low_pair, kicker)
          - One pair: (pair_rank, kicker1, kicker2, kicker3)
          - High card: (card1, card2, card3, card4, card5) in descending order
          
    Note:
        - Ace-low straights (A-2-3-4-5) are handled specially with high card = 5
        - Hand categories are compared first, then rank values for tiebreaking
        - Assumes HandCategory enum defines the hand type constants
        
        
    Example:
        - Input: (2, 3, 4, 5, 6)
        - Output: (8, (6,))
    """
    
    ranks = sorted([rank_of(c) for c in cards5], reverse=True)
    suits = [suit_of(c) for c in cards5]

    cnt: Dict[int, int] = {}
    for r in ranks: cnt[r] = cnt.get(r, 0) + 1
    bycnt = sorted(cnt.items(), key=lambda x: (x[1], x[0]), reverse=True)

    is_flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)

    def straight_high(uniq_ranks: List[int]) -> Optional[int]:
        if {14, 5, 4, 3, 2}.issubset(set(uniq_ranks)):
            return 5
        for i in range(len(uniq_ranks) - 4):
            window = uniq_ranks[i:i+5]
            if window[0] - window[4] == 4 and len(set(window)) == 5:
                return window[0]
        
        return None

    s_high = straight_high(uniq)
    print(f"DEBUG: ranks={ranks}, bycnt={bycnt}, is_flush={is_flush}, s_high={s_high}, uniq={uniq}, bycnt={bycnt}")
    if is_flush and s_high is not None:
        return (HandCategory.STRAIGHT_FLUSH, (s_high,))
    if bycnt[0][1] == 4:
        quad = bycnt[0][0]; kicker = max([r for r in ranks if r != quad])
        return (HandCategory.FOUR, (quad, kicker))
    if bycnt[0][1] == 3 and bycnt[1][1] == 2:
        trips = bycnt[0][0]; pair = bycnt[1][0]
        return (HandCategory.FULL_HOUSE, (trips, pair))
    if is_flush:
        return (HandCategory.FLUSH, tuple(ranks))
    if s_high is not None:
        return (HandCategory.STRAIGHT, (s_high,))
    if bycnt[0][1] == 3:
        trips = bycnt[0][0]; kickers = [r for r in ranks if r != trips][:2]
        return (HandCategory.TRIPS, (trips, *kickers))
    if bycnt[0][1] == 2 and bycnt[1][1] == 2:
        hp = max(bycnt[0][0], bycnt[1][0]); lp = min(bycnt[0][0], bycnt[1][0])
        kicker = max([r for r in ranks if r not in (hp, lp)])
        return (HandCategory.TWO_PAIR, (hp, lp, kicker))
    if bycnt[0][1] == 2:
        pair = bycnt[0][0]; kickers = [r for r in ranks if r != pair][:3]
        return (HandCategory.ONE_PAIR, (pair, *kickers))
    return (HandCategory.HIGH, tuple(ranks))

def best5_rank_from_7(cards7: Tuple[int, int, int, int, int, int, int]) -> Tuple[int, Tuple[int, ...]]:
    """
    Determines the best 5-card poker hand rank from a 7-card combination.
    
    Args:
        cards7: A tuple of 7 integers representing card values in a 7-card hand.
        
    Returns:
        A tuple containing:
            - int: The rank of the best 5-card hand (lower values indicate better hands)
            - Tuple[int, ...]: Additional ranking information or tiebreaker values
    Raises:
        NotImplementedError: This is a placeholder function that needs concrete implementation.
    Note:
        This function evaluates all possible 5-card combinations from the given 7 cards
        and returns the ranking information for the strongest poker hand found.
    """
    
    raise NotImplementedError("No evaluation method available")
    return (-1, ())

try:
    import nlhe_eval as _nlhe
    def best5_rank_from_7(cards7: Tuple[int, int, int, int, int, int, int]) -> Tuple[int, Tuple[int, ...]]:
        """
        Evaluate the best 5-card poker hand from 7 given cards and return its rank.
        Args:
            cards7: A tuple of 7 integers representing poker cards.
        Returns:
            A tuple containing:
                - int: The category/rank of the best 5-card hand (e.g., pair, flush, etc.)
                - Tuple[int, ...]: Tiebreaker values for comparing hands of the same category
        Note:
            This function uses the internal _nlhe module to perform the evaluation
            and converts the results to appropriate Python types.
        """
        
        cat, tb = _nlhe.best5_rank_from_7_py(list(cards7))
        return (int(cat), tuple(int(x) for x in tb))
    RUST_EVAL_ACTIVE = True
except Exception as _e:
    raise ImportError("Failed to import nlhe_eval") from _e
    print("Warning: nlhe_eval import failed, using Python fallback:", _e)
    RUST_EVAL_ACTIVE = False
    def best5_rank_from_7(cards7: Tuple[int, int, int, int, int, int, int]) -> Tuple[int, Tuple[int, ...]]:
        """
        Find the best 5-card poker hand from 7 cards.
        This function evaluates all possible 5-card combinations from the given 7 cards
        and returns the highest-ranking hand according to poker rules.
        
        Args:
            cards7: A tuple of 7 integers representing cards
            
        Returns:
            A tuple containing:
            - int: The rank/score of the best 5-card hand
            - Tuple[int, ...]: Tiebreaker values for comparing hands of the same rank
            
        Note:
            This is a Python fallback implementation that uses brute force enumeration
            of all 21 possible 5-card combinations from the 7 input cards.
        """
        
        # Python fallback: enumerate all 5-card combos from 7
        from itertools import combinations
        best = (-1, ())
        for c5 in combinations(cards7, 5):
            r = hand_rank_5(c5)
            if r > best: best = r
        return best


if __name__ == "__main__":
    # Example usage
    hand=(14,2,3,4,5)
    print(hand_rank_5(hand))
    
    hand7=(14,2,3,4,5,6,7)
    print(best5_rank_from_7(hand7))
    
