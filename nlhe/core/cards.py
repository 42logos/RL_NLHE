from typing import List
"""
Card representation and utilities for poker games.

This module provides a compact integer-based representation of playing cards,
where each card is represented as an integer from 0 to 51.

Constants:
    RANKS: List of valid card ranks (2-14, where 11=Jack, 12=Queen, 13=King, 14=Ace)
    SUITS: List of valid card suits (0-3)

Functions:
    rank_of(c): Get the rank of a card (2-14)
    suit_of(c): Get the suit of a card (0-3)
    make_deck(): Create a standard 52-card deck as a list of integers

Card Encoding:
    Cards 0-12: 2♠, 3♠, 4♠, ..., A♠
    Cards 13-25: 2♥, 3♥, 4♥, ..., A♥
    Cards 26-38: 2♦, 3♦, 4♦, ..., A♦
    Cards 39-51: 2♣, 3♣, 4♣, ..., A♣
"""

    



RANKS = list(range(2, 15))   # 2..14
SUITS = list(range(0, 4))    # 0..3

_CARD_RANK = [2 + (i % 13) for i in range(52)]
_CARD_SUIT = [i // 13 for i in range(52)]

def rank_of(c: int) -> int:
    """
    Get the rank of a card.
    
    Args:
        c: Card integer (0-51)
        
    Returns:
        int: Card rank (2-14, where 11=Jack, 12=Queen, 13=King, 14=Ace)
    """
    return _CARD_RANK[c]

def suit_of(c: int) -> int:
    """
    Get the suit of a card.
    
    Args:
        c: Card integer (0-51)
        
    Returns:
        int: Card suit (0=Spades, 1=Hearts, 2=Diamonds, 3=Clubs)
    """
    return _CARD_SUIT[c]

def make_deck() -> List[int]:
    """
    Create a standard 52-card deck represented as a list of integers.
    
    Returns:
        List[int]: A list containing integers from 0 to 51, where each integer
                   represents a unique playing card in the deck.
    """
    
    return list(range(52))
