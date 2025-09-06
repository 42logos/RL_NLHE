# nlhe/core/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

class ActionType(Enum):
    """
    Enumeration of possible poker actions in a No-Limit Hold'em game.
    This enum defines the four fundamental actions a player can take during
    their turn in a poker hand.
    
    Attributes:
        FOLD: Player forfeits their hand and any claim to the current pot
        CHECK: Player passes the action without betting (only valid when no bet is required)
        CALL: Player matches the current bet amount
        RAISE_TO: Player increases the bet to a specified amount
    """
    
    FOLD = auto()
    CHECK = auto()
    CALL = auto()
    RAISE_TO = auto()

@dataclass
class Action:
    """
    Represents a player action in a poker game.
    This class encapsulates the different types of actions a player can take
    during a poker hand, along with any associated amount for betting actions.
    
    Attributes:
        kind (ActionType): The type of action being performed (e.g., fold, call, raise).
        amount (Optional[int]): The amount associated with the action. Only used
                               for RAISE_TO actions, None for other action types.
                               Defaults to None.
                               
    Example:
        >>> fold_action = Action(kind=ActionType.FOLD)
        >>> raise_action = Action(kind=ActionType.RAISE_TO, amount=100)
    """
    
    kind: ActionType
    amount: Optional[int] = None  # only for RAISE_TO

@dataclass
class PlayerState:
    """
    Represents the state of a player in a poker game.
    
    Attributes:
        hole (Optional[Tuple[int, int]]): The player's hole cards represented as a tuple of two integers.
            None if no cards are dealt or cards are unknown.
        stack (int): The amount of chips/money the player currently has available.
            Defaults to 0.
        bet (int): The amount the player has bet in the current betting round.
            Defaults to 0.
        cont (int): The player's contribution to the pot in the current hand.
            Defaults to 0.
        status (str): The current status of the player. Can be 'active', 'folded', or 'allin'.
            Defaults to 'active'.
        rho (int): A utility or evaluation value associated with the player's position/strategy.
            Defaults to -10^9 (negative billion).
    """
    
    hole: Optional[Tuple[int, int]] = None
    stack: int = 0
    bet: int = 0
    cont: int = 0
    status: str = "active"  # 'active' | 'folded' | 'allin'
    rho: int = -10**9

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

@dataclass
class GameState:
    """Represents the complete state of a No Limit Hold'em poker game at any given point.
        This class encapsulates all the information needed to describe a poker game state,
        including player positions, betting information, cards, and game flow control.
    
    Attributes:
        button (int): Position of the dealer button (0-indexed player position).
        round_label (str): Current betting round identifier (e.g., 'preflop', 'flop', 'turn', 'river').
        board (List[int]): Community cards currently visible on the board.
        undealt (List[int]): Remaining cards in the deck that haven't been dealt.
        players (List[PlayerState]): List of all players and their current states.
        current_bet (int): The current highest bet amount in the current betting round.
        min_raise (int): Minimum raise amount required for the next raise action.
        tau (int): Time limit or timeout value for player actions.
        next_to_act (Optional[int]): Index of the player who should act next, None if no action required.
        step_idx (int): Current step/action index in the game sequence.
        pot (int): Total amount of chips in the pot.
        sb (int): Small blind amount.
        bb (int): Big blind amount.
        actions_log (List[Tuple[int, int, int, int]]): History of all actions taken in the game,
            each tuple typically containing (player_id, action_type, amount, step).
    """
    

    button: int
    round_label: str
    board: List[int]
    undealt: List[int]
    players: List["PlayerState"]
    current_bet: int
    min_raise: int
    tau: int
    next_to_act: Optional[int]
    step_idx: int
    pot: int
    sb: int
    bb: int
    actions_log: List[Tuple[int, int, int, int]] = field(default_factory=list)


@dataclass
class LegalActionInfo:
    """
    Information about legal actions available to a player during their turn.
    This class encapsulates the complete set of legal actions a player can take,
    along with betting constraints and raise permissions.
    
    Attributes:
        actions (List[Action]): List of all legal actions available to the player.
        min_raise_to (Optional[int]): Minimum amount the player can raise to, if raising is allowed.
                                     None if raising is not applicable or not allowed.
        max_raise_to (Optional[int]): Maximum amount the player can raise to, if raising is allowed.
                                     None if raising is not applicable or not allowed.
        has_raise_right (Optional[bool]): Whether the player has the right to raise in this situation.
                                         None if raise rights are not applicable to current game state.
    """
    
    actions: List[Action]
    min_raise_to: Optional[int] = None
    max_raise_to: Optional[int] = None
    has_raise_right: Optional[bool] = None
