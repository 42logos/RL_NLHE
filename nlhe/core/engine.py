# nlhe/core/engine.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random
from .cards import make_deck
from .eval import best5_rank_from_7
from .types import Action, ActionType, GameState, PlayerState, LegalActionInfo

_ACTION_ID = { ActionType.FOLD: 0, ActionType.CHECK: 1, ActionType.CALL: 2, ActionType.RAISE_TO: 3 }
_ROUND_ID = { 'Preflop': 0, 'Flop': 1, 'Turn': 2, 'River': 3 }

def action_type_id(kind: ActionType) -> int:
    return _ACTION_ID[kind]

def round_label_id(label: str) -> int:
    return _ROUND_ID.get(label, 3)

class NLHEngine:
    def __init__(self, sb: int = 1, bb: int = 2, start_stack: int = 100,
                 num_players: int = 6, rng: Optional[random.Random] = None):
        assert num_players == 6, "Engine fixed to 6 players per spec"
        self.N = num_players
        self.sb = sb; self.bb = bb; self.start_stack = start_stack
        self.rng = rng or random.Random()

    # ----- lifecycle -----
    def reset_hand(self, button: int = 0) -> GameState:
        """
        Reset and initialize a new poker hand.
        Creates a fresh game state with shuffled deck, dealt hole cards, posted blinds,
        and initialized player states for the start of a new hand.
        
        Args:
            button (int, optional): Position of the dealer button (0-indexed seat number).
                                   Defaults to 0.
                                   
        Returns:
            GameState: Complete initial game state for the new hand including:
            
                    - Shuffled deck with hole cards dealt to all players
                    - Players with starting stacks and posted blinds
                    - Empty board (preflop)
                    - Current bet set to big blind amount
                    - Next player to act positioned after big blind
                    - Pot containing small blind + big blind amounts
                      
        Side Effects:
            - Modifies player stacks by deducting blind amounts
            - Sets player statuses to 'allin' if they have no chips remaining after posting blinds
            - Updates player bet and contribution amounts for blind posts
            
        Note:
            Small blind is posted by (button + 1) % N player
            Big blind is posted by (button + 2) % N player
            First to act is (button + 3) % N player
        """
        
        deck = make_deck(); self.rng.shuffle(deck)
        players = [PlayerState(stack=self.start_stack) for _ in range(self.N)]
        for i in range(self.N):
            c1, c2 = deck.pop(), deck.pop()
            players[i].hole = (c1, c2)
        board: List[int] = []; undealt = deck[:]

        sb_seat = (button + 1) % self.N
        bb_seat = (button + 2) % self.N
        # SB
        sb_amt = min(self.sb, players[sb_seat].stack)  # Small blind amount
        players[sb_seat].stack -= sb_amt; players[sb_seat].bet += sb_amt; players[sb_seat].cont += sb_amt
        if players[sb_seat].stack == 0 and sb_amt > 0: players[sb_seat].status = 'allin'
        # BB
        bb_amt = min(self.bb, players[bb_seat].stack)  # Big blind amount
        players[bb_seat].stack -= bb_amt; players[bb_seat].bet += bb_amt; players[bb_seat].cont += bb_amt
        if players[bb_seat].stack == 0 and bb_amt > 0: players[bb_seat].status = 'allin'

        current_bet = self.bb; pot = sum(p.cont for p in players)
        for p in players: p.rho = -10**9

        state = GameState(button=button, round_label='Preflop', board=board, undealt=undealt,
                          players=players, current_bet=current_bet, min_raise=self.bb, tau=0,
                          next_to_act=(button + 3) % self.N, step_idx=0, pot=pot, sb=self.sb, bb=self.bb)
        assert state.pot == self.sb + self.bb
        assert state.current_bet == self.bb
        assert state.next_to_act == (button + 3) % self.N
        return state

    # ----- helpers -----
    def owed(self, s: GameState, i: int) -> int:
        """
        Calculate the amount a player owes to match the current bet.
        
        Args:
            s (GameState): The current game state containing betting information.
            i (int): The index of the player in the players list.
            
        Returns:
            int: The amount the player needs to bet to match the current bet.
                 Returns 0 if the player has already bet at or above the current bet.
        """

        return max(0, s.current_bet - s.players[i].bet)

    def _one_survivor(self, s: GameState) -> Optional[int]:
        alive = [i for i, p in enumerate(s.players) if p.status != 'folded']
        return alive[0] if len(alive) == 1 else None

    def _everyone_allin_or_folded(self, s: GameState) -> bool:
        return all((p.status != 'active') or (p.stack == 0) for p in s.players)

    def legal_actions(self, s: GameState) -> LegalActionInfo:
        """
        Determine the legal actions available to the current player in the given game state.
        
        Args:
            s (GameState): The current state of the game including player information,
                          betting rounds, and game parameters.
                          
        Returns:
            LegalActionInfo: Object containing:
                - actions: List of legal Action objects (FOLD, CHECK, CALL, RAISE_TO)
                - min_raise_to: Minimum amount for a raise (if raising is legal)
                - max_raise_to: Maximum amount for a raise (if raising is legal)
                - has_raise_right: Whether the player has the right to raise
                
        Notes:
            - Returns empty actions list if no player is next to act or player is not active
            - FOLD is available when player owes money to the pot
            - CHECK is available when player owes nothing (current_bet equals player's bet)
            - CALL is available when player owes money and has chips
            - RAISE_TO is available when player is active, has chips, and can raise above current bet
            - Raise rights depend on player's rho value relative to tau or if no current bet exists
            - Min raise amount considers minimum raise rules and current betting situation
            - Max raise amount is limited by player's total chips (bet + stack)
        """
        
        i = s.next_to_act
        if i is None: return LegalActionInfo(actions=[])
        p = s.players[i]
        if p.status != 'active': return LegalActionInfo(actions=[])

        owe = self.owed(s, i)
        acts: List[Action] = []
        if owe > 0: acts.append(Action(ActionType.FOLD))
        if owe == 0: acts.append(Action(ActionType.CHECK))
        if owe > 0: acts.append(Action(ActionType.CALL))

        can_raise = (p.status == 'active') and (p.stack > 0)
        if not can_raise:
            return LegalActionInfo(actions=acts)

        if s.current_bet == 0:
            min_to = max(s.min_raise, 1)
        else:
            min_to = s.current_bet + s.min_raise
        max_to = p.bet + p.stack
        has_rr = (p.rho < s.tau) or (s.current_bet == 0)
        if max_to > s.current_bet:
            acts.append(Action(ActionType.RAISE_TO))
            return LegalActionInfo(actions=acts, min_raise_to=min_to, max_raise_to=max_to, has_raise_right=has_rr)
        return LegalActionInfo(actions=acts)

    # ----- step -----
    def step(self, s: GameState, a: Action) -> Tuple[GameState, bool, Optional[List[int]], Dict[str, Any]]:
        """
        Execute a single action in the poker game and update the game state.
        
        Args:
            s (GameState): Current game state containing player information, betting rounds, etc.
            a (Action): Action to be executed (FOLD, CHECK, CALL, or RAISE_TO)
            
        Returns:
            Tuple containing:
                - GameState: Updated game state after the action
                - bool: Whether the game/round is finished
                - Optional[List[int]]: Reward/payout list for players if game is done, None otherwise
                - Dict[str, Any]: Additional information dictionary (currently empty)
                
        Raises:
            ValueError: If action type is unknown or RAISE_TO action lacks amount
            AssertionError: If action violates game rules (e.g., invalid bet amounts, 
                           player not active, insufficient stack, etc.)
                           
        Notes:
            - Updates player status, betting amounts, and game progression
            - Handles betting rights and minimum raise requirements
            - Logs all actions for game history
            - Automatically advances to next player or next betting round as needed
            - Player with insufficient chips for full call/raise will go all-in
        """
        
        i = s.next_to_act; assert i is not None
        p = s.players[i]; assert p.status == 'active'
        s.step_idx += 1

        def advance_next():
            j = (i + 1) % self.N
            for _ in range(self.N):
                pj = s.players[j]
                if pj.status == 'active':
                    owej = self.owed(s, j)
                    if pj.rho < s.tau or owej > 0:
                        s.next_to_act = j; return
                j = (j + 1) % self.N
            s.next_to_act = None

        def add_chips(idx: int, amount: int):
            sp = s.players[idx]
            assert amount >= 0 and sp.stack >= amount
            sp.stack -= amount; sp.bet += amount; sp.cont += amount

        owe = self.owed(s, i); B_old = s.current_bet

        if a.kind == ActionType.FOLD:
            p.status = 'folded'; p.rho = s.step_idx; advance_next()

        elif a.kind == ActionType.CHECK:
            assert owe == 0; p.rho = s.step_idx; advance_next()

        elif a.kind == ActionType.CALL:
            assert owe > 0
            call_amt = min(owe, p.stack); add_chips(i, call_amt)
            if p.stack == 0: p.status = 'allin'
            p.rho = s.step_idx; advance_next()

        elif a.kind == ActionType.RAISE_TO:
            raise_to = a.amount; 
            if raise_to is None: raise ValueError("RAISE_TO requires amount")
            assert raise_to > s.current_bet
            max_to = p.bet + p.stack; assert raise_to <= max_to
            has_rr = (p.rho < s.tau) or (s.current_bet == 0)
            required_min_to = max(s.min_raise, 1) if s.current_bet == 0 else s.current_bet + s.min_raise
            if not has_rr:
                assert raise_to == max_to, "Only all-in allowed; raise rights are closed"
            else:
                assert (raise_to >= required_min_to) or (raise_to == max_to)
            delta = raise_to - p.bet; add_chips(i, delta)
            if p.stack == 0: p.status = 'allin'
            s.current_bet = raise_to; p.rho = s.step_idx
            full_increment = raise_to - B_old
            if full_increment >= s.min_raise:
                s.tau = s.step_idx; s.min_raise = full_increment
                for j, pj in enumerate(s.players):
                    if j != i and pj.status == 'active': pj.rho = -10**9
            advance_next()
        else:
            raise ValueError("Unknown action type")

        # log
        aid = _ACTION_ID[a.kind]; rid = _ROUND_ID.get(s.round_label, 3)
        log_amt = 0
        if a.kind == ActionType.CALL: log_amt = min(owe, p.bet) if False else 0  # amount tracked via cont/bet
        if a.kind == ActionType.RAISE_TO: log_amt = s.current_bet
        s.actions_log.append((i, aid, int(log_amt), rid))

        s.pot = sum(pl.cont for pl in s.players)
        s.current_bet = max(pl.bet for pl in s.players)
        done, rewards = self.advance_round_if_needed(s)
        info: Dict[str, Any] = {}
        return s, done, rewards, info

    # ----- public advance / showdown -----
    def advance_round_if_needed(self, s: GameState) -> Tuple[bool, Optional[List[int]]]:
        """
        Advance the poker game round if conditions are met and return game completion status.
        This method checks if the current betting round should advance to the next stage
        or if the game should end. It handles various end-game scenarios including
        single survivor situations and showdowns.
        
        Args:
            s (GameState): The current state of the poker game containing player
                          information, pot size, community cards, and round details.
                          
        Returns:
            terminated (bool): True if the game/hand is complete, False if betting continues
            rewards (Optional[List[int]]): List of rewards for each player if game is complete,
                    None if game continues.
                    Rewards represent net winnings (pot winnings minus contributions).

        Game advancement logic:
            - If only one player remains (others folded), awards pot to survivor
            - If betting round is complete and players are all-in or folded before River,
              deals remaining community cards and proceeds to showdown
            - If betting round is complete for Preflop/Flop/Turn, deals next street
              and resets betting round
            - If River betting is complete, proceeds to showdown
            - Returns False with None rewards if betting round should continue
            
        Note:
            Rewards calculation ensures zero-sum game where sum of all rewards equals 0.
        """
        
        lone = self._one_survivor(s)
        if lone is not None:
            rewards = [0]*self.N
            for i, p in enumerate(s.players):
                rewards[i] = (s.pot if i == lone else 0) - p.cont
            assert sum(rewards) == 0
            return True, rewards

        if not self._round_open(s):
            if s.round_label in ('Preflop', 'Flop', 'Turn') and self._everyone_allin_or_folded(s):
                while s.round_label != 'River':
                    self._deal_next_street(s)
                s.round_label = 'Showdown'
                rewards = self._settle_showdown(s)
                return True, rewards

            if s.round_label == 'Preflop':
                self._deal_next_street(s); self._reset_round(s); return False, None
            elif s.round_label == 'Flop':
                self._deal_next_street(s); self._reset_round(s); return False, None
            elif s.round_label == 'Turn':
                self._deal_next_street(s); self._reset_round(s); return False, None
            elif s.round_label == 'River':
                s.round_label = 'Showdown'
                rewards = self._settle_showdown(s); return True, rewards
        return False, None

    def _round_open(self, s: GameState) -> bool:
        for i, p in enumerate(s.players):
            if p.status == 'active':
                if p.rho < s.tau or self.owed(s, i) > 0:
                    return True
        return False

    def _deal_next_street(self, s: GameState) -> None:
        if s.round_label == 'Preflop':
            draw = [s.undealt.pop() for _ in range(3)]; s.board.extend(draw); s.round_label = 'Flop'
        elif s.round_label == 'Flop':
            s.board.append(s.undealt.pop()); s.round_label = 'Turn'
        elif s.round_label == 'Turn':
            s.board.append(s.undealt.pop()); s.round_label = 'River'
        else:
            raise RuntimeError('No further streets to deal')

    def _reset_round(self, s: GameState) -> None:
        for p in s.players:
            p.bet = 0
            if p.status == 'active': p.rho = -10**9
        s.current_bet = 0; s.min_raise = s.bb; s.tau = 0
        n = (s.button + 1) % self.N
        for _ in range(self.N):
            if s.players[n].status == 'active':
                s.next_to_act = n; break
            n = (n + 1) % self.N
        else:
            s.next_to_act = None

    def _settle_showdown(self, s: GameState) -> List[int]:
        A = [i for i, p in enumerate(s.players) if p.status != 'folded']
        levels = sorted({p.cont for p in s.players if p.cont > 0})
        if not levels: return [0]*self.N
        ranks = {}
        for i in A:
            hole = s.players[i].hole; assert hole is not None
            seven = list(hole) + s.board; ranks[i] = best5_rank_from_7(seven)
        rewards = [0]*self.N; y_prev = 0; carry = 0
        last_nonempty_winners: Optional[List[int]] = None
        for y in levels:
            contributors_count = sum(1 for p in s.players if p.cont >= y)
            Pk = contributors_count * (y - y_prev) + carry
            elig = [i for i in A if s.players[i].cont >= y]
            if elig:
                best_val = max(ranks[i] for i in elig)
                winners = [i for i in elig if ranks[i] == best_val]
                last_nonempty_winners = winners[:]
                share, rem = divmod(Pk, len(winners))
                for w in winners: rewards[w] += share
                if rem:
                    start = (s.button + 1) % self.N
                    ordered = sorted(winners, key=lambda j: (j - start) % self.N)
                    for k in range(rem): rewards[ordered[k % len(ordered)]] += 1
                carry = 0
            else:
                carry = Pk
            y_prev = y
        if carry and last_nonempty_winners:
            winners = last_nonempty_winners
            share, rem = divmod(carry, len(winners))
            for w in winners: rewards[w] += share
            if rem:
                start = (s.button + 1) % self.N
                ordered = sorted(winners, key=lambda j: (j - start) % self.N)
                for k in range(rem): rewards[ordered[k % len(ordered)]] += 1
        assert sum(rewards) == s.pot
        RL = [rewards[i] - s.players[i].cont for i in range(self.N)]
        assert sum(RL) == 0
        return RL
