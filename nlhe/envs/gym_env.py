from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from ..core.engine import NLHEngine
from ..core.types import Action, ActionType, GameState
from ..agents.tamed_random import TamedRandomAgent

class NLHEGymEnv(gym.Env):
    """
    A Gym environment for No Limit Hold'em (NLHE) poker with 6 players.
    This environment provides a reinforcement learning interface for training poker agents
    in a 6-max No Limit Hold'em setting. The environment handles multiple bot opponents
    and allows a "hero" agent (controlled by the RL algorithm) to play against them.
    
    Parameters:
        hero_seat (int): The seat position (0-5) of the hero player controlled by the RL agent.
                         Default is 0.
        seed (Optional[int]): Random seed for reproducible gameplay. Default is None.
        sb (int): Small blind amount. Default is 1.
        bb (int): Big blind amount. Default is 2.
        start_stack (int): Starting stack size for all players. Default is 100.
        bot_kwargs (Optional[dict]): Additional keyword arguments to pass to bot agents.
                                    Default is None.
                                    
    Attributes:                                
        Observation_Space Dict : Dictionary space containing:
        
            - hero_hole: Box(2,) - Hero's hole cards (card indices, -1 if unknown)
            - board: Box(5,) - Community cards (padded with -1 for incomplete boards)
            - stacks: Box(6,) - Current stack sizes for all players
            - bets: Box(6,) - Current bet amounts for all players
            - conts: Box(6,) - Total contributions to pot for all players
            - status: MultiDiscrete([3]*6) - Player status (0=active, 1=folded, 2=all-in)
            - button: Discrete(6) - Button position
            - next_to_act: Discrete(6) - Next player to act
            - round: Discrete(5) - Current betting round (0=Preflop, 1=Flop, 2=Turn, 3=River, 4=Showdown)
            - current_bet: Box(1,) - Current bet to match
            - min_raise: Box(1,) - Minimum raise amount
            - action_mask: MultiBinary(7) - Legal actions mask

        Action_Space Dict: Discrete(7) with actions:
        
            - 0: Fold
            - 1: Check
            - 2: Call
            - 3: Raise to minimum
            - 4: Raise to minimum + 2*big_blind
            - 5: Raise to minimum + 4*big_blind
            - 6: All-in (raise to maximum)
            
    Methods:
        - reset: Reset the environment to an initial state
        - step: Take a step in the environment using the provided action
        - render: Render the current state of the environment

    Notes:
        - The environment automatically plays bot actions until it's the hero's turn or the hand ends
        - Cards are represented as integers from 0-51 (standard deck encoding)
        - Rewards are given only at hand completion and represent chip gains/losses
        - The environment ensures only legal actions can be taken through action masking
    """
    
    metadata = {"render_modes": ["ansi"], "name": "NLHE-6Max-v0"}

    def __init__(self, hero_seat: int = 0, seed: Optional[int] = None,
                 sb: int = 1, bb: int = 2, start_stack: int = 100,
                 bot_kwargs: Optional[dict] = None):
        
        assert 0 <= hero_seat < 6
        self.hero = hero_seat
        self.rng = random.Random(seed)
        self.env = NLHEngine(sb=sb, bb=bb, start_stack=start_stack, rng=self.rng)
        self.bots = [TamedRandomAgent(self.rng) for _ in range(self.env.N)]
        if bot_kwargs:
            self.bots = [TamedRandomAgent(self.rng, **bot_kwargs) for _ in range(self.env.N)]
        self.bots[self.hero] = None  # type: ignore

        self.observation_space = spaces.Dict({
            "hero_hole": spaces.Box(low=-1, high=51, shape=(2,), dtype=np.int32),
            "board": spaces.Box(low=-1, high=51, shape=(5,), dtype=np.int32),
            "stacks": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
            "bets": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
            "conts": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int32),
            "status": spaces.MultiDiscrete([3]*6),
            "button": spaces.Discrete(6),
            "next_to_act": spaces.Discrete(6),
            "round": spaces.Discrete(5),
            "current_bet": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int32),
            "min_raise": spaces.Box(low=0, high=10**6, shape=(1,), dtype=np.int32),
            "action_mask": spaces.MultiBinary(7),
        })
        self.action_space = spaces.Discrete(7)
        self._state: Optional[GameState] = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to start a new episode.
        
        Args:
            seed (Optional[int], optional): Random seed for reproducibility. If provided,
                the random number generator will be seeded with this value. Defaults to None.
            options (Optional[dict], optional): Additional options for resetting the environment.
                Currently unused. Defaults to None.
                
        Returns:
            tuple: A tuple containing:
                - obs: The initial observation of the environment after reset
                - info (dict): Information dictionary. Contains "terminal_rewards" key with
                  final rewards if the episode terminates immediately after reset, otherwise empty.
        """
        
        if seed is not None: self.rng.seed(seed)
        self._state = self.env.reset_hand(button=0)
        term, final_rewards = self._auto_to_hero_or_terminal()
        obs = self._obs(); info = {}
        if term: info["terminal_rewards"] = final_rewards
        return obs, info

    def step(self, action: int):
        """
        Execute one step in the environment with the given action.
        
        Args:
            action (int): The action to take, which will be mapped to the appropriate
                         internal action representation.
                         
        Returns:
            tuple: A 5-tuple containing:
                - obs: Current observation after the step
                - reward (float): Reward for the hero player (0.0 if game continues)
                - terminated (bool): True if the episode has ended
                - truncated (bool): Always False in this implementation
                - info (dict): Additional information, may contain "rewards_all" 
                              with rewards for all players when episode ends
                              
        Raises:
            AssertionError: If the environment state is None or if it's not the 
                           hero's turn to act.
                           
        Notes:
            - Automatically progresses the game to the hero's next turn or terminal state
            - Returns rewards only when the episode terminates
            - The terminated flag is set to True when the game reaches a final state
        """
        
        assert self._state is not None
        s = self._state; assert s.next_to_act == self.hero
        a = self._map_action(action)
        s, done, rewards, info = self.env.step(s, a)
        self._state = s
        if done:
            hero_reward = rewards[self.hero]; obs = self._obs()
            return obs, hero_reward, True, False, {"rewards_all": rewards}
        term, final_rewards = self._auto_to_hero_or_terminal()
        obs = self._obs()
        if term:
            hero_reward = final_rewards[self.hero]
            return obs, hero_reward, True, False, {"rewards_all": final_rewards}
        return obs, 0.0, False, False, {}

    def render(self):
        """
        Render the current state of the poker environment as a human-readable string.
        
        Returns:
            str: A formatted string representation of the current game state including:
                - Button position, round label, pot size, and current bet
                - Community board cards
                - Player information for each seat (status, stack, bet, contribution)
                
        Raises:
            AssertionError: If the environment state is None (not initialized).

        Example:
            env = PokerEnv()
            env.reset()
            print(env.render())   
            BTN=0 rd=Preflop pot=9 B=2
            
            >>> board=
            >>> seat 0: st=active stack=100 bet=0 cont=0
            >>> seat 1: st=active stack=99 bet=1 cont=1
            >>> seat 2: st=active stack=98 bet=2 cont=2
            >>> seat 3: st=active stack=98 bet=2 cont=2
            >>> seat 4: st=active stack=98 bet=2 cont=2
            >>> seat 5: st=active stack=98 bet=2 cont=2  
            """
        
        assert self._state is not None
        from io import StringIO
        buf = StringIO(); s = self._state
        buf.write(f"BTN={s.button} rd={s.round_label} pot={s.pot} B={s.current_bet}\n")
        buf.write("board=" + " ".join(str(c) for c in s.board) + "\n")
        for i, p in enumerate(s.players):
            buf.write(f"seat {i}: st={p.status} stack={p.stack} bet={p.bet} cont={p.cont}\n")
        return buf.getvalue()

    # helpers
    def _auto_to_hero_or_terminal(self) -> Tuple[bool, Optional[List[int]]]:
        assert self._state is not None
        s = self._state; done = False; rewards = None
        while (s.next_to_act is not None) and (s.next_to_act != self.hero):
            seat = s.next_to_act; bot = self.bots[seat]
            assert bot is not None
            a = bot.act(self.env, s, seat)
            s, done, rewards, _ = self.env.step(s, a)
            self._state = s
            if done: return True, rewards
        if s.next_to_act is None:
            done, rewards = self.env.advance_round_if_needed(s); self._state = s
            if done: return True, rewards
        return False, None

    def _obs(self):
        assert self._state is not None
        s = self._state
        rd_map = {"Preflop":0, "Flop":1, "Turn":2, "River":3, "Showdown":4}
        hole = s.players[self.hero].hole
        hero_hole = np.array([hole[0], hole[1]], dtype=np.int32) if hole else np.array([-1,-1], dtype=np.int32)
        board = s.board + ([-1] * (5 - len(s.board))); board = np.array(board, dtype=np.int32)
        stacks = np.array([p.stack for p in s.players], dtype=np.int32)
        bets = np.array([p.bet for p in s.players], dtype=np.int32)
        conts = np.array([p.cont for p in s.players], dtype=np.int32)
        status_map = {"active":0, "folded":1, "allin":2}
        status = np.array([status_map[p.status] for p in s.players], dtype=np.int64)
        button = np.int32(s.button)
        next_to_act = np.int32(s.next_to_act if s.next_to_act is not None else 0)
        round_id = np.int32(rd_map[s.round_label])
        current_bet = np.array([s.current_bet], dtype=np.int32)
        min_raise = np.array([s.min_raise], dtype=np.int32)
        mask = self._action_mask()
        return {
            "hero_hole": hero_hole, "board": board, "stacks": stacks, "bets": bets, "conts": conts,
            "status": status, "button": button, "next_to_act": next_to_act, "round": round_id,
            "current_bet": current_bet, "min_raise": min_raise, "action_mask": mask,
        }

    def _action_mask(self):
        s = self._state; import numpy as np
        mask = np.zeros(7, dtype=np.int8)
        if s is None or s.next_to_act != self.hero: return mask
        i = self.hero; info = self.env.legal_actions(s)
        acts = getattr(info, "actions", [])
        owe = self.env.owed(s, i)
        has = lambda k: any(a.kind == k for a in acts)
        if owe > 0 and has(ActionType.FOLD): mask[0] = 1
        if owe == 0 and has(ActionType.CHECK): mask[1] = 1
        if owe > 0 and has(ActionType.CALL): mask[2] = 1
        for a in acts:
            if a.kind == ActionType.RAISE_TO:
                min_to = getattr(info, "min_raise_to", s.current_bet)
                max_to = getattr(info, "max_raise_to", s.current_bet)
                has_rr = getattr(info, "has_raise_right", False)
                def legal_target(t):
                    if t <= s.current_bet or t > max_to: return False
                    if has_rr:
                        req = (s.current_bet + s.min_raise) if s.current_bet > 0 else max(s.min_raise,1)
                        return (t >= req) or (t == max_to)
                    else:
                        return t == max_to
                req_min = min_to
                if legal_target(req_min): mask[3] = 1
                t2 = min(min_to + 2*s.bb, max_to)
                if legal_target(t2): mask[4] = 1
                t4 = min(min_to + 4*s.bb, max_to)
                if legal_target(t4): mask[5] = 1
                if max_to > s.current_bet and legal_target(max_to): mask[6] = 1
                break
        return mask

    def _map_action(self, action_idx: int) -> Action:
        s = self._state; assert s is not None
        info = self.env.legal_actions(s)
        min_to = getattr(info, "min_raise_to", s.current_bet)
        max_to = getattr(info, "max_raise_to", s.current_bet)
        if action_idx == 0: return Action(ActionType.FOLD)
        if action_idx == 1: return Action(ActionType.CHECK)
        if action_idx == 2: return Action(ActionType.CALL)
        if action_idx == 3: target = min_to
        elif action_idx == 4: target = min(min_to + 2*s.bb, max_to)
        elif action_idx == 5: target = min(min_to + 4*s.bb, max_to)
        elif action_idx == 6: target = max_to
        else: raise ValueError("Invalid action index")
        return Action(ActionType.RAISE_TO, amount=int(target))


if __name__ == "__main__":
    env = NLHEGymEnv(hero_seat=0, seed=42)
    env.reset()
    print(env.render())