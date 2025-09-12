import types
import sys

# Only create stub if nlhe_engine is not already available
if "nlhe_engine" not in sys.modules:
    try:
        import nlhe_engine
    except ImportError:
        # Provide stub for compiled evaluation module used by the engine
        stub = types.ModuleType("nlhe_engine")
        sys.modules.setdefault("nlhe_engine", stub)

from PyQt6 import QtCore

from nlhe.demo.controller import GameController
from nlhe.agents.tamed_random import TamedRandomAgent


def play_hand(controller: GameController) -> list:
    """Run the controller until a hand finishes and return rewards."""
    hero_agent = TamedRandomAgent(controller.rng)
    rewards: list | None = None

    def _capture(r: list) -> None:
        nonlocal rewards
        rewards = r

    controller.hand_finished.connect(_capture)
    while rewards is None:
        controller.play_loop()
        if controller.state.next_to_act == controller.hero_seat:
            action = hero_agent.act(controller.engine, controller.state, controller.hero_seat)
            controller.submit_action(action)
    return rewards


def test_controller_completes_hand():
    app = QtCore.QCoreApplication([])
    controller = GameController(hero_seat=0, seed=1)
    controller.timer.stop()
    rewards = play_hand(controller)
    assert len(rewards) == controller.engine.N


def test_state_changed_emitted():
    app = QtCore.QCoreApplication([])
    controller = GameController(hero_seat=0, seed=2)
    controller.timer.stop()
    states = []
    controller.state_changed.connect(lambda s: states.append(s))
    play_hand(controller)
    assert states, "state_changed should emit during play"
