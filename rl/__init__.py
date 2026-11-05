"""CAT's closed-loop adversarial training without MetaDrive: a batched
log-replay simulator in plain torch (sim.py) over CAT's scenes (scenes.py),
CAT's adversary selection batched over worlds (adversary.py), the episode
geometry (geometry.py) and TD3 (td3.py). Everything runs in one Python 3.11
environment (CAT-K's), on CPU or GPU; training is scripts/rl/train_td3.py.
"""
