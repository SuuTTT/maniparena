"""ManipArena — adversarial market simulation as a manipulation benchmark (track sibling of Warn2Trade / CausalGate).

Package layout (all new files; nothing under warn2trade/ or causalgate/ is rewritten):
    sim/     order book matching engine, background agents, discrete-event market, calibration, P&L fidelity
    manip/   manipulation families (spoofing, layering, pump-and-dump, momentum ignition, quote stuffing) with
             difficulty levels L0 (published template) .. L3 (RL-adapted against the co-trained detector)
    detect/  feature pipeline (event-level and bar-level), detector contract, ArenaDetector (ours)
    arena/   detector-vs-manipulator co-training loop and benchmark release builder
    data/    LOBSession / BarFrame contracts, real-data loaders D1-D10, static injection of published templates
    eval/    PR-curve metrics, event-level detection, evasion, transfer, protocols (P-S / P-R / P-A / P-H / P-T)
    baselines/ B1-B10 wrappers sharing the detector contract
"""
__version__ = "0.1.0"
