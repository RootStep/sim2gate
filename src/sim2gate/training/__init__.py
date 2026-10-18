"""Training-time diagnostics. Needs torch; BSRSPPO also needs rsl-rl-lib (pip install "sim2gate[rslrl]")."""
from .signals import rollout_signals, tail_predictor, RolloutSignals
from .monitor import TrainingMonitor


def __getattr__(name):
    if name == "BSRSPPO":
        from .bsrs_ppo import BSRSPPO
        return BSRSPPO
    if name == "EvidenceRecorder":
        from .evidence import EvidenceRecorder
        return EvidenceRecorder
    raise AttributeError(name)
