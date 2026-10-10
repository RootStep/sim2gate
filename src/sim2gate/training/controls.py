"""Negative controls for Sim2Gate's own acceptance checks. Never use for training results."""
from .bsrs_ppo import BSRSPPO


class IgnoresEtaBSRSPPO(BSRSPPO):
    """Claims a nonzero eta (so the monitor treats it as BSRS training) but trains on stock PPO's advantages.
    The smoke test runs it and requires the trainer-match check to catch it."""

    def compute_returns(self, obs):
        return super(BSRSPPO, self).compute_returns(obs)
