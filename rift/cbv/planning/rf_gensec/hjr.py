"""Risk-bounded Hamilton-Jacobi reachability critic for RF-Gensec."""

from pathlib import Path

import torch
import torch.nn as nn

from .networks import CriticPPO, CriticTwin


class RiskHJR(nn.Module):
    """Offline safety critic migrated from FREA's ``HJ_Reachability_risk``."""

    BARRIER_TYPES = ("step", "continuous")

    def __init__(self, state_dim=18, action_dim=2, dims=(256, 256), gamma=0.98,
                 tau=5e-3, expectile=0.9, risk_threshold=0.345, penalty=4.0,
                 barrier_type="step", barrier_delta=0.1):
        super().__init__()
        if barrier_type not in self.BARRIER_TYPES:
            raise ValueError(f"barrier_type must be one of {self.BARRIER_TYPES}, got {barrier_type!r}")
        if barrier_delta <= 0:
            raise ValueError("barrier_delta must be positive")
        self.gamma, self.tau, self.expectile = gamma, tau, expectile
        self.risk_threshold, self.penalty = risk_threshold, penalty
        self.barrier_type, self.barrier_delta = barrier_type, barrier_delta
        self.q = CriticTwin(list(dims), state_dim, action_dim)
        self.q_target = CriticTwin(list(dims), state_dim, action_dim)
        self.v = CriticPPO(list(dims), state_dim, action_dim)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.requires_grad_(False)

    def feasibility_value(self, state: torch.Tensor) -> torch.Tensor:
        return self.v(state.reshape(state.shape[0], -1))

    def barrier_value(self, total_risk: torch.Tensor) -> torch.Tensor:
        """Return the configured A0 step or A1 continuous bounded barrier."""
        if self.barrier_type == "continuous":
            return ((total_risk - self.risk_threshold) / self.barrier_delta).clamp(
                min=-1.0, max=self.penalty)
        # Strict '>' matches the paper's Eq. (7): h(s) = -1 for R <= R_thre and
        # h(s) = M for R > R_thre, so h(s) > 0 denotes a threshold violation.
        return torch.where(
            total_risk > self.risk_threshold,
            torch.as_tensor(self.penalty, dtype=total_risk.dtype, device=total_risk.device),
            torch.as_tensor(-1.0, dtype=total_risk.dtype, device=total_risk.device),
        )

    def bellman_target(self, hazard, next_value, done):
        """Discounted obstacle target used by the migrated FREA implementation."""
        target = ((1.0 - self.gamma) * hazard
                  + self.gamma * torch.maximum(hazard, next_value))
        return target * (1.0 - done) + hazard * done

    @staticmethod
    def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
        weight = torch.where(diff < 0, expectile, 1.0 - expectile)
        return (weight * diff.square()).mean()

    def losses(self, state, action, next_state, done, total_risk):
        state, next_state = state.flatten(1), next_state.flatten(1)
        done, total_risk = done.reshape(-1), total_risk.reshape(-1)
        hazard = self.barrier_value(total_risk)
        with torch.no_grad():
            target_v = self.v(next_state)
            target = self.bellman_target(hazard, target_v, done)
            target_q = self.q_target.get_q_max(state, action)
        q1, q2 = self.q.get_q1_q2(state, action)
        q_loss = nn.functional.mse_loss(q1, target) + nn.functional.mse_loss(q2, target)
        v_loss = self.expectile_loss(target_q - self.v(state), self.expectile)
        return q_loss, v_loss

    @torch.no_grad()
    def soft_update(self):
        for target, source in zip(self.q_target.parameters(), self.q.parameters()):
            target.lerp_(source, self.tau)

    def checkpoint(self, step=None, q_optimizer=None, v_optimizer=None):
        payload = {
            "Qh_net": self.q.state_dict(),
            "Vh_net": self.v.state_dict(),
            "step": step,
            "rf_gensec": {
                "risk_threshold": self.risk_threshold,
                "penalty": self.penalty,
                "gamma": self.gamma,
                "expectile": self.expectile,
                "barrier_type": self.barrier_type,
                "barrier_delta": self.barrier_delta,
            },
        }
        if q_optimizer is not None:
            payload["Qh_optim"] = q_optimizer.state_dict()
        if v_optimizer is not None:
            payload["Vh_optim"] = v_optimizer.state_dict()
        return payload

    def load_checkpoint(self, path, load_q=True, strict=True):
        """Load a RIFT or legacy FREA HJR checkpoint.

        Online RF-Gensec only consumes ``Vh_net`` and may set ``load_q=False``.
        This also permits a legacy FREA checkpoint whose action dimension differs
        from the CBV policy action dimension.
        """
        checkpoint = torch.load(Path(path), map_location="cpu")
        metadata = checkpoint.get("rf_gensec", {})
        saved_barrier = metadata.get("barrier_type")
        if saved_barrier is not None and saved_barrier != self.barrier_type:
            raise ValueError(
                f"checkpoint barrier_type={saved_barrier!r} does not match "
                f"model barrier_type={self.barrier_type!r}")
        saved_delta = metadata.get("barrier_delta")
        if (self.barrier_type == "continuous" and saved_delta is not None
                and abs(float(saved_delta) - self.barrier_delta) > 1e-12):
            raise ValueError(
                f"checkpoint barrier_delta={saved_delta} does not match "
                f"model barrier_delta={self.barrier_delta}")
        if "Vh_net" not in checkpoint:
            raise KeyError(f"{path} does not contain Vh_net")
        self.v.load_state_dict(checkpoint["Vh_net"], strict=strict)
        if load_q:
            if "Qh_net" not in checkpoint:
                raise KeyError(f"{path} does not contain Qh_net")
            self.q.load_state_dict(checkpoint["Qh_net"], strict=strict)
            self.q_target.load_state_dict(checkpoint["Qh_net"], strict=strict)
        return checkpoint
