"""Standalone offline trainer for the RF-Gensec reachability critic."""

from pathlib import Path

import torch
import torch.nn.utils as nn_utils


class RiskHJRTrainer:
    def __init__(self, model, dataset, output_dir, lr=3e-4, device="cpu",
                 grad_clip=0.5):
        self.model = model.to(device)
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)
        self.grad_clip = float(grad_clip)
        self.q_optimizer = torch.optim.Adam(model.q.parameters(), lr=lr, eps=1e-5)
        self.v_optimizer = torch.optim.Adam(model.v.parameters(), lr=lr, eps=1e-5)
        self.step = 0

    def load(self, path):
        payload = self.model.load_checkpoint(path, load_q=True)
        if "Qh_optim" in payload:
            self.q_optimizer.load_state_dict(payload["Qh_optim"])
        if "Vh_optim" in payload:
            self.v_optimizer.load_state_dict(payload["Vh_optim"])
        self.step = int(payload.get("step") or 0)

    def train_step(self, batch_size):
        batch = self.dataset.sample(batch_size, self.device)
        q_loss, v_loss = self.model.losses(
            batch["obs"], batch["actions"], batch["next_obs"],
            batch["dones"], batch["total_risk"])
        self.v_optimizer.zero_grad()
        v_loss.backward()
        nn_utils.clip_grad_norm_(self.model.v.parameters(), self.grad_clip)
        self.v_optimizer.step()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        nn_utils.clip_grad_norm_(self.model.q.parameters(), self.grad_clip)
        self.q_optimizer.step()
        self.model.soft_update()
        self.step += 1
        return {"q_loss": q_loss.item(), "v_loss": v_loss.item()}

    def save(self):
        path = self.output_dir / f"model.HJR.{self.step:07d}.torch"
        torch.save(self.model.checkpoint(
            self.step, self.q_optimizer, self.v_optimizer), path)
        return path
