from rift.cbv.planning.fine_tuner.rlft.rlft_rf_gensec.base.trainer import LightningTrainer as PPOTrainer


class LightningTrainer(PPOTrainer):
    def forward(self, data):
        # eval disables encoder dropout without disabling pi_head gradients.
        self.model.eval()
        return self.model(data)
