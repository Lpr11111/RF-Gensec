"""Pure tensor objectives shared by RF-Gensec policies and tests."""

import torch


def adversarial_reward(av_risk, environment_reward):
    """Equal-weight RF-Gensec objective: AV risk plus FREA environment reward."""
    if av_risk.shape != environment_reward.shape:
        raise ValueError("av_risk and environment_reward must have the same shape")
    return av_risk + environment_reward


def normalise_advantage(value):
    return (value - value.mean()) / (value.std(dim=0) + 1e-5)


def hard_gate_advantages(reward_adv, feasibility_adv, value, next_value,
                         restoration_weight=1.0):
    """A0 post-transition hard gate from the original RF-Gensec method.

    Implements the paper's indicator form directly:

        I(s, s') = 1(s in S_f and s' in S_f)
        A_total  = I * A_adv + eta * (1 - I) * A_safe

    ``restoration_weight`` is the paper's eta. It is accepted here so that the
    hard and soft gates apply the same restoration scaling; without it a sweep
    over eta would silently affect only the soft-gate variants and make the
    A0/A1 vs A2/A3 ablation unfair.
    """
    if restoration_weight < 0:
        raise ValueError("restoration_weight must be non-negative")
    safe = value <= 0.0
    next_safe = next_value <= 0.0
    advantages = torch.where(
        safe & next_safe,
        normalise_advantage(reward_adv),
        restoration_weight * normalise_advantage(feasibility_adv),
    )
    return advantages, safe, next_safe


def soft_gate_advantages(reward_adv, feasibility_adv, value, next_value,
                         temperature=0.1, restoration_weight=1.0):
    """Blend adversarial/restoration advantages around the learned boundary.

    The hard gate selects exactly one standardised advantage, so its output has
    unit scale. A convex blend does not: for standardised, roughly uncorrelated
    inputs, ``std(g * A1 + (1 - g) * A2) = sqrt(g^2 + (1 - g)^2) <= 1``, which
    reaches 0.71 at ``g = 0.5``. Left uncorrected, any soft-gate run would show
    systematically smaller policy gradients than a hard-gate run purely from
    that shrinkage, confounding gradient-norm comparisons between A0/A1 and
    A2/A3 with the actual effect of the gate.

    Re-standardising the mixture keeps the two gate types on a common scale, so
    a measured gradient difference reflects the gate's direction rather than its
    magnitude.
    """
    if temperature <= 0:
        raise ValueError("gate temperature must be positive")
    if restoration_weight < 0:
        raise ValueError("restoration_weight must be non-negative")
    boundary_value = torch.maximum(value, next_value)
    adversarial_weight = torch.sigmoid(-boundary_value / temperature)
    mixture = (
        adversarial_weight * normalise_advantage(reward_adv)
        + restoration_weight * (1.0 - adversarial_weight)
        * normalise_advantage(feasibility_adv)
    )
    return normalise_advantage(mixture), adversarial_weight
