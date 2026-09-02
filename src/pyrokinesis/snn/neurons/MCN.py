from __future__ import annotations

from pyrokinesis import StepOutput, Tensor, clamp_positive
from pyrokinesis.snn import Reset, SnnModule


class MCN(SnnModule):
    """
    Three-compartment neuron: basal dendrite, apical dendrite, soma.

    The basal dendrite integrates the feedforward input ``x_b``, the apical
    dendrite integrates a second, modulatory input ``x_a``, and the somatic
    potential ``u`` integrates the two dendritic potentials (paper eq. 2-4).
    The neuron fires when ``u`` crosses ``threshold``.

    Math (discrete-time / Euler form, dt = 1):

        V_b = (1 - 1/tau_B) * V_b + (1/tau_B) * x_b              # eq. (2)
        V_a = (1 - 1/tau_A) * V_a + (1/tau_A) * x_a              # eq. (3)
        u   = m * u + (gB/(gL*tau_L)) * V_b + (gA/(gL*tau_L)) * V_a
        spk = spike_grad(u - threshold)
        u  -= spk * threshold     # subtract reset, consistent with LIF

    with ``m = 1 - 1/tau_L - gB/(gL*tau_L) - gA/(gL*tau_L)`` (paper eq. 38).
    The ``u`` update reads the current-step dendritic potentials, which are
    themselves pure functions of the pre-step states; all states in a step are
    therefore functions of the pre-step states and the inputs only.

    The paper's surrogate ``d(spk)/du = 2*tau_L/(4 + (pi*tau_L*u)**2)`` is
    exactly the classic ATan surrogate with ``beta = tau_L`` (see
    ``pyrokinesis.util.atan_surrogate``); if you change ``tau_L`` from its
    default, pass ``spike_grad=atan_surrogate(tau_L)`` explicitly.

    Defaults match the paper's Fig. 3(F): ``tau_A = tau_B = 2.0``,
    ``tau_L = 4.0``, ``gA = gB = gL = 1.0``, ``threshold = 0.8``.
    """

    class Params:
        tau_B = SnnModule.Param(
            default=2.0,
            constraint=clamp_positive,
        )
        tau_A = SnnModule.Param(
            default=2.0,
            constraint=clamp_positive,
        )
        tau_L = SnnModule.Param(
            default=4.0,
            constraint=clamp_positive,
        )
        gB = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        gA = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        gL = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        threshold = SnnModule.Param(
            default=0.8,
            constraint=clamp_positive,
        )

    class Inputs:
        x_b: Tensor
        x_a: Tensor

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        V_b = SnnModule.StateSpec()
        V_a = SnnModule.StateSpec()
        u = SnnModule.StateSpec(reset=Reset.subtract("threshold"))

    def _step(
        self,
        x_b: Tensor,
        x_a: Tensor,
        V_b: Tensor,
        V_a: Tensor,
        u: Tensor,
    ) -> StepOutput:
        tau_B, tau_A, tau_L, gB, gA, gL, threshold = self.constrained()

        V_b_new = (1 - 1 / tau_B) * V_b + (1 / tau_B) * x_b
        V_a_new = (1 - 1 / tau_A) * V_a + (1 / tau_A) * x_a

        m = 1 - 1 / tau_L - gB / (gL * tau_L) - gA / (gL * tau_L)
        u_new = m * u + (gB / (gL * tau_L)) * V_b_new + (gA / (gL * tau_L)) * V_a_new

        spk = self.spike_grad(u_new - threshold)

        return spk, V_b_new, V_a_new, u_new