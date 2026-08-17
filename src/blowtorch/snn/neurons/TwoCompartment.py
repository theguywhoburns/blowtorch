from blowtorch.base import (
    StepOutput,
    Tensor,
    clamp_positive,
)
from blowtorch.snn import Reset, SnnModule
from blowtorch.util import positive


class TwoCompartment(SnnModule):
    """
    Two-compartment neuron with a coupled soma and dendrite.

    Both compartments leak toward the reversal potential ``EL``; a coupling
    conductance ``g_c`` exchanges current between them. Only the soma receives
    the input current and produces the output spike. ``dt`` is in arbitrary
    time units; keep it small relative to the time constants.

    Math:
        v_soma += (gL * (EL - v_soma) + g_c * (v_dend - v_soma) + x) / C * dt
        v_dend += (gL * (EL - v_dend) + g_c * (v_soma - v_dend)) / C * dt
        spk = spike_grad(v_soma - threshold)
        v_soma -= spk * threshold
    """

    dt = SnnModule.Constant(default=0.01, validate=positive, dtype=float)

    class Params:
        gL = SnnModule.Param(
            default=0.1,
            constraint=clamp_positive,
        )
        g_c = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        C = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        EL = SnnModule.Param(
            default=0.0,
        )
        threshold = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        v_soma = SnnModule.StateSpec(reset=Reset.subtract("threshold"))
        v_dend = SnnModule.StateSpec()

    def _step(
        self,
        x: Tensor,
        v_soma: Tensor,
        v_dend: Tensor,
    ) -> StepOutput:
        gL, g_c, C, EL, threshold = self.constrained()
        dt = self.dt

        # Both updates read the pre-step potentials: each state is a pure
        # function of the step inputs, independent of the other's update.
        v_soma_new = v_soma + (
            gL * (EL - v_soma) + g_c * (v_dend - v_soma) + x
        ) / C * dt
        v_dend_new = v_dend + (
            gL * (EL - v_dend) + g_c * (v_soma - v_dend)
        ) / C * dt
        spk = self.spike_grad(v_soma_new - threshold)

        return spk, v_soma_new, v_dend_new