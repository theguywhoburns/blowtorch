from blowtorch import (
    StepOutput,
    Tensor,
    clamp_positive,
)
from blowtorch.snn import SnnModule
from blowtorch.util import positive


class HH(SnnModule):
    """
    Hodgkin-Huxley neuron.

    Voltages are absolute millivolts (``ENa=+50``, ``EK=-77``, leak reversal
    ``EL=-54.4``); with the default conductances the resting potential is
    ``-65`` mV. State is ``(mem, m, h, n)`` with the gating variables in
    ``[0, 1]``.     ``dt`` is the integration step; ``substeps`` subdivides it for
    stability in long rollouts (each substep integrates ``dt/substeps``).

    Math:
        INa = gNa * m^3 * h * (mem - ENa)
        IK = gK * n^4 * (mem - EK)
        IL = gL * (mem - EL)
        mem += (x - INa - IK - IL) / C * dt
        dm/dt = am * (1 - m) - bm * m   (am, bm = rate functions of mem)
        spk = spike_grad(mem - threshold)
    """

    dt = SnnModule.Constant(default=0.01, dtype=float)
    substeps = SnnModule.Constant(
        default=1,
        validate=positive,
        dtype=int,
    )
    # Ceiling for the six rate functions. A rate above this saturates its gate
    # to an endpoint within one ``dt`` anyway, and a finite cap keeps the gate
    # updates free of ``inf * 0 -> nan`` under extreme membrane potentials.
    rate_cap = SnnModule.Constant(
        default=1e4,
        validate=positive,
        dtype=float,
    )

    class Params:
        gNa = SnnModule.Param(
            default=120.0,
            constraint=clamp_positive,
        )
        gK = SnnModule.Param(
            default=36.0,
            constraint=clamp_positive,
        )
        gL = SnnModule.Param(
            default=0.3,
            constraint=clamp_positive,
        )
        ENa = SnnModule.Param(
            default=50.0,
        )
        EK = SnnModule.Param(
            default=-77.0,
        )
        EL = SnnModule.Param(
            default=-54.4,
        )
        C = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        threshold = SnnModule.Param(
            default=0.0,
        )

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(default=-65.0)
        m = SnnModule.StateSpec(default=0.0529)
        h = SnnModule.StateSpec(default=0.5961)
        n = SnnModule.StateSpec(default=0.3177)

    def _hh_rate(self, x: Tensor, a: float, c: float) -> Tensor:
        """
        Stable HH forward rate ``a * x / (1 - exp(-x / c))``.

        Uses the analytical limit ``a * c`` when ``x`` is near zero. The
        denominator is replaced by 1 inside the mask so both ``where`` branches
        stay numerically safe in the backward pass. ``safe_exp`` keeps the
        exponential finite for any membrane potential.
        """
        mask = x.abs() < 1e-4
        d = (1.0 - self.safe_exp(-x / c)).where(~mask, 1.0)
        return (a * x / d).where(~mask, a * c)

    def _step(
        self,
        x: Tensor,
        mem: Tensor,
        m: Tensor,
        h: Tensor,
        n: Tensor,
    ) -> StepOutput:
        gNa, gK, gL, ENa, EK, EL, C, threshold = self.constrained()
        dt = self.dt / self.substeps
        cap = self.rate_cap

        for _ in range(self.substeps):
            INa = gNa * (m ** 3) * h * (mem - ENa)
            IK = gK * (n ** 4) * (mem - EK)
            IL = gL * (mem - EL)
            mem = mem + (x - INa - IK - IL) / C * dt

            am = self._hh_rate(mem + 40, 0.1, 10.0).clamp(max=cap)
            bm = (4.0 * self.safe_exp(-(mem + 65) / 18)).clamp(max=cap)
            ah = (0.07 * self.safe_exp(-(mem + 65) / 20)).clamp(max=cap)
            bh = 1.0 / (1 + self.safe_exp(-(mem + 35) / 10))
            an = self._hh_rate(mem + 55, 0.01, 10.0).clamp(max=cap)
            bn = (0.125 * self.safe_exp(-(mem + 65) / 80)).clamp(max=cap)

            m = (m + (am * (1 - m) - bm * m) * dt).clamp(0.0, 1.0)
            h = (h + (ah * (1 - h) - bh * h) * dt).clamp(0.0, 1.0)
            n = (n + (an * (1 - n) - bn * n) * dt).clamp(0.0, 1.0)

        spk = self.spike_grad(mem - threshold)

        return spk, mem, m, h, n