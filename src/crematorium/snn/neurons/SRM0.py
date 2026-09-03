from crematorium import (
    StepOutput,
    Tensor,
    clamp_positive,
)
from crematorium.snn import Reset, SnnModule
from crematorium.util import positive


class SRM0(SnnModule):
    """
    Spike response model, zeroth order (Gerstner).

    The membrane potential is a leaky PSP sum. A refractory trace ``adapt``
    accumulates spikes and raises the effective threshold, then decays with
    ``tau_ref``. ``dt`` is in milliseconds.

    Math:
        mem = mem * exp(-dt / tau_mem) + x
        thr = threshold + adapt
        spk = spike_grad(mem - thr)
        mem = mem - spk * threshold
        adapt = adapt * exp(-dt / tau_ref) + spk
    """

    dt = SnnModule.Constant(default=1.0, validate=positive, dtype=float)

    class Params:
        tau_mem = SnnModule.Param(
            default=10.0,
            constraint=clamp_positive,
        )
        tau_ref = SnnModule.Param(
            default=2.0,
            constraint=clamp_positive,
        )
        threshold = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))
        adapt = SnnModule.StateSpec()

    def _step(
        self,
        x: Tensor,
        mem: Tensor,
        adapt: Tensor,
    ) -> StepOutput:
        tau_mem = self.constrain("tau_mem")
        tau_ref = self.constrain("tau_ref")
        threshold = self.constrain("threshold")

        decay_mem = (-self.dt / tau_mem).exp()
        decay_ref = (-self.dt / tau_ref).exp()

        mem = mem * decay_mem + x
        spk = self.spike_grad(mem - (threshold + adapt))
        adapt = adapt * decay_ref + spk

        return spk, mem, adapt