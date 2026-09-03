from crematorium import (
    StepOutput,
    Tensor,
    clamp_positive,
    clamp_unit_interval,
)
from crematorium.snn import Reset, SnnModule


class ALIF(SnnModule):
    """
    Adaptive leaky integrate-and-fire neuron (Bellec et al., 2020).

    A spike-triggered adaptation trace raises the effective threshold after
    each spike and decays back with ``tau_a``, giving the neuron a memory
    longer than its membrane time constant.

    Math:
        thr = threshold + beta_a * adapt
        mem = beta * mem + x
        spk = spike_grad(mem - thr)
        mem = mem - spk * threshold
        adapt = tau_a * adapt + spk
    """

    class Params:
        beta = SnnModule.Param(
            default=0.9,
            constraint=clamp_unit_interval,
        )
        threshold = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        beta_a = SnnModule.Param(
            default=0.5,
            constraint=clamp_positive,
        )
        tau_a = SnnModule.Param(
            default=0.9,
            constraint=clamp_unit_interval,
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
        beta = self.constrain("beta")
        threshold = self.constrain("threshold")
        beta_a = self.constrain("beta_a")
        tau_a = self.constrain("tau_a")

        mem = beta * mem + x
        spk = self.spike_grad(mem - (threshold + beta_a * adapt))
        adapt = tau_a * adapt + spk

        return spk, mem, adapt
