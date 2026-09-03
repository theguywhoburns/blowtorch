from pyrokinesis import (
    StepOutput,
    Tensor,
    clamp_positive,
    clamp_unit_interval,
)
from pyrokinesis.snn import Reset, SnnModule


class LIF(SnnModule):
    """
    Leaky integrate-and-fire neuron.

    Math:
        mem = beta * mem + x
        spk = spike_grad(mem - threshold)
        mem = mem - spk * threshold
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

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.subtract("threshold"))

    def _step(self, x: Tensor, mem: Tensor) -> StepOutput:
        beta = self.constrain("beta")
        threshold = self.constrain("threshold")

        mem = beta * mem + x
        spk = self.spike_grad(mem - threshold)

        return spk, mem