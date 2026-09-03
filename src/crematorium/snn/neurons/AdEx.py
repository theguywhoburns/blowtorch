from crematorium import (
    StepOutput,
    Tensor,
    clamp_positive,
)
from crematorium.snn import Reset, SnnModule


class AdEx(SnnModule):
    """
    Adaptive exponential integrate-and-fire neuron.

    Two state variables: membrane potential ``mem`` and adaptation current
    ``adapt``. Spike resets ``mem`` to ``V_reset`` (set reset) and injects ``b``
    into ``adapt`` (add reset).

    Math:
        dv = (-(mem - V_rest) + delta_T * exp((mem - V_T) / delta_T) - adapt + x) / tau_m
        dw = (a * (mem - V_rest) - adapt) / tau_w
        mem = mem + dv
        adapt = adapt + dw
        spk = spike_grad(mem - V_T)
        mem = (1 - spk) * mem + spk * V_reset
        adapt = adapt + b * spk
    """

    class Params:
        tau_m = SnnModule.Param(
            default=10.0,
            constraint=clamp_positive,
        )
        tau_w = SnnModule.Param(
            default=100.0,
            constraint=clamp_positive,
        )
        V_rest = SnnModule.Param(
            default=0.0,
        )
        V_reset = SnnModule.Param(
            default=0.0,
        )
        V_T = SnnModule.Param(
            default=1.0,
            constraint=clamp_positive,
        )
        delta_T = SnnModule.Param(
            default=0.5,
            constraint=clamp_positive,
        )
        a = SnnModule.Param(
            default=0.1,
            constraint=clamp_positive,
        )
        b = SnnModule.Param(
            default=0.2,
            constraint=clamp_positive,
        )

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        mem = SnnModule.StateSpec(reset=Reset.set("V_reset"))
        adapt = SnnModule.StateSpec(reset=Reset.add("b"))

    def _step(
        self,
        x: Tensor,
        mem: Tensor,
        adapt: Tensor,
    ) -> StepOutput:
        # Single-var access: V_reset and b are consumed by declarative resets
        # (set/add), not by the dynamics — only the needed params resolve.
        tau_m = self.constrain("tau_m")
        tau_w = self.constrain("tau_w")
        V_rest = self.constrain("V_rest")
        V_T = self.constrain("V_T")
        delta_T = self.constrain("delta_T")
        a = self.constrain("a")

        dv = (
            -(mem - V_rest)
            + delta_T * self.safe_exp((mem - V_T) / delta_T)
            - adapt
            + x
        ) / tau_m
        dw = (a * (mem - V_rest) - adapt) / tau_w
        mem = mem + dv
        adapt = adapt + dw
        spk = self.spike_grad(mem - V_T)

        return spk, mem, adapt