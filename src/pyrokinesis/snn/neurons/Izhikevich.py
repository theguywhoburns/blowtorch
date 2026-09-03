from pyrokinesis import (
    StepOutput,
    Tensor,
)
from pyrokinesis.snn import Reset, SnnModule
from pyrokinesis.util import positive


class Izhikevich(SnnModule):
    """
    Izhikevich spiking neuron (2003).

    Two-dimensional system: membrane ``v`` and recovery variable ``u``.
    Voltages are absolute millivolts and time is in milliseconds (``dt``).
    Fires when ``v`` crosses ``v_peak`` (default 30 mV); on a spike ``v``
    resets to ``c`` and ``u`` increases by ``d``.

    Defaults reproduce regular spiking (RS): ``a=0.02, b=0.2, c=-65, d=8``.
    Fast spiking uses ``a=0.1, b=0.2, c=-65, d=2``.

    Math:
        v += (0.04 * v^2 + 5 * v + 140 - u + x) * dt
        u += (a * (b * v - u)) * dt
        spk = spike_grad(v - v_peak)
        v -> c, u -> u + d   (on spike)
    """

    dt = SnnModule.Constant(default=1.0, validate=positive, dtype=float)

    class Params:
        a = SnnModule.Param(default=0.02)
        b = SnnModule.Param(default=0.2)
        c = SnnModule.Param(default=-65.0)
        d = SnnModule.Param(default=8.0)
        # Peak voltage was hardcoded to 30 mV before v1; every other constant
        # in the library is a declared Param, so this one is too now.
        v_peak = SnnModule.Param(default=30.0)

    class Specs:
        spk = SnnModule.OutputSpec(differentiable=False)
        v = SnnModule.StateSpec(default=-65.0, reset=Reset.set("c"))
        u = SnnModule.StateSpec(default=-13.0, reset=Reset.add("d"))

    def _step(
        self,
        x: Tensor,
        v: Tensor,
        u: Tensor,
    ) -> StepOutput:
        # Name-based access: inserting a Param must not shift values into the
        # wrong variable (positional unpacking is load-bearing otherwise).
        p = self.constrained_named()
        a, b, dt = p["a"], p["b"], self.dt

        # Both updates read the pre-step potentials: each state is a pure
        # function of the step inputs, independent of the other's update.
        v_new = v + (0.04 * v ** 2 + 5 * v + 140 - u + x) * dt
        u_new = u + (a * (b * v - u)) * dt
        spk = self.spike_grad(v_new - p["v_peak"])

        return spk, v_new, u_new