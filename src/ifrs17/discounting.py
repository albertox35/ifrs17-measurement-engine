"""Curvas de descuento y tasas bloqueadas.

IFRS 17 distingue dos tasas que se confunden con facilidad:

- La **tasa corriente**, con la que se miden los flujos de cumplimiento en cada
  cierre.
- La **tasa bloqueada en el reconocimiento inicial**, con la que se devenga
  interes sobre el CSM (parrafo B72(b)).

Mantener las dos es obligatorio: si el CSM se devengara a tasa corriente, los
movimientos de tasa entrarian en el resultado por la via equivocada. La clase
:class:`DiscountCurve` es inmutable justamente para que una curva bloqueada no
pueda mutarse por accidente.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DiscountCurve"]


@dataclass(frozen=True)
class DiscountCurve:
    """Curva cupon cero anual en tiempo discreto.

    Parameters
    ----------
    spot_rates:
        Tasas spot anuales, una por periodo. ``spot_rates[t]`` aplica al flujo
        que ocurre al final del periodo ``t`` (indexado desde cero).
    """

    spot_rates: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.spot_rates, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("spot_rates debe ser un vector no vacio")
        if np.any(arr <= -1.0):
            raise ValueError("una tasa spot menor o igual a -100% no es admisible")
        object.__setattr__(self, "spot_rates", arr)

    @classmethod
    def flat(cls, rate: float, n_periods: int) -> DiscountCurve:
        """Curva plana, util para pruebas con solucion cerrada."""
        return cls(np.full(n_periods, float(rate)))

    @property
    def n_periods(self) -> int:
        """Numero de periodos cubiertos por la curva."""
        return int(self.spot_rates.size)

    def discount_factors(self, n_periods: int | None = None) -> np.ndarray:
        """Factores de descuento al final de cada periodo.

        ``df[t] = (1 + s_t)^{-(t+1)}``, es decir el flujo del periodo ``t`` se
        descuenta ``t + 1`` periodos. La convencion de fin de periodo se mantiene
        en todo el motor.
        """
        n = self.n_periods if n_periods is None else n_periods
        if n > self.n_periods:
            raise ValueError(
                f"la curva cubre {self.n_periods} periodos, se piden {n}; "
                "extender la curva explicitamente en lugar de extrapolar en silencio"
            )
        exponents = np.arange(1, n + 1)
        return (1.0 + self.spot_rates[:n]) ** (-exponents)

    def present_value(self, cash_flows: np.ndarray) -> float:
        """Valor presente de un vector de flujos de fin de periodo."""
        cf = np.asarray(cash_flows, dtype=float)
        return float(np.dot(cf, self.discount_factors(cf.size)))

    def forward_rate(self, period: int) -> float:
        """Tasa forward de un periodo, implicita en la curva spot."""
        if period == 0:
            return float(self.spot_rates[0])
        df = self.discount_factors(period + 1)
        return float(df[period - 1] / df[period] - 1.0)
