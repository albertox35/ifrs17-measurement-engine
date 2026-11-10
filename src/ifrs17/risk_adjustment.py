"""Ajuste por riesgo no financiero.

IFRS 17 exige un ajuste por riesgo (parrafo 37) pero **no prescribe el metodo**,
solo que se divulgue el nivel de confianza equivalente (parrafo 119). Las dos
practicas dominantes son el enfoque de percentil y el de costo de capital.

Aqui se implementa el de **costo de capital**, que es el que se alinea con el
margen de riesgo de Solvencia II y con la CUSF, mas una utilidad para traducir el
resultado a nivel de confianza equivalente, que es lo que la norma pide divulgar.

El capital requerido se aproxima como una fraccion de los flujos pendientes. Es
un proxy declarado, no un calculo de capital economico: un modelo de capital
real exige la estructura de riesgos completa, que esta fuera del alcance de este
repositorio.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve

__all__ = ["RiskAdjustmentConfig", "cost_of_capital_risk_adjustment", "equivalent_confidence_level"]


@dataclass(frozen=True)
class RiskAdjustmentConfig:
    """Parametros del ajuste por riesgo por costo de capital.

    Parameters
    ----------
    cost_of_capital_rate:
        Tasa de costo de capital anual. Solvencia II usa 6 % para el margen de
        riesgo; es un punto de partida habitual.
    capital_factor:
        Fraccion de los flujos pendientes que se toma como capital requerido.
    """

    cost_of_capital_rate: float = 0.06
    capital_factor: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 <= self.cost_of_capital_rate <= 1.0:
            raise ValueError("cost_of_capital_rate debe estar en [0, 1]")
        if not 0.0 <= self.capital_factor <= 1.0:
            raise ValueError("capital_factor debe estar en [0, 1]")


def required_capital_profile(
    cash_flows: FulfilmentCashFlows, config: RiskAdjustmentConfig
) -> np.ndarray:
    """Perfil de capital requerido, como proxy de los flujos pendientes.

    ``capital[t]`` es la fraccion ``capital_factor`` de los siniestros y gastos
    que restan por pagar desde el periodo ``t`` inclusive.
    """
    gross = cash_flows.claims + cash_flows.expenses
    remaining = np.cumsum(gross[::-1])[::-1]
    return config.capital_factor * remaining


def cost_of_capital_risk_adjustment(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    config: RiskAdjustmentConfig | None = None,
    from_period: int = 0,
) -> float:
    """Ajuste por riesgo por costo de capital.

    .. math::

        RA = \\sum_{t} \\mathrm{CoC} \\cdot K_t \\cdot v(t)

    Parameters
    ----------
    cash_flows:
        Flujos de cumplimiento del grupo.
    curve:
        Curva de descuento.
    config:
        Parametros; si es ``None`` se usan los valores por omision.
    from_period:
        Primer periodo a considerar. Permite reevaluar el ajuste por riesgo de la
        cobertura restante en cada cierre, que es lo que alimenta su liberacion
        en el roll-forward.

    Returns
    -------
    float
        Ajuste por riesgo, siempre no negativo.
    """
    cfg = config or RiskAdjustmentConfig()
    capital = required_capital_profile(cash_flows, cfg)[from_period:]
    if capital.size == 0:
        return 0.0
    df = curve.discount_factors(capital.size)
    return float(cfg.cost_of_capital_rate * np.dot(capital, df))


def equivalent_confidence_level(
    risk_adjustment: float,
    best_estimate: float,
    coefficient_of_variation: float,
) -> float:
    """Nivel de confianza equivalente al ajuste por riesgo.

    El parrafo 119 obliga a divulgarlo cuando el ajuste no se calcula por
    percentil. Bajo el supuesto de que los flujos son lognormales con el
    coeficiente de variacion dado, se despeja el percentil cuyo exceso sobre la
    mejor estimacion iguala el ajuste por riesgo.

    Returns
    -------
    float
        Nivel de confianza en ``(0, 1)``, o ``nan`` si los argumentos no
        permiten despejarlo.
    """
    if best_estimate <= 0.0 or coefficient_of_variation <= 0.0:
        return float("nan")
    if risk_adjustment <= 0.0:
        return 0.5

    sigma = float(np.sqrt(np.log(1.0 + coefficient_of_variation**2)))
    mu = float(np.log(best_estimate) - 0.5 * sigma**2)
    target = best_estimate + risk_adjustment
    return float(stats.lognorm.cdf(target, s=sigma, scale=np.exp(mu)))
