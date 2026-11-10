"""Enfoque de asignacion de prima (PAA, parrafos 53-59).

El PAA es una simplificacion del GMM, no un modelo alternativo libre: solo se
puede aplicar si el grupo **cumple el test de elegibilidad**, y ese test es la
parte que se omite con mas frecuencia. Un grupo es elegible si la cobertura no
excede un anio, o si se puede demostrar que el PAA no produce una medicion
materialmente distinta de la del GMM.

Por eso :func:`paa_eligibility` devuelve el motivo junto con el veredicto, y
:func:`measure_paa` exige haber pasado el test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve
from .gmm import initial_recognition
from .risk_adjustment import RiskAdjustmentConfig

__all__ = ["PAAEligibility", "PAAPeriod", "PAAResult", "paa_eligibility", "measure_paa"]


@dataclass(frozen=True)
class PAAEligibility:
    """Veredicto del test de elegibilidad del PAA."""

    eligible: bool
    reason: str
    gmm_liability: float | None = None
    paa_liability: float | None = None
    relative_difference: float | None = None


@dataclass(frozen=True)
class PAAPeriod:
    """Una linea del roll-forward del pasivo por cobertura restante."""

    period: int
    lrc_opening: float
    premiums_received: float
    acquisition_amortisation: float
    revenue_recognised: float
    lrc_closing: float
    claims_incurred: float

    @property
    def insurance_service_result(self) -> float:
        """Resultado del servicio de seguro del periodo."""
        return self.revenue_recognised - self.claims_incurred - self.acquisition_amortisation

    def reconciles(self, tol: float = 1e-7) -> bool:
        """Comprueba que la linea cuadra."""
        esperado = (
            self.lrc_opening
            + self.premiums_received
            - self.revenue_recognised
            - self.acquisition_amortisation
        )
        return abs(esperado - self.lrc_closing) < tol


@dataclass(frozen=True)
class PAAResult:
    """Resultado de la medicion por PAA."""

    eligibility: PAAEligibility
    periods: list[PAAPeriod]

    @property
    def total_revenue(self) -> float:
        """Ingreso de seguro total reconocido."""
        return float(sum(p.revenue_recognised for p in self.periods))

    @property
    def total_insurance_service_result(self) -> float:
        """Resultado del servicio de seguro acumulado."""
        return float(sum(p.insurance_service_result for p in self.periods))

    @property
    def closing_lrc(self) -> float:
        """Pasivo por cobertura restante al final; debe ser cero."""
        return self.periods[-1].lrc_closing if self.periods else 0.0

    def reconciles(self, tol: float = 1e-7) -> bool:
        """``True`` si todas las lineas cuadran."""
        return all(p.reconciles(tol) for p in self.periods)


def paa_eligibility(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    materiality: float = 0.05,
) -> PAAEligibility:
    """Evalua si un grupo es elegible para el PAA.

    Dos vias, en el orden del parrafo 53:

    1. **Cobertura de un periodo o menos**: elegible de forma automatica.
    2. **No diferencia material frente al GMM**: se mide el grupo por los dos
       enfoques y se compara el pasivo inicial.

    Parameters
    ----------
    materiality:
        Umbral relativo por encima del cual la diferencia se considera material.

    Returns
    -------
    PAAEligibility
        Con el motivo explicito, para que quede traza de por que se aplico el
        simplificado.
    """
    if cash_flows.n_periods <= 1:
        return PAAEligibility(
            eligible=True,
            reason="la cobertura no excede un periodo (parrafo 53(b))",
        )

    gmm = initial_recognition(cash_flows, curve, risk_config)
    pasivo_gmm = gmm.insurance_contract_liability
    # Bajo PAA el pasivo inicial es la prima cobrada menos los flujos de
    # adquisicion, sin descontar ni ajustar por riesgo.
    pasivo_paa = float(cash_flows.premiums[0] - cash_flows.acquisition_cash_flows)

    escala = max(abs(pasivo_gmm), abs(pasivo_paa), 1e-12)
    diff = abs(pasivo_gmm - pasivo_paa) / escala
    elegible = diff <= materiality
    motivo = (
        f"diferencia relativa frente al GMM de {diff:.2%}, "
        f"{'dentro' if elegible else 'por encima'} del umbral de {materiality:.0%} "
        "(parrafo 53(a))"
    )
    return PAAEligibility(
        eligible=elegible,
        reason=motivo,
        gmm_liability=pasivo_gmm,
        paa_liability=pasivo_paa,
        relative_difference=diff,
    )


def measure_paa(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    materiality: float = 0.05,
    expense_acquisition_immediately: bool = False,
    require_eligibility: bool = True,
) -> PAAResult:
    """Mide un grupo por el enfoque de asignacion de prima.

    El ingreso se reconoce segun el **paso del tiempo**, usando las unidades de
    cobertura como medida, que es la opcion por omision del parrafo B126.

    Parameters
    ----------
    expense_acquisition_immediately:
        Si ``True`` los flujos de adquisicion se reconocen de inmediato, opcion
        permitida por el parrafo 59(a) cuando la cobertura no excede un anio.
    require_eligibility:
        Si ``True`` se lanza una excepcion cuando el grupo no es elegible, en
        lugar de producir una medicion que la norma no permite.

    Returns
    -------
    PAAResult
    """
    elegibilidad = paa_eligibility(cash_flows, curve, risk_config, materiality)
    if require_eligibility and not elegibilidad.eligible:
        raise ValueError(
            f"el grupo no es elegible para el PAA: {elegibilidad.reason}. "
            "Medir por el modelo general con ifrs17.gmm.measure()."
        )

    n = cash_flows.n_periods
    cu = cash_flows.coverage_units
    patron = cu / cu.sum()

    if expense_acquisition_immediately:
        amortizacion = np.zeros(n)
        amortizacion[0] = cash_flows.acquisition_cash_flows
    else:
        amortizacion = cash_flows.acquisition_cash_flows * patron

    lrc = 0.0
    periodos: list[PAAPeriod] = []
    prima_total = float(np.sum(cash_flows.premiums))
    for t in range(n):
        apertura = lrc
        recibida = float(cash_flows.premiums[t])
        ingreso = prima_total * float(patron[t])
        amort = float(amortizacion[t])
        cierre = apertura + recibida - ingreso - amort
        periodos.append(
            PAAPeriod(
                period=t,
                lrc_opening=apertura,
                premiums_received=recibida,
                acquisition_amortisation=amort,
                revenue_recognised=ingreso,
                lrc_closing=cierre,
                claims_incurred=float(cash_flows.claims[t] + cash_flows.expenses[t]),
            )
        )
        lrc = cierre

    return PAAResult(eligibility=elegibilidad, periods=periodos)
