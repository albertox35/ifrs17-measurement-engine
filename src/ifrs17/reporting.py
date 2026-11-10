"""Analisis de movimientos del parrafo 101.

El parrafo 101 exige una conciliacion de apertura a cierre de los saldos de
contratos de seguro, separando los componentes. Es el estado que un auditor lee
primero, y es la salida que hace util el motor: una cifra de CSM sin su
conciliacion no es verificable.

La funcion :func:`movement_analysis` devuelve esa conciliacion como
``DataFrame``, y :func:`check_reconciliation` verifica que cuadre linea por
linea en lugar de confiar en que lo haga.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .gmm import GMMResult
from .paa import PAAResult

__all__ = [
    "ReconciliationReport",
    "movement_analysis",
    "paa_movement_analysis",
    "check_reconciliation",
]


@dataclass(frozen=True)
class ReconciliationReport:
    """Resultado de verificar la conciliacion."""

    reconciles: bool
    max_absolute_error: float
    failing_periods: tuple[int, ...]

    def __bool__(self) -> bool:
        return self.reconciles


def movement_analysis(result: GMMResult) -> pd.DataFrame:
    """Analisis de movimientos de una medicion GMM.

    Las columnas siguen la estructura del parrafo 101: apertura, los movimientos
    que la explican, y cierre, para cada uno de los tres componentes (CSM,
    ajuste por riesgo y componente de perdida).
    """
    filas = [
        {
            "periodo": m.period,
            "csm_apertura": m.csm_opening,
            "csm_interes_devengado": m.csm_interest_accretion,
            "csm_cambio_servicio_futuro": m.csm_change_future_service,
            "csm_liberacion": m.csm_release,
            "csm_cierre": m.csm_closing,
            "ra_apertura": m.risk_adjustment_opening,
            "ra_liberacion": m.risk_adjustment_release,
            "ra_cierre": m.risk_adjustment_closing,
            "componente_perdida_apertura": m.loss_component_opening,
            "componente_perdida_movimiento": m.loss_component_movement,
            "componente_perdida_cierre": m.loss_component_closing,
            "siniestros_y_gastos_esperados": m.expected_claims_and_expenses,
            "ingreso_de_seguro": m.insurance_revenue,
            "gasto_servicio_seguro": m.insurance_service_expense,
            "resultado_servicio_seguro": m.insurance_service_result,
        }
        for m in result.movements
    ]
    return pd.DataFrame(filas).set_index("periodo")


def paa_movement_analysis(result: PAAResult) -> pd.DataFrame:
    """Analisis de movimientos del pasivo por cobertura restante bajo PAA."""
    filas = [
        {
            "periodo": p.period,
            "lrc_apertura": p.lrc_opening,
            "primas_recibidas": p.premiums_received,
            "amortizacion_adquisicion": p.acquisition_amortisation,
            "ingreso_reconocido": p.revenue_recognised,
            "lrc_cierre": p.lrc_closing,
            "siniestros_incurridos": p.claims_incurred,
            "resultado_servicio_seguro": p.insurance_service_result,
        }
        for p in result.periods
    ]
    return pd.DataFrame(filas).set_index("periodo")


def check_reconciliation(result: GMMResult, tol: float = 1e-7) -> ReconciliationReport:
    """Verifica la conciliacion del analisis de movimientos.

    Comprueba dos cosas distintas:

    1. Que cada linea cuadre: apertura mas movimientos igual a cierre.
    2. Que el cierre de un periodo sea la apertura del siguiente, que es la
       condicion de continuidad que un error de arrastre rompe.
    """
    fallos: list[int] = []
    error_max = 0.0
    movimientos = result.movements

    for m in movimientos:
        esperado = (
            m.csm_opening + m.csm_interest_accretion + m.csm_change_future_service - m.csm_release
        )
        error = abs(esperado - m.csm_closing)
        error_max = max(error_max, error)
        if error >= tol:
            fallos.append(m.period)

    for anterior, siguiente in zip(movimientos, movimientos[1:]):
        error = abs(anterior.csm_closing - siguiente.csm_opening)
        error_max = max(error_max, error)
        if error >= tol:
            fallos.append(siguiente.period)

    return ReconciliationReport(
        reconciles=not fallos,
        max_absolute_error=error_max,
        failing_periods=tuple(sorted(set(fallos))),
    )
