"""Modelo general de medicion (GMM / building block approach).

El CSM es el corazon de IFRS 17 y la fuente de la mayoria de los errores de
implementacion. Tres reglas lo gobiernan, y las tres estan codificadas aqui:

1. **El CSM no puede ser negativo** (parrafo 38). Si los flujos de cumplimiento
   mas el ajuste por riesgo exceden las primas, el grupo es *oneroso*: el CSM es
   cero, la perdida se reconoce de inmediato en resultados y nace un
   **componente de perdida** que se sigue por separado (parrafos 47-52).
2. **El interes se devenga a la tasa bloqueada** en el reconocimiento inicial
   (parrafo B72(b)), no a la tasa corriente.
3. **Los cambios de supuestos que afectan servicio futuro ajustan el CSM**
   (parrafo B96); los que afectan servicio pasado o corriente van a resultados.
   Un cambio favorable en un grupo con componente de perdida primero revierte
   ese componente y solo despues restablece CSM (parrafo 50).

Un motor que no implemente estas tres reglas no esta calculando CSM.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve
from .risk_adjustment import RiskAdjustmentConfig, cost_of_capital_risk_adjustment

__all__ = ["InitialRecognition", "PeriodMovement", "GMMResult", "initial_recognition", "measure"]

_TOL = 1e-9


@dataclass(frozen=True)
class InitialRecognition:
    """Medicion en el reconocimiento inicial (parrafos 32-38)."""

    pv_future_cash_flows: float
    """Valor presente de los flujos de cumplimiento, positivo si son salida neta."""

    risk_adjustment: float
    """Ajuste por riesgo no financiero."""

    acquisition_cash_flows: float
    """Flujos de adquisicion reconocidos al inicio."""

    csm: float
    """Margen de servicio contractual; cero o positivo, nunca negativo."""

    loss_component: float
    """Componente de perdida; positivo solo si el grupo es oneroso."""

    @property
    def fulfilment_cash_flows(self) -> float:
        """Flujos de cumplimiento mas ajuste por riesgo mas adquisicion."""
        return self.pv_future_cash_flows + self.risk_adjustment + self.acquisition_cash_flows

    @property
    def is_onerous(self) -> bool:
        """``True`` si el grupo es oneroso en el reconocimiento inicial."""
        return self.loss_component > _TOL

    @property
    def insurance_contract_liability(self) -> float:
        """Pasivo por contratos de seguro al inicio."""
        return self.fulfilment_cash_flows + self.csm


@dataclass(frozen=True)
class PeriodMovement:
    """Una linea del analisis de movimientos (parrafo 101)."""

    period: int
    csm_opening: float
    csm_interest_accretion: float
    csm_change_future_service: float
    csm_release: float
    csm_closing: float
    risk_adjustment_opening: float
    risk_adjustment_release: float
    risk_adjustment_closing: float
    loss_component_opening: float
    loss_component_movement: float
    loss_component_closing: float
    expected_claims_and_expenses: float
    insurance_revenue: float
    insurance_service_expense: float

    @property
    def insurance_service_result(self) -> float:
        """Resultado del servicio de seguro del periodo."""
        return self.insurance_revenue - self.insurance_service_expense

    def reconciles(self, tol: float = 1e-7) -> bool:
        """Comprueba que la linea del roll-forward cuadra.

        Es la propiedad que hace auditable el analisis de movimientos: apertura
        mas devengo mas cambios menos liberacion debe ser exactamente el cierre.
        """
        esperado = (
            self.csm_opening
            + self.csm_interest_accretion
            + self.csm_change_future_service
            - self.csm_release
        )
        return abs(esperado - self.csm_closing) < tol


@dataclass(frozen=True)
class GMMResult:
    """Resultado completo de la medicion GMM."""

    initial: InitialRecognition
    movements: list[PeriodMovement] = field(default_factory=list)
    locked_in_rate: float = 0.0

    @property
    def total_csm_released(self) -> float:
        """CSM total liberado a resultados durante la cobertura."""
        return float(sum(m.csm_release for m in self.movements))

    @property
    def total_interest_accreted(self) -> float:
        """Interes total devengado sobre el CSM."""
        return float(sum(m.csm_interest_accretion for m in self.movements))

    @property
    def total_insurance_service_result(self) -> float:
        """Resultado del servicio de seguro acumulado."""
        return float(sum(m.insurance_service_result for m in self.movements))

    @property
    def closing_csm(self) -> float:
        """CSM al final del periodo de cobertura; debe ser cero."""
        return self.movements[-1].csm_closing if self.movements else self.initial.csm

    def reconciles(self, tol: float = 1e-7) -> bool:
        """``True`` si todas las lineas del analisis de movimientos cuadran."""
        return all(m.reconciles(tol) for m in self.movements)


def initial_recognition(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
) -> InitialRecognition:
    """Mide un grupo de contratos en el reconocimiento inicial.

    .. math::

        \\mathrm{CSM}_0 = \\max\\left(0,\\; -\\left(PV(\\mathrm{FCF})
            + RA + AC\\right)\\right)

    Si la expresion interior es positiva, el grupo es oneroso y ese importe se
    reconoce de inmediato como componente de perdida.

    Examples
    --------
    >>> import numpy as np
    >>> from ifrs17 import DiscountCurve, FulfilmentCashFlows
    >>> cf = FulfilmentCashFlows(
    ...     premiums=np.array([1000.0, 0.0, 0.0]),
    ...     claims=np.array([200.0, 300.0, 250.0]),
    ...     expenses=np.zeros(3),
    ...     coverage_units=np.ones(3),
    ... )
    >>> ir = initial_recognition(cf, DiscountCurve.flat(0.0, 3))
    >>> ir.is_onerous
    False
    """
    pv = curve.present_value(cash_flows.net_outflow)
    ra = cost_of_capital_risk_adjustment(cash_flows, curve, risk_config)
    acq = cash_flows.acquisition_cash_flows
    fulfilment = pv + ra + acq
    return InitialRecognition(
        pv_future_cash_flows=pv,
        risk_adjustment=ra,
        acquisition_cash_flows=acq,
        csm=max(0.0, -fulfilment),
        loss_component=max(0.0, fulfilment),
    )


def measure(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    locked_in_rate: float | None = None,
    assumption_changes: dict[int, float] | None = None,
) -> GMMResult:
    """Mide un grupo de contratos y construye su analisis de movimientos.

    Parameters
    ----------
    cash_flows:
        Flujos de cumplimiento del grupo.
    curve:
        Curva de descuento corriente al reconocimiento inicial.
    risk_config:
        Parametros del ajuste por riesgo.
    locked_in_rate:
        Tasa bloqueada para devengar interes sobre el CSM. Si es ``None`` se toma
        la tasa spot del primer periodo de la curva, que es la practica habitual
        cuando la curva es plana.
    assumption_changes:
        Cambios de supuestos relativos a **servicio futuro**, como
        ``{periodo: importe}``. Un importe positivo es un cambio favorable que
        aumenta el CSM; uno negativo lo reduce. Si la reduccion agota el CSM, el
        exceso crea componente de perdida. Si hay componente de perdida vigente,
        un cambio favorable lo revierte primero (parrafo 50).

    Returns
    -------
    GMMResult
    """
    inicial = initial_recognition(cash_flows, curve, risk_config)
    n = cash_flows.n_periods
    tasa = float(curve.spot_rates[0]) if locked_in_rate is None else float(locked_in_rate)
    cambios = assumption_changes or {}
    if any(not 0 <= p < n for p in cambios):
        raise ValueError(f"assumption_changes tiene periodos fuera de [0, {n})")

    csm = inicial.csm
    loss = inicial.loss_component
    ra_prev = inicial.risk_adjustment
    movimientos: list[PeriodMovement] = []

    for t in range(n):
        csm_apertura = csm
        loss_apertura = loss

        # (1) Devengo de interes a la tasa bloqueada, no a la corriente.
        devengo = csm_apertura * tasa
        csm_tras_interes = csm_apertura + devengo

        # (2) Cambios de supuestos de servicio futuro.
        cambio_bruto = float(cambios.get(t, 0.0))
        movimiento_loss = 0.0
        cambio_csm = 0.0
        if cambio_bruto > 0.0:
            # Un cambio favorable revierte primero el componente de perdida.
            reversion = min(loss, cambio_bruto)
            loss -= reversion
            movimiento_loss = -reversion
            cambio_csm = cambio_bruto - reversion
        elif cambio_bruto < 0.0:
            # Un cambio adverso consume CSM; el exceso crea perdida.
            absorbido = min(csm_tras_interes, -cambio_bruto)
            cambio_csm = -absorbido
            exceso = -cambio_bruto - absorbido
            loss += exceso
            movimiento_loss = exceso

        csm_antes_liberacion = csm_tras_interes + cambio_csm
        # El CSM nunca es negativo: la regla se aplica, no se supone.
        csm_antes_liberacion = max(0.0, csm_antes_liberacion)

        # (3) Liberacion por unidades de cobertura del periodo.
        cu_restantes = cash_flows.remaining_coverage_units(t)
        proporcion = (
            float(cash_flows.coverage_units[t]) / cu_restantes if cu_restantes > 0.0 else 1.0
        )
        liberacion = csm_antes_liberacion * proporcion
        csm_cierre = csm_antes_liberacion - liberacion

        # (4) Liberacion del ajuste por riesgo: se reevalua la cobertura restante.
        ra_cierre = cost_of_capital_risk_adjustment(
            cash_flows, curve, risk_config, from_period=t + 1
        )
        ra_liberado = ra_prev - ra_cierre

        # (5) Ingreso de seguro (parrafos B120-B124) y gasto del servicio.
        esperados = float(cash_flows.claims[t] + cash_flows.expenses[t])
        ingreso = liberacion + ra_liberado + esperados
        gasto = esperados
        if loss_apertura > 0.0 or movimiento_loss != 0.0:
            # El componente de perdida no genera ingreso: su movimiento se
            # presenta como gasto (o reversion de gasto) del servicio.
            gasto += movimiento_loss

        movimientos.append(
            PeriodMovement(
                period=t,
                csm_opening=csm_apertura,
                csm_interest_accretion=devengo,
                csm_change_future_service=cambio_csm,
                csm_release=liberacion,
                csm_closing=csm_cierre,
                risk_adjustment_opening=ra_prev,
                risk_adjustment_release=ra_liberado,
                risk_adjustment_closing=ra_cierre,
                loss_component_opening=loss_apertura,
                loss_component_movement=movimiento_loss,
                loss_component_closing=loss,
                expected_claims_and_expenses=esperados,
                insurance_revenue=ingreso,
                insurance_service_expense=gasto,
            )
        )
        csm = csm_cierre
        ra_prev = ra_cierre

    return GMMResult(initial=inicial, movements=movimientos, locked_in_rate=tasa)
