"""Unidad de cuenta: cartera, cohorte anual y grupo de rentabilidad.

Esta es la pieza que la propuesta original omitia por completo, y sin ella no se
esta aplicando IFRS 17. La norma no permite medir un libro de contratos como un
todo: obliga a descomponerlo (parrafos 14-24).

- Una **cartera** agrupa contratos sujetos a riesgos similares y gestionados
  juntos (parrafo 14).
- Dentro de la cartera, los contratos se separan por **rentabilidad** en al
  menos tres grupos: onerosos al inicio, sin posibilidad significativa de
  volverse onerosos, y el resto (parrafo 16).
- Y se separan en **cohortes anuales**: un grupo no puede contener contratos
  emitidos con mas de un anio de diferencia (parrafo 22).

La consecuencia economica es la que importa: la **prohibicion de compensar**. Un
contrato oneroso no puede esconderse dentro de un grupo rentable, porque su
perdida se reconoce de inmediato mientras el beneficio del rentable se difiere.
Agrupar mal adelanta o retrasa resultados, y es el incumplimiento mas comun.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve
from .gmm import initial_recognition
from .risk_adjustment import RiskAdjustmentConfig

__all__ = ["ProfitabilityBucket", "Contract", "ContractGroup", "classify", "group_contracts"]


class ProfitabilityBucket(str, Enum):
    """Grupos de rentabilidad del parrafo 16."""

    ONEROUS = "oneroso"
    """Oneroso en el reconocimiento inicial."""

    NO_SIGNIFICANT_RISK = "sin_riesgo_significativo"
    """Sin posibilidad significativa de volverse oneroso."""

    REMAINING = "resto"
    """El resto de los contratos de la cartera."""


@dataclass(frozen=True)
class Contract:
    """Un contrato, o un conjunto homogeneo tratado como una unidad.

    Parameters
    ----------
    identifier:
        Identificador unico, usado en la traza de auditoria.
    portfolio:
        Cartera a la que pertenece.
    issue_year:
        Anio de emision, que determina la cohorte anual.
    cash_flows:
        Flujos de cumplimiento esperados.
    """

    identifier: str
    portfolio: str
    issue_year: int
    cash_flows: FulfilmentCashFlows


@dataclass(frozen=True)
class ContractGroup:
    """Unidad de cuenta: cartera, cohorte anual y grupo de rentabilidad."""

    portfolio: str
    cohort: int
    bucket: ProfitabilityBucket
    contracts: tuple[Contract, ...]

    @property
    def key(self) -> tuple[str, int, str]:
        """Clave identificadora del grupo."""
        return (self.portfolio, self.cohort, self.bucket.value)

    @property
    def n_contracts(self) -> int:
        """Numero de contratos en el grupo."""
        return len(self.contracts)

    def aggregate_cash_flows(self) -> FulfilmentCashFlows:
        """Suma los flujos de los contratos del grupo.

        La agregacion **dentro** de un grupo si esta permitida: es precisamente
        la unidad de medicion. Lo que la norma prohibe es agregar entre grupos de
        distinta rentabilidad o cohorte.
        """
        if not self.contracts:
            raise ValueError("el grupo no tiene contratos")
        n = self.contracts[0].cash_flows.n_periods
        if any(c.cash_flows.n_periods != n for c in self.contracts):
            raise ValueError(
                "los contratos del grupo tienen horizontes distintos; alinear la "
                "proyeccion antes de agregar"
            )
        primas = sum(c.cash_flows.premiums for c in self.contracts)
        siniestros = sum(c.cash_flows.claims for c in self.contracts)
        gastos = sum(c.cash_flows.expenses for c in self.contracts)
        adquisicion = sum(c.cash_flows.acquisition_cash_flows for c in self.contracts)
        unidades = sum(c.cash_flows.coverage_units for c in self.contracts)
        return FulfilmentCashFlows(
            premiums=primas,
            claims=siniestros,
            expenses=gastos,
            acquisition_cash_flows=float(adquisicion),
            coverage_units=unidades,
        )


def classify(
    contract: Contract,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    comfortable_margin: float = 0.10,
) -> ProfitabilityBucket:
    """Clasifica un contrato en su grupo de rentabilidad.

    Parameters
    ----------
    comfortable_margin:
        CSM minimo, como fraccion del valor presente de las primas, para
        considerar que no hay posibilidad significativa de volverse oneroso. La
        norma no cuantifica este umbral, asi que es un parametro explicito y no
        una constante escondida.
    """
    medicion = initial_recognition(contract.cash_flows, curve, risk_config)
    if medicion.is_onerous:
        return ProfitabilityBucket.ONEROUS

    pv_primas = curve.present_value(contract.cash_flows.premiums)
    if pv_primas > 0.0 and medicion.csm / pv_primas >= comfortable_margin:
        return ProfitabilityBucket.NO_SIGNIFICANT_RISK
    return ProfitabilityBucket.REMAINING


def group_contracts(
    contracts: list[Contract],
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    comfortable_margin: float = 0.10,
) -> list[ContractGroup]:
    """Descompone una lista de contratos en unidades de cuenta.

    Returns
    -------
    list[ContractGroup]
        Grupos ordenados por cartera, cohorte y rentabilidad. Ningun grupo mezcla
        cohortes ni rentabilidades, que es lo que garantiza que no haya
        compensacion indebida entre contratos onerosos y rentables.
    """
    cubetas: dict[tuple[str, int, ProfitabilityBucket], list[Contract]] = defaultdict(list)
    for contrato in contracts:
        bucket = classify(contrato, curve, risk_config, comfortable_margin)
        cubetas[(contrato.portfolio, contrato.issue_year, bucket)].append(contrato)

    grupos = [
        ContractGroup(
            portfolio=portafolio,
            cohort=cohorte,
            bucket=bucket,
            contracts=tuple(lista),
        )
        for (portafolio, cohorte, bucket), lista in cubetas.items()
    ]
    return sorted(grupos, key=lambda g: (g.portfolio, g.cohort, g.bucket.value))
