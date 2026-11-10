"""Traza de auditoria.

«Trazabilidad financiera auditable» solo significa algo si se concreta. Aqui se
concreta en dos cosas verificables:

1. Un **hash criptografico** del conjunto completo de supuestos e insumos que
   produjo una medicion. Dos corridas con los mismos insumos dan el mismo hash;
   cualquier cambio, por pequenio que sea, lo cambia. Eso permite demostrar a un
   auditor que una cifra publicada proviene de un conjunto de supuestos concreto.
2. Un **registro de linaje** que asocia cada linea del analisis de movimientos
   con ese hash y con la version del motor.

El hash es estable entre plataformas y entre ejecuciones porque se construye
sobre una representacion canonica de los arreglos, no sobre su direccion de
memoria ni sobre un ``repr`` que podria cambiar.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve
from .risk_adjustment import RiskAdjustmentConfig

__all__ = ["AuditTrail", "input_hash", "build_audit_trail"]

_ENGINE_VERSION = "0.1.0"
_ROUND = 10


def _canonical(value: Any) -> Any:
    """Representacion canonica y estable de un insumo."""
    if isinstance(value, np.ndarray):
        return [round(float(x), _ROUND) for x in value.ravel()]
    if isinstance(value, (np.floating, float)):
        return round(float(value), _ROUND)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    return value


def input_hash(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """SHA-256 del conjunto de insumos de una medicion.

    Returns
    -------
    str
        Hash hexadecimal de 64 caracteres.

    Examples
    --------
    El hash es determinista y sensible a cualquier cambio de insumo:

    >>> import numpy as np
    >>> from ifrs17 import DiscountCurve, FulfilmentCashFlows
    >>> cf = FulfilmentCashFlows(
    ...     premiums=np.array([100.0, 0.0]),
    ...     claims=np.array([30.0, 30.0]),
    ...     expenses=np.zeros(2),
    ...     coverage_units=np.ones(2),
    ... )
    >>> curva = DiscountCurve.flat(0.03, 2)
    >>> input_hash(cf, curva) == input_hash(cf, curva)
    True
    >>> input_hash(cf, curva) == input_hash(cf.with_scaled_claims(1.0001), curva)
    False
    """
    cfg = risk_config or RiskAdjustmentConfig()
    payload = {
        "engine_version": _ENGINE_VERSION,
        "premiums": _canonical(cash_flows.premiums),
        "claims": _canonical(cash_flows.claims),
        "expenses": _canonical(cash_flows.expenses),
        "acquisition_cash_flows": _canonical(cash_flows.acquisition_cash_flows),
        "coverage_units": _canonical(cash_flows.coverage_units),
        "spot_rates": _canonical(curve.spot_rates),
        "cost_of_capital_rate": _canonical(cfg.cost_of_capital_rate),
        "capital_factor": _canonical(cfg.capital_factor),
        "extra": _canonical(extra or {}),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass(frozen=True)
class AuditTrail:
    """Linaje de una medicion."""

    input_digest: str
    engine_version: str
    n_periods: int
    locked_in_rate: float
    assumptions: dict[str, Any]

    def matches(
        self,
        cash_flows: FulfilmentCashFlows,
        curve: DiscountCurve,
        risk_config: RiskAdjustmentConfig | None = None,
    ) -> bool:
        """Comprueba que unos insumos reproducen este linaje.

        Es la operacion que ejecutaria un auditor: tomar los flujos y la curva de
        un archivo y verificar que corresponden a la cifra publicada.

        La traza reconstruye por si misma la parte de los insumos que ya tiene
        registrada (los cambios de supuestos). Exigir al verificador que la
        rearme a mano seria una trampa: cualquier omision daria un falso negativo
        y el mecanismo de auditoria perderia valor.
        """
        if risk_config is None:
            risk_config = RiskAdjustmentConfig(
                cost_of_capital_rate=float(self.assumptions["cost_of_capital_rate"]),
                capital_factor=float(self.assumptions["capital_factor"]),
            )
        extra = {"assumption_changes": self.assumptions.get("assumption_changes", {})}
        return input_hash(cash_flows, curve, risk_config, extra) == self.input_digest

    def to_dict(self) -> dict[str, Any]:
        """Representacion serializable del linaje."""
        return {
            "input_digest": self.input_digest,
            "engine_version": self.engine_version,
            "n_periods": self.n_periods,
            "locked_in_rate": self.locked_in_rate,
            "assumptions": self.assumptions,
        }


def build_audit_trail(
    cash_flows: FulfilmentCashFlows,
    curve: DiscountCurve,
    risk_config: RiskAdjustmentConfig | None = None,
    locked_in_rate: float | None = None,
    assumption_changes: dict[int, float] | None = None,
) -> AuditTrail:
    """Construye el linaje de una medicion GMM."""
    cfg = risk_config or RiskAdjustmentConfig()
    extra = {"assumption_changes": assumption_changes or {}}
    tasa = float(curve.spot_rates[0]) if locked_in_rate is None else float(locked_in_rate)
    return AuditTrail(
        input_digest=input_hash(cash_flows, curve, cfg, extra),
        engine_version=_ENGINE_VERSION,
        n_periods=cash_flows.n_periods,
        locked_in_rate=tasa,
        assumptions={
            "cost_of_capital_rate": cfg.cost_of_capital_rate,
            "capital_factor": cfg.capital_factor,
            "assumption_changes": dict(assumption_changes or {}),
        },
    )
