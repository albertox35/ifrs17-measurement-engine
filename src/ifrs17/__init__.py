"""Motor de medicion IFRS 17.

Implementa el modelo general (GMM) y el enfoque de asignacion de prima (PAA),
con la unidad de cuenta, el ajuste por riesgo, el componente de perdida y el
analisis de movimientos del parrafo 101.

Las tres reglas del CSM que gobiernan todo el motor, codificadas y no supuestas:

1. El CSM nunca es negativo; el exceso crea componente de perdida.
2. El interes se devenga a la tasa bloqueada en el reconocimiento inicial.
3. Un cambio favorable revierte primero el componente de perdida y solo despues
   restablece CSM.
"""

from __future__ import annotations

from .audit import AuditTrail, build_audit_trail, input_hash
from .cashflows import FulfilmentCashFlows
from .discounting import DiscountCurve
from .gmm import GMMResult, InitialRecognition, PeriodMovement, initial_recognition, measure
from .grouping import Contract, ContractGroup, ProfitabilityBucket, classify, group_contracts
from .paa import PAAEligibility, PAAResult, measure_paa, paa_eligibility
from .reporting import check_reconciliation, movement_analysis, paa_movement_analysis
from .risk_adjustment import (
    RiskAdjustmentConfig,
    cost_of_capital_risk_adjustment,
    equivalent_confidence_level,
)

__version__ = "0.1.0"

__all__ = [
    "AuditTrail",
    "Contract",
    "ContractGroup",
    "DiscountCurve",
    "FulfilmentCashFlows",
    "GMMResult",
    "InitialRecognition",
    "PAAEligibility",
    "PAAResult",
    "PeriodMovement",
    "ProfitabilityBucket",
    "RiskAdjustmentConfig",
    "__version__",
    "build_audit_trail",
    "check_reconciliation",
    "classify",
    "cost_of_capital_risk_adjustment",
    "equivalent_confidence_level",
    "group_contracts",
    "initial_recognition",
    "input_hash",
    "measure",
    "measure_paa",
    "movement_analysis",
    "paa_eligibility",
    "paa_movement_analysis",
]
