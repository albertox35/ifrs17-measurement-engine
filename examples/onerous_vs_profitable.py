"""Grupo rentable frente a grupo oneroso, y por que la agrupacion importa.

Muestra la asimetria que define IFRS 17: el beneficio de un grupo rentable se
difiere a lo largo de la cobertura via el CSM, mientras que la perdida de un
grupo oneroso se reconoce de inmediato. Por eso la norma prohibe agruparlos.

Ejecutar con:

    python examples/onerous_vs_profitable.py
"""

from __future__ import annotations

import numpy as np

from ifrs17 import (
    Contract,
    DiscountCurve,
    FulfilmentCashFlows,
    RiskAdjustmentConfig,
    build_audit_trail,
    check_reconciliation,
    group_contracts,
    initial_recognition,
    measure,
    movement_analysis,
)

N = 5
CURVA = DiscountCurve.flat(0.04, N)
RIESGO = RiskAdjustmentConfig(cost_of_capital_rate=0.06, capital_factor=0.15)


def flujos(prima: float, siniestro: float, adquisicion: float = 60.0) -> FulfilmentCashFlows:
    return FulfilmentCashFlows(
        premiums=np.array([prima] + [0.0] * (N - 1)),
        claims=np.full(N, siniestro),
        expenses=np.full(N, 15.0),
        acquisition_cash_flows=adquisicion,
        coverage_units=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    )


def main() -> None:
    rentable = flujos(prima=1200.0, siniestro=140.0)
    oneroso = flujos(prima=600.0, siniestro=160.0)

    print("=== Reconocimiento inicial ===")
    for nombre, cf in (("rentable", rentable), ("oneroso", oneroso)):
        ir = initial_recognition(cf, CURVA, RIESGO)
        print(f"\n{nombre}:")
        print(f"  VP flujos de cumplimiento : {ir.pv_future_cash_flows:>12,.2f}")
        print(f"  ajuste por riesgo         : {ir.risk_adjustment:>12,.2f}")
        print(f"  flujos de adquisicion     : {ir.acquisition_cash_flows:>12,.2f}")
        print(f"  CSM                       : {ir.csm:>12,.2f}")
        print(f"  componente de perdida     : {ir.loss_component:>12,.2f}")
        print(f"  pasivo inicial            : {ir.insurance_contract_liability:>12,.2f}")
        print(f"  oneroso                   : {ir.is_onerous}")

    print("\n\n=== Analisis de movimientos, grupo rentable (parrafo 101) ===")
    res = measure(rentable, CURVA, RIESGO)
    df = movement_analysis(res)
    columnas = [
        "csm_apertura",
        "csm_interes_devengado",
        "csm_liberacion",
        "csm_cierre",
        "ra_liberacion",
        "ingreso_de_seguro",
    ]
    print(df[columnas].round(2).to_string())

    print(f"\nCSM liberado en total      : {res.total_csm_released:,.2f}")
    print(f"interes devengado en total : {res.total_interest_accreted:,.2f}")
    print(f"CSM inicial + interes      : {res.initial.csm + res.total_interest_accreted:,.2f}")
    print(f"CSM al cierre              : {res.closing_csm:,.2f}")
    reporte = check_reconciliation(res)
    print(f"conciliacion cuadra        : {reporte.reconciles} "
          f"(error maximo {reporte.max_absolute_error:.2e})")

    print("\n\n=== Efecto de un cambio adverso de supuestos en el periodo 2 ===")
    choque = measure(rentable, CURVA, RIESGO, assumption_changes={2: -900.0})
    df2 = movement_analysis(choque)
    print(
        df2[
            [
                "csm_apertura",
                "csm_cambio_servicio_futuro",
                "csm_cierre",
                "componente_perdida_cierre",
            ]
        ]
        .round(2)
        .to_string()
    )
    print("\nEl CSM se agota y el exceso crea componente de perdida: nunca se vuelve negativo.")

    print("\n\n=== Agrupacion: la prohibicion de compensar ===")
    contratos = [
        Contract("P-2025-a", "auto", 2025, flujos(1200.0, 140.0)),
        Contract("P-2026-a", "auto", 2026, flujos(1200.0, 140.0)),
        Contract("P-2026-b", "auto", 2026, flujos(600.0, 160.0)),
        Contract("V-2026-a", "vida", 2026, flujos(1500.0, 120.0)),
    ]
    for g in group_contracts(contratos, CURVA, RIESGO):
        ids = ", ".join(c.identifier for c in g.contracts)
        print(f"  {g.portfolio:<6} cohorte {g.cohort}  {g.bucket.value:<24} -> {ids}")
    print("\nCuatro contratos producen cuatro grupos: ni las cohortes ni las")
    print("rentabilidades se mezclan, asi que el oneroso no se esconde en el rentable.")

    print("\n\n=== Traza de auditoria ===")
    traza = build_audit_trail(rentable, CURVA, RIESGO)
    print(f"hash de insumos : {traza.input_digest}")
    print(f"version motor   : {traza.engine_version}")
    print(f"verifica        : {traza.matches(rentable, CURVA, RIESGO)}")
    print(f"detecta cambio  : {not traza.matches(rentable.with_scaled_claims(1.0001), CURVA, RIESGO)}")


if __name__ == "__main__":
    main()
