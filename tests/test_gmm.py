"""Validacion del modelo general de medicion.

Las pruebas se concentran en las propiedades que **definen** el CSM. Un motor que
las incumpla produce estados financieros incorrectos, asi que se verifican de
forma explicita y no por inspeccion.
"""

from __future__ import annotations

import numpy as np
import pytest

from ifrs17 import (
    DiscountCurve,
    FulfilmentCashFlows,
    RiskAdjustmentConfig,
    check_reconciliation,
    initial_recognition,
    measure,
    movement_analysis,
)

SIN_RIESGO = RiskAdjustmentConfig(cost_of_capital_rate=0.0, capital_factor=0.0)


def grupo_rentable(n: int = 5) -> FulfilmentCashFlows:
    """Grupo con prima suficiente para ser rentable."""
    return FulfilmentCashFlows(
        premiums=np.array([1000.0] + [0.0] * (n - 1)),
        claims=np.full(n, 120.0),
        expenses=np.full(n, 20.0),
        acquisition_cash_flows=50.0,
        coverage_units=np.ones(n),
    )


def grupo_oneroso(n: int = 5) -> FulfilmentCashFlows:
    """Grupo cuya prima no alcanza a cubrir los siniestros esperados."""
    return FulfilmentCashFlows(
        premiums=np.array([400.0] + [0.0] * (n - 1)),
        claims=np.full(n, 150.0),
        expenses=np.full(n, 30.0),
        acquisition_cash_flows=40.0,
        coverage_units=np.ones(n),
    )


# ------------------------------------------------------- reconocimiento inicial


def test_csm_igual_al_margen_sin_descuento_ni_ajuste_por_riesgo():
    """Caso con solucion cerrada: tasa cero y ajuste por riesgo cero.

    El CSM debe ser exactamente primas menos siniestros, gastos y adquisicion.
    """
    cf = grupo_rentable()
    ir = initial_recognition(cf, DiscountCurve.flat(0.0, cf.n_periods), SIN_RIESGO)
    esperado = (
        float(np.sum(cf.premiums))
        - float(np.sum(cf.claims))
        - float(np.sum(cf.expenses))
        - cf.acquisition_cash_flows
    )
    assert ir.csm == pytest.approx(esperado)
    assert ir.loss_component == pytest.approx(0.0)


def test_el_csm_nunca_es_negativo():
    cf = grupo_oneroso()
    ir = initial_recognition(cf, DiscountCurve.flat(0.03, cf.n_periods))
    assert ir.csm == 0.0
    assert ir.loss_component > 0.0
    assert ir.is_onerous


def test_grupo_oneroso_reconoce_la_perdida_de_inmediato():
    cf = grupo_oneroso()
    ir = initial_recognition(cf, DiscountCurve.flat(0.0, cf.n_periods), SIN_RIESGO)
    deficit = (
        float(np.sum(cf.claims))
        + float(np.sum(cf.expenses))
        + cf.acquisition_cash_flows
        - float(np.sum(cf.premiums))
    )
    assert ir.loss_component == pytest.approx(deficit)


def test_el_ajuste_por_riesgo_reduce_el_csm():
    cf = grupo_rentable()
    curva = DiscountCurve.flat(0.03, cf.n_periods)
    sin = initial_recognition(cf, curva, SIN_RIESGO)
    con = initial_recognition(cf, curva, RiskAdjustmentConfig(0.06, 0.15))
    assert con.risk_adjustment > 0.0
    assert con.csm < sin.csm
    assert con.csm == pytest.approx(sin.csm - con.risk_adjustment)


def test_el_pasivo_inicial_iguala_el_componente_de_perdida():
    """Identidad contable del reconocimiento inicial.

    El pasivo es ``FCF + CSM``. Como ``CSM = max(0, -FCF)``, el pasivo se reduce a
    ``max(0, FCF)``, que es exactamente el componente de perdida: cero para un
    grupo rentable y el deficit para uno oneroso. Un motor que no cumpla esta
    identidad esta reconociendo un pasivo inicial que no corresponde.
    """
    curva = DiscountCurve.flat(0.03, 5)
    for cf in (grupo_rentable(), grupo_oneroso()):
        ir = initial_recognition(cf, curva)
        assert ir.insurance_contract_liability == pytest.approx(ir.loss_component)

    assert initial_recognition(grupo_rentable(), curva).insurance_contract_liability == (
        pytest.approx(0.0)
    )
    assert initial_recognition(grupo_oneroso(), curva).insurance_contract_liability > 0.0


# ------------------------------------------------------------- roll-forward


def test_el_csm_se_libera_por_completo_al_final_de_la_cobertura():
    cf = grupo_rentable()
    res = measure(cf, DiscountCurve.flat(0.03, cf.n_periods))
    assert res.closing_csm == pytest.approx(0.0, abs=1e-9)


def test_la_liberacion_total_iguala_csm_inicial_mas_interes_devengado():
    """Identidad de conservacion del CSM.

    Sin cambios de supuestos, todo el CSM inicial mas el interes devengado acaba
    liberandose a resultados. Si no cuadra, hay una fuga en el roll-forward.
    """
    cf = grupo_rentable()
    res = measure(cf, DiscountCurve.flat(0.04, cf.n_periods))
    assert res.total_csm_released == pytest.approx(
        res.initial.csm + res.total_interest_accreted, rel=1e-10
    )


def test_cada_linea_del_analisis_de_movimientos_cuadra():
    cf = grupo_rentable()
    res = measure(cf, DiscountCurve.flat(0.03, cf.n_periods))
    reporte = check_reconciliation(res)
    assert reporte.reconciles, f"periodos que no cuadran: {reporte.failing_periods}"
    assert reporte.max_absolute_error < 1e-9
    assert res.reconciles()


def test_el_interes_se_devenga_a_la_tasa_bloqueada_no_a_la_corriente():
    cf = grupo_rentable()
    curva = DiscountCurve.flat(0.03, cf.n_periods)
    res = measure(cf, curva, locked_in_rate=0.08)
    primero = res.movements[0]
    assert primero.csm_interest_accretion == pytest.approx(primero.csm_opening * 0.08)
    # Y no a la tasa de la curva.
    assert primero.csm_interest_accretion != pytest.approx(primero.csm_opening * 0.03)


def test_el_csm_permanece_no_negativo_en_todo_el_roll_forward():
    cf = grupo_rentable()
    # Un cambio adverso que excede con holgura el CSM disponible.
    res = measure(
        cf,
        DiscountCurve.flat(0.03, cf.n_periods),
        assumption_changes={1: -10_000.0},
    )
    for m in res.movements:
        assert m.csm_opening >= 0.0
        assert m.csm_closing >= 0.0


def test_un_cambio_adverso_que_agota_el_csm_crea_componente_de_perdida():
    cf = grupo_rentable()
    res = measure(
        cf,
        DiscountCurve.flat(0.0, cf.n_periods),
        SIN_RIESGO,
        assumption_changes={1: -10_000.0},
    )
    m = res.movements[1]
    assert m.csm_closing == pytest.approx(0.0)
    assert m.loss_component_closing > 0.0


def test_un_cambio_favorable_revierte_primero_el_componente_de_perdida():
    """Parrafo 50: no se restablece CSM mientras quede perdida por revertir."""
    cf = grupo_oneroso()
    res = measure(
        cf,
        DiscountCurve.flat(0.0, cf.n_periods),
        SIN_RIESGO,
        assumption_changes={2: 50.0},
    )
    m = res.movements[2]
    perdida_inicial = res.initial.loss_component
    assert perdida_inicial > 50.0, "el caso debe tener mas perdida que el cambio favorable"
    assert m.loss_component_movement == pytest.approx(-50.0)
    assert m.csm_change_future_service == pytest.approx(0.0)
    assert m.csm_closing == pytest.approx(0.0)


def test_un_cambio_favorable_mayor_que_la_perdida_restablece_csm():
    cf = grupo_oneroso()
    perdida = initial_recognition(
        cf, DiscountCurve.flat(0.0, cf.n_periods), SIN_RIESGO
    ).loss_component
    exceso = 100.0
    res = measure(
        cf,
        DiscountCurve.flat(0.0, cf.n_periods),
        SIN_RIESGO,
        assumption_changes={0: perdida + exceso},
    )
    m = res.movements[0]
    assert m.loss_component_closing == pytest.approx(0.0)
    assert m.csm_change_future_service == pytest.approx(exceso)


def test_el_ajuste_por_riesgo_se_libera_por_completo():
    cf = grupo_rentable()
    res = measure(cf, DiscountCurve.flat(0.03, cf.n_periods))
    liberado = sum(m.risk_adjustment_release for m in res.movements)
    assert liberado == pytest.approx(res.initial.risk_adjustment, rel=1e-9)
    assert res.movements[-1].risk_adjustment_closing == pytest.approx(0.0)


def test_las_unidades_de_cobertura_determinan_el_perfil_de_liberacion():
    """Con cobertura concentrada al inicio, el CSM se libera antes."""
    n = 5
    base = dict(
        premiums=np.array([1000.0] + [0.0] * (n - 1)),
        claims=np.full(n, 120.0),
        expenses=np.full(n, 20.0),
        acquisition_cash_flows=50.0,
    )
    plano = measure(
        FulfilmentCashFlows(**base, coverage_units=np.ones(n)),
        DiscountCurve.flat(0.0, n),
        SIN_RIESGO,
    )
    frontal = measure(
        FulfilmentCashFlows(**base, coverage_units=np.array([5.0, 4.0, 3.0, 2.0, 1.0])),
        DiscountCurve.flat(0.0, n),
        SIN_RIESGO,
    )
    assert frontal.movements[0].csm_release > plano.movements[0].csm_release
    # Pero el total liberado es el mismo: cambia el perfil, no el importe.
    assert frontal.total_csm_released == pytest.approx(plano.total_csm_released, rel=1e-10)


def test_las_unidades_de_cobertura_son_obligatorias():
    with pytest.raises(ValueError, match="coverage_units es obligatorio"):
        FulfilmentCashFlows(
            premiums=np.ones(3), claims=np.ones(3), expenses=np.zeros(3)
        )


def test_rechaza_cambios_de_supuestos_fuera_de_rango():
    cf = grupo_rentable()
    with pytest.raises(ValueError, match="periodos fuera de"):
        measure(cf, DiscountCurve.flat(0.0, cf.n_periods), assumption_changes={99: 1.0})


# -------------------------------------------------------------------- reporte


def test_el_analisis_de_movimientos_tiene_una_fila_por_periodo():
    cf = grupo_rentable(7)
    df = movement_analysis(measure(cf, DiscountCurve.flat(0.03, 7)))
    assert len(df) == 7
    assert "csm_liberacion" in df.columns
    assert "componente_perdida_cierre" in df.columns


def test_el_cierre_de_un_periodo_es_la_apertura_del_siguiente():
    cf = grupo_rentable(6)
    res = measure(cf, DiscountCurve.flat(0.03, 6))
    for anterior, siguiente in zip(res.movements, res.movements[1:]):
        assert siguiente.csm_opening == pytest.approx(anterior.csm_closing)
        assert siguiente.risk_adjustment_opening == pytest.approx(anterior.risk_adjustment_closing)
