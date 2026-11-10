"""Validacion de la unidad de cuenta, el PAA y la traza de auditoria."""

from __future__ import annotations

import numpy as np
import pytest

from ifrs17 import (
    Contract,
    DiscountCurve,
    FulfilmentCashFlows,
    ProfitabilityBucket,
    RiskAdjustmentConfig,
    build_audit_trail,
    classify,
    equivalent_confidence_level,
    group_contracts,
    input_hash,
    measure_paa,
    paa_eligibility,
    paa_movement_analysis,
)

SIN_RIESGO = RiskAdjustmentConfig(cost_of_capital_rate=0.0, capital_factor=0.0)


def flujos(prima: float, siniestro: float, n: int = 4) -> FulfilmentCashFlows:
    return FulfilmentCashFlows(
        premiums=np.array([prima] + [0.0] * (n - 1)),
        claims=np.full(n, siniestro),
        expenses=np.zeros(n),
        coverage_units=np.ones(n),
    )


# --------------------------------------------------------------- unidad de cuenta


def test_clasifica_oneroso_rentable_y_resto():
    curva = DiscountCurve.flat(0.0, 4)
    oneroso = Contract("A", "auto", 2026, flujos(100.0, 50.0))
    holgado = Contract("B", "auto", 2026, flujos(1000.0, 50.0))
    justo = Contract("C", "auto", 2026, flujos(205.0, 50.0))

    assert classify(oneroso, curva, SIN_RIESGO) is ProfitabilityBucket.ONEROUS
    assert classify(holgado, curva, SIN_RIESGO) is ProfitabilityBucket.NO_SIGNIFICANT_RISK
    assert classify(justo, curva, SIN_RIESGO) is ProfitabilityBucket.REMAINING


def test_las_cohortes_anuales_no_se_mezclan():
    """Parrafo 22: un grupo no puede contener contratos de anios distintos."""
    curva = DiscountCurve.flat(0.0, 4)
    contratos = [
        Contract("A", "auto", 2025, flujos(1000.0, 50.0)),
        Contract("B", "auto", 2026, flujos(1000.0, 50.0)),
    ]
    grupos = group_contracts(contratos, curva, SIN_RIESGO)
    assert len(grupos) == 2
    assert {g.cohort for g in grupos} == {2025, 2026}


def test_onerosos_y_rentables_no_se_compensan_dentro_de_un_grupo():
    """Parrafo 16: la prohibicion de compensar es el nucleo de la agrupacion.

    Si un contrato oneroso pudiera agruparse con uno rentable, su perdida se
    diferiria en lugar de reconocerse de inmediato.
    """
    curva = DiscountCurve.flat(0.0, 4)
    contratos = [
        Contract("oneroso", "auto", 2026, flujos(100.0, 50.0)),
        Contract("rentable", "auto", 2026, flujos(1000.0, 50.0)),
    ]
    grupos = group_contracts(contratos, curva, SIN_RIESGO)
    assert len(grupos) == 2
    for g in grupos:
        assert g.n_contracts == 1
    assert {g.bucket for g in grupos} == {
        ProfitabilityBucket.ONEROUS,
        ProfitabilityBucket.NO_SIGNIFICANT_RISK,
    }


def test_las_carteras_distintas_no_se_mezclan():
    curva = DiscountCurve.flat(0.0, 4)
    contratos = [
        Contract("A", "auto", 2026, flujos(1000.0, 50.0)),
        Contract("B", "vida", 2026, flujos(1000.0, 50.0)),
    ]
    assert len({g.portfolio for g in group_contracts(contratos, curva, SIN_RIESGO)}) == 2


def test_la_agregacion_dentro_de_un_grupo_suma_los_flujos():
    curva = DiscountCurve.flat(0.0, 4)
    contratos = [
        Contract("A", "auto", 2026, flujos(1000.0, 50.0)),
        Contract("B", "auto", 2026, flujos(1000.0, 50.0)),
    ]
    grupos = group_contracts(contratos, curva, SIN_RIESGO)
    assert len(grupos) == 1
    agregado = grupos[0].aggregate_cash_flows()
    assert agregado.premiums[0] == pytest.approx(2000.0)
    assert agregado.claims[0] == pytest.approx(100.0)


def test_rechaza_agregar_contratos_con_horizontes_distintos():
    curva = DiscountCurve.flat(0.0, 6)
    contratos = [
        Contract("A", "auto", 2026, flujos(1000.0, 50.0, n=4)),
        Contract("B", "auto", 2026, flujos(1000.0, 50.0, n=6)),
    ]
    grupos = group_contracts(contratos, curva, SIN_RIESGO)
    with pytest.raises(ValueError, match="horizontes distintos"):
        grupos[0].aggregate_cash_flows()


# --------------------------------------------------------------------------- PAA


def test_un_grupo_de_un_solo_periodo_es_elegible_automaticamente():
    cf = FulfilmentCashFlows(
        premiums=np.array([1000.0]),
        claims=np.array([600.0]),
        expenses=np.array([50.0]),
        coverage_units=np.array([1.0]),
    )
    veredicto = paa_eligibility(cf, DiscountCurve.flat(0.03, 1))
    assert veredicto.eligible
    assert "no excede un periodo" in veredicto.reason


def test_un_grupo_largo_con_diferencia_material_no_es_elegible():
    cf = flujos(1000.0, 100.0, n=10)
    veredicto = paa_eligibility(cf, DiscountCurve.flat(0.06, 10))
    assert not veredicto.eligible
    assert veredicto.relative_difference is not None


def test_medir_por_paa_un_grupo_no_elegible_lanza_excepcion():
    cf = flujos(1000.0, 100.0, n=10)
    with pytest.raises(ValueError, match="no es elegible para el PAA"):
        measure_paa(cf, DiscountCurve.flat(0.06, 10))


def test_el_pasivo_por_cobertura_restante_se_agota():
    cf = flujos(1000.0, 200.0, n=4)
    res = measure_paa(cf, DiscountCurve.flat(0.0, 4), SIN_RIESGO, require_eligibility=False)
    assert res.closing_lrc == pytest.approx(0.0, abs=1e-9)
    assert res.reconciles()


def test_el_ingreso_total_del_paa_iguala_la_prima():
    cf = flujos(1000.0, 200.0, n=4)
    res = measure_paa(cf, DiscountCurve.flat(0.0, 4), SIN_RIESGO, require_eligibility=False)
    assert res.total_revenue == pytest.approx(float(np.sum(cf.premiums)))


def test_el_pasivo_por_cobertura_restante_decrece():
    cf = flujos(1000.0, 200.0, n=5)
    res = measure_paa(cf, DiscountCurve.flat(0.0, 5), SIN_RIESGO, require_eligibility=False)
    cierres = [p.lrc_closing for p in res.periods]
    assert all(b <= a + 1e-9 for a, b in zip(cierres, cierres[1:]))


def test_gastar_la_adquisicion_de_inmediato_adelanta_el_gasto():
    cf = FulfilmentCashFlows(
        premiums=np.array([1000.0, 0.0, 0.0, 0.0]),
        claims=np.full(4, 150.0),
        expenses=np.zeros(4),
        acquisition_cash_flows=80.0,
        coverage_units=np.ones(4),
    )
    curva = DiscountCurve.flat(0.0, 4)
    diferido = measure_paa(cf, curva, SIN_RIESGO, require_eligibility=False)
    inmediato = measure_paa(
        cf, curva, SIN_RIESGO, expense_acquisition_immediately=True, require_eligibility=False
    )
    assert inmediato.periods[0].acquisition_amortisation == pytest.approx(80.0)
    assert diferido.periods[0].acquisition_amortisation == pytest.approx(20.0)
    # El resultado acumulado no cambia: cambia cuando se reconoce.
    assert inmediato.total_insurance_service_result == pytest.approx(
        diferido.total_insurance_service_result
    )


def test_el_analisis_de_movimientos_del_paa_tiene_una_fila_por_periodo():
    cf = flujos(1000.0, 200.0, n=4)
    res = measure_paa(cf, DiscountCurve.flat(0.0, 4), SIN_RIESGO, require_eligibility=False)
    df = paa_movement_analysis(res)
    assert len(df) == 4
    assert "lrc_cierre" in df.columns


# --------------------------------------------------------------------- auditoria


def test_el_hash_es_determinista():
    cf = flujos(1000.0, 200.0)
    curva = DiscountCurve.flat(0.03, 4)
    assert input_hash(cf, curva) == input_hash(cf, curva)
    assert len(input_hash(cf, curva)) == 64


def test_el_hash_cambia_ante_cualquier_cambio_de_insumo():
    cf = flujos(1000.0, 200.0)
    curva = DiscountCurve.flat(0.03, 4)
    base = input_hash(cf, curva)
    assert input_hash(cf.with_scaled_claims(1.000001), curva) != base
    assert input_hash(cf, DiscountCurve.flat(0.0300001, 4)) != base
    assert input_hash(cf, curva, RiskAdjustmentConfig(0.07, 0.15)) != base


def test_la_traza_de_auditoria_verifica_los_insumos():
    cf = flujos(1000.0, 200.0)
    curva = DiscountCurve.flat(0.03, 4)
    traza = build_audit_trail(cf, curva)
    assert traza.matches(cf, curva, RiskAdjustmentConfig())
    assert not traza.matches(cf.with_scaled_claims(1.1), curva)


def test_la_traza_registra_los_cambios_de_supuestos():
    cf = flujos(1000.0, 200.0)
    curva = DiscountCurve.flat(0.03, 4)
    traza = build_audit_trail(cf, curva, assumption_changes={1: -25.0})
    assert traza.assumptions["assumption_changes"] == {1: -25.0}
    sin_cambios = build_audit_trail(cf, curva)
    assert traza.input_digest != sin_cambios.input_digest


def test_la_traza_es_serializable():
    cf = flujos(1000.0, 200.0)
    traza = build_audit_trail(cf, DiscountCurve.flat(0.03, 4))
    d = traza.to_dict()
    assert d["input_digest"] == traza.input_digest
    assert d["n_periods"] == 4


# ------------------------------------------------------- nivel de confianza


def test_el_nivel_de_confianza_equivalente_crece_con_el_ajuste_por_riesgo():
    bajo = equivalent_confidence_level(50.0, 1000.0, 0.20)
    alto = equivalent_confidence_level(200.0, 1000.0, 0.20)
    assert 0.5 < bajo < alto < 1.0


def test_un_ajuste_por_riesgo_nulo_corresponde_a_la_mediana():
    assert equivalent_confidence_level(0.0, 1000.0, 0.20) == pytest.approx(0.5)


def test_el_nivel_de_confianza_no_esta_definido_sin_dispersion():
    assert np.isnan(equivalent_confidence_level(50.0, 1000.0, 0.0))


# --------------------------------------------------------------------- curva


def test_la_curva_no_extrapola_en_silencio():
    curva = DiscountCurve.flat(0.03, 3)
    with pytest.raises(ValueError, match="extender la curva"):
        curva.discount_factors(5)


def test_los_factores_de_descuento_decrecen():
    df = DiscountCurve.flat(0.05, 10).discount_factors()
    assert np.all(np.diff(df) < 0.0)
    assert df[0] == pytest.approx(1.0 / 1.05)


def test_la_forward_de_una_curva_plana_es_la_propia_tasa():
    curva = DiscountCurve.flat(0.04, 5)
    for t in range(5):
        assert curva.forward_rate(t) == pytest.approx(0.04)
