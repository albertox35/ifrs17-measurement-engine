"""Flujos de efectivo de cumplimiento y unidades de cobertura.

Los **flujos de cumplimiento** son las entradas y salidas esperadas dentro del
limite del contrato, ponderadas por probabilidad. Se mantienen con el signo
economico natural (entradas positivas, salidas negativas) y se expone
``net_outflow`` para no tener que recordar la convencion en cada calculo.

Las **unidades de cobertura** merecen atencion aparte: son la base de la
liberacion del CSM (parrafo B119) y la norma no prescribe como medirlas. La
eleccion cambia el perfil de reconocimiento del resultado, asi que se exige
suministrarlas de forma explicita en lugar de asumir un reparto lineal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["FulfilmentCashFlows"]


@dataclass(frozen=True)
class FulfilmentCashFlows:
    """Flujos de cumplimiento de un grupo de contratos.

    Parameters
    ----------
    premiums:
        Primas esperadas por periodo (entradas, positivas).
    claims:
        Siniestros esperados por periodo (salidas, positivas en magnitud).
    expenses:
        Gastos atribuibles por periodo (salidas, positivas en magnitud).
    acquisition_cash_flows:
        Flujos de adquisicion, pagados en el reconocimiento inicial.
    coverage_units:
        Unidades de cobertura por periodo. Determinan el reparto del CSM.
    """

    premiums: np.ndarray
    claims: np.ndarray
    expenses: np.ndarray
    acquisition_cash_flows: float = 0.0
    coverage_units: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = {}
        for name in ("premiums", "claims", "expenses"):
            arr = np.asarray(getattr(self, name), dtype=float)
            if arr.ndim != 1:
                raise ValueError(f"{name} debe ser un vector")
            arrays[name] = arr
        sizes = {k: v.size for k, v in arrays.items()}
        if len(set(sizes.values())) != 1:
            raise ValueError(f"los flujos deben tener la misma longitud, se recibio {sizes}")
        for name, arr in arrays.items():
            object.__setattr__(self, name, arr)

        n = arrays["premiums"].size
        if self.coverage_units is None:
            raise ValueError(
                "coverage_units es obligatorio: la norma no prescribe una medida y la "
                "eleccion determina el perfil de reconocimiento del CSM, asi que no "
                "se asume un reparto lineal por omision"
            )
        cu = np.asarray(self.coverage_units, dtype=float)
        if cu.size != n:
            raise ValueError(f"coverage_units debe tener {n} elementos, tiene {cu.size}")
        if np.any(cu < 0.0):
            raise ValueError("las unidades de cobertura no pueden ser negativas")
        if cu.sum() <= 0.0:
            raise ValueError("la suma de unidades de cobertura debe ser positiva")
        object.__setattr__(self, "coverage_units", cu)
        if self.acquisition_cash_flows < 0.0:
            raise ValueError("acquisition_cash_flows se expresa en magnitud positiva")

    @property
    def n_periods(self) -> int:
        """Numero de periodos de proyeccion."""
        return int(self.premiums.size)

    @property
    def net_outflow(self) -> np.ndarray:
        """Salida neta por periodo: siniestros mas gastos menos primas.

        Un valor positivo significa que el grupo consume efectivo en ese periodo.
        """
        return self.claims + self.expenses - self.premiums

    def with_scaled_claims(self, factor: float) -> FulfilmentCashFlows:
        """Copia con los siniestros escalados.

        Util para representar un cambio de supuestos sin mutar el original, que
        es lo que permite reconstruir el roll-forward de forma auditable.
        """
        return FulfilmentCashFlows(
            premiums=self.premiums,
            claims=self.claims * float(factor),
            expenses=self.expenses,
            acquisition_cash_flows=self.acquisition_cash_flows,
            coverage_units=self.coverage_units,
        )

    def remaining_coverage_units(self, from_period: int) -> float:
        """Unidades de cobertura pendientes a partir de ``from_period``."""
        return float(np.sum(self.coverage_units[from_period:]))
