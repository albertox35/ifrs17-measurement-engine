# ifrs17-measurement-engine

Motor de medición IFRS 17: modelo general (GMM) y enfoque de asignación de prima
(PAA), con la unidad de cuenta, el ajuste por riesgo, el componente de pérdida y
el análisis de movimientos del párrafo 101.

El eje del diseño son las **tres reglas que definen el CSM**. Están codificadas y
verificadas por pruebas, no supuestas:

1. **El CSM nunca es negativo** (párrafo 38). Si los flujos de cumplimiento más el
   ajuste por riesgo exceden las primas, el grupo es oneroso: el CSM es cero, la
   pérdida se reconoce de inmediato y nace un componente de pérdida que se sigue
   por separado.
2. **El interés se devenga a la tasa bloqueada** en el reconocimiento inicial
   (párrafo B72(b)), no a la tasa corriente.
3. **Un cambio favorable revierte primero el componente de pérdida** y solo
   después restablece CSM (párrafo 50).

```python
import numpy as np
from ifrs17 import DiscountCurve, FulfilmentCashFlows, measure, movement_analysis

flujos = FulfilmentCashFlows(
    premiums=np.array([1200.0, 0.0, 0.0, 0.0, 0.0]),
    claims=np.full(5, 140.0),
    expenses=np.full(5, 15.0),
    acquisition_cash_flows=60.0,
    coverage_units=np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
)

resultado = measure(flujos, DiscountCurve.flat(0.04, 5))
print(movement_analysis(resultado))
print(resultado.closing_csm)          # 0.0: el CSM se libera por completo
print(resultado.reconciles())         # True: cada línea cuadra
```

---

## Qué hace

| Módulo | Contenido |
|---|---|
| `ifrs17.gmm` | Reconocimiento inicial y *roll-forward* de CSM con devengo, cambios de servicio futuro, liberación y componente de pérdida |
| `ifrs17.paa` | Enfoque simplificado **con su test de elegibilidad**, que es la parte que suele omitirse |
| `ifrs17.grouping` | Unidad de cuenta: cartera, cohorte anual y grupo de rentabilidad (párrafos 14-24) |
| `ifrs17.risk_adjustment` | Ajuste por riesgo por costo de capital y nivel de confianza equivalente (párrafo 119) |
| `ifrs17.discounting` | Curva corriente y curva bloqueada, mantenidas separadas |
| `ifrs17.reporting` | Análisis de movimientos del párrafo 101 y verificación de la conciliación |
| `ifrs17.audit` | Hash criptográfico de insumos y registro de linaje |

### La unidad de cuenta no es opcional

Es la pieza que faltaba en el diseño original, y sin ella no se está aplicando la
norma. IFRS 17 no permite medir un libro como un todo: obliga a separar por
cartera, por cohorte anual y por rentabilidad. La consecuencia económica es la
**prohibición de compensar**, y es asimétrica:

- El beneficio de un grupo rentable **se difiere** a lo largo de la cobertura vía
  el CSM.
- La pérdida de un grupo oneroso **se reconoce de inmediato**.

Si los dos pudieran agruparse, la pérdida quedaría escondida. En el ejemplo
incluido, cuatro contratos producen cuatro grupos distintos: ni las cohortes ni
las rentabilidades se mezclan.

### Trazabilidad, concretada

«Trazabilidad auditable» solo significa algo si se puede verificar. Aquí son dos
cosas comprobables:

- Un **SHA-256 del conjunto completo de insumos** —flujos, curva, parámetros de
  riesgo y cambios de supuestos—, construido sobre una representación canónica
  para que sea estable entre plataformas y ejecuciones.
- Un **registro de linaje** que un verificador usa con `traza.matches(flujos,
  curva)`: toma los supuestos del archivo y confirma que producen la cifra
  publicada. La traza reconstruye por sí misma lo que ya tiene registrado, para
  que una omisión del verificador no produzca un falso negativo.

---

## Resultados de validación

44 pruebas. Las que importan son identidades contables exactas, no comparaciones
aproximadas:

| Propiedad verificada | Resultado |
|---|---|
| CSM con tasa y ajuste por riesgo nulos = primas − siniestros − gastos − adquisición | exacto |
| Pasivo en reconocimiento inicial = componente de pérdida | exacto |
| **Conservación del CSM**: liberado total = CSM inicial + interés devengado | exacto (422.07 = 422.07 en el ejemplo) |
| CSM al cierre de la cobertura = 0 | exacto |
| Conciliación del párrafo 101, línea por línea | error máximo 0.00 |
| Cierre de un periodo = apertura del siguiente | exacto |
| CSM ≥ 0 bajo un choque adverso de 900 sobre un CSM de 166 | se agota y crea pérdida de 726.91 |
| Cambio favorable menor que la pérdida vigente | revierte pérdida, no restablece CSM |
| Ingreso total bajo PAA = prima total | exacto |
| El hash detecta un cambio de 0.01 % en los siniestros | sí |

La prueba de conservación del CSM es la más valiosa del conjunto: cualquier fuga
en el *roll-forward* —un devengo mal aplicado, una liberación mal proporcionada,
un arrastre perdido— la rompe de inmediato.

---

## Qué no hace

- **Solo GMM y PAA. No implementa VFA** (*variable fee approach*, párrafos
  B101-B118) para contratos con participación directa. El VFA exige modelar los
  elementos subyacentes y la participación en su rendimiento, que es un motor de
  proyección de activos: otro problema.
- **Sin enfoques de transición.** No implementa el retrospectivo completo, el de
  valor razonable ni el retrospectivo modificado (párrafos C1-C24). El motor mide
  desde el reconocimiento inicial.
- **Sin opción OCI.** Los cambios en la tasa de descuento no se desagregan entre
  resultados y otro resultado integral (párrafo 88(b)). Implementarlo exige
  mantener la curva histórica completa por cohorte.
- **El ajuste por riesgo usa un proxy declarado de capital**: una fracción de los
  flujos pendientes. No es un cálculo de capital económico; un modelo real exige
  la estructura de riesgos completa. El parámetro está expuesto, no escondido.
- **Sin reaseguro mantenido** (párrafos 60-70), que se mide por separado y con
  signo invertido.
- **Sin contabilidad de cobertura ni presentación en estados financieros.** El
  motor produce la conciliación; no genera los estados.
- **El límite del contrato se toma como dado.** Determinar qué flujos están dentro
  del límite (párrafos 34-35) es un juicio contractual, no un cálculo.

---

## Alcance y decisiones de diseño

**Las unidades de cobertura son obligatorias, sin valor por omisión.** La norma no
prescribe cómo medirlas (párrafo B119) y la elección determina el perfil de
reconocimiento del resultado. Asumir un reparto lineal en silencio escondería una
decisión contable material dentro de un valor por defecto, así que
`FulfilmentCashFlows` las exige de forma explícita.

**El PAA exige pasar su test de elegibilidad.** `measure_paa` lanza una excepción
si el grupo no es elegible, en lugar de producir una medición que la norma no
permite. El veredicto incluye el motivo, para que quede traza de por qué se aplicó
el simplificado.

**La curva de descuento es inmutable.** Una curva bloqueada que pudiera mutarse por
accidente llevaría los movimientos de tasa al resultado por la vía equivocada.

**El repositorio es autocontenido.** No depende de una librería núcleo compartida
con los demás repositorios del conjunto. La contrapartida es duplicación numérica
entre repositorios; la ventaja, que este se clona y se audita por sí solo.

---

## Instalación

```bash
pip install -e ".[dev]"
```

```bash
python -m pytest -q
```

```bash
python examples/onerous_vs_profitable.py
```

Requiere Python 3.11 o superior, NumPy, SciPy y pandas.

---

## Referencias

- IASB (2017, con enmiendas de 2020 y 2022). *NIIF 17 Contratos de Seguro*.
  Párrafos citados en el código: 14-24 (unidad de cuenta), 32-38 (reconocimiento
  inicial), 47-52 (contratos onerosos), 53-59 (PAA), 83 y B120-B124 (ingreso de
  seguro), 101 (conciliación), 119 (nivel de confianza), B72 (tasas), B96
  (cambios de supuestos), B119 (unidades de cobertura).
- IASB. *IFRS 17 Basis for Conclusions*, BC18-BC25 sobre la unidad de cuenta.
- CNSF. *Circular Única de Seguros y Fianzas*, para la interacción con la reserva
  estatutaria mexicana.

## Licencia

MIT. Ver [LICENSE](LICENSE).
