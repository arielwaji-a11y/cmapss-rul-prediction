"""Modèle de coûts paramétrable et simulation des stratégies de maintenance.

Toutes les valeurs en dollars sont des HYPOTHÈSES de travail, à documenter dans
le rapport et à soumettre à une analyse de sensibilité. Elles ne sont pas
des chiffres réels de l'industrie.

La simulation se fait sur des moteurs dont on connaît le cycle de panne
(moteurs de validation tirés de l'entraînement), avec des prédictions hors
échantillon pour chaque cycle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class CostAssumptions:
    planned_maintenance: float = 50_000.0     # maintenance planifiée ($ par intervention)
    unplanned_failure: float = 500_000.0      # panne non planifiée : réparation + immobilisation (AOG)
    value_per_cycle: float = 1_000.0          # valeur d'un cycle de vie perdu par remplacement précoce
    lead_time: int = 5                        # cycles entre la décision et l'intervention

    def as_dict(self) -> dict:
        return asdict(self)


def _engine_cost(failure_cycle: int, maintenance_cycle: float | None, a: CostAssumptions) -> dict:
    """Coût d'un moteur selon le cycle où la maintenance a lieu (None = jamais)."""
    if maintenance_cycle is None or maintenance_cycle >= failure_cycle:
        return {"panne": 1, "cycles_perdus": 0, "coût": a.unplanned_failure}
    lost = failure_cycle - maintenance_cycle
    return {"panne": 0, "cycles_perdus": lost, "coût": a.planned_maintenance + lost * a.value_per_cycle}


def _summarize(rows: list[dict], strategy: str, param) -> dict:
    df = pd.DataFrame(rows)
    return {
        "stratégie": strategy,
        "paramètre": param,
        "coût total": df["coût"].sum(),
        "coût par moteur": df["coût"].mean(),
        "pannes": int(df["panne"].sum()),
        "cycles perdus moyens": df["cycles_perdus"].mean(),
    }


def fixed_interval(failure_cycles: pd.Series, interval: int, a: CostAssumptions) -> dict:
    """Maintenance systématique au cycle `interval`, quel que soit l'état du moteur."""
    rows = [_engine_cost(fc, interval, a) for fc in failure_cycles]
    return _summarize(rows, "intervalle fixe", interval)


def ideal(failure_cycles: pd.Series, a: CostAssumptions) -> dict:
    """Connaissance parfaite : intervention au dernier cycle sûr (borne inférieure du coût)."""
    rows = [_engine_cost(fc, fc - 1, a) for fc in failure_cycles]
    return _summarize(rows, "idéale", None)


def predictive(preds: pd.DataFrame, threshold: float, a: CostAssumptions) -> dict:
    """Maintenance déclenchée quand la RUL prédite passe sous `threshold`.

    `preds` doit contenir les colonnes : unit, cycle, failure_cycle, pred.
    L'intervention a lieu `lead_time` cycles après la décision.
    """
    rows = []
    for _, grp in preds.sort_values(["unit", "cycle"]).groupby("unit"):
        fc = int(grp["failure_cycle"].iloc[0])
        hit = grp.loc[grp["pred"] <= threshold, "cycle"]
        decision = int(hit.iloc[0]) if len(hit) else None
        rows.append(_engine_cost(fc, None if decision is None else decision + a.lead_time, a))
    return _summarize(rows, "prédictive", threshold)


def best_fixed_interval(failure_cycles: pd.Series, a: CostAssumptions, grid=range(50, 400, 5)) -> dict:
    """Intervalle fixe optimal a posteriori : c'est le concurrent le plus exigeant."""
    return min((fixed_interval(failure_cycles, t, a) for t in grid), key=lambda r: r["coût total"])


def threshold_sweep(preds: pd.DataFrame, a: CostAssumptions, thresholds=range(5, 101, 5)) -> pd.DataFrame:
    return pd.DataFrame([predictive(preds, t, a) for t in thresholds])


def compare_strategies(preds: pd.DataFrame, a: CostAssumptions | None = None) -> pd.DataFrame:
    """Compare intervalle fixe optimal, prédictive (seuil optimal) et idéale."""
    a = a or CostAssumptions()
    failures = preds.groupby("unit")["failure_cycle"].first()
    sweep = threshold_sweep(preds, a)
    best_pred = sweep.loc[sweep["coût total"].idxmin()].to_dict()
    table = pd.DataFrame([best_fixed_interval(failures, a), best_pred, ideal(failures, a)])
    fixed_cost = table.loc[0, "coût total"]
    table["économie vs intervalle fixe"] = fixed_cost - table["coût total"]
    return table


def sensitivity(preds: pd.DataFrame, base: CostAssumptions | None = None,
                failure_costs=(250_000, 500_000, 1_000_000),
                cycle_values=(500, 1_000, 2_000)) -> pd.DataFrame:
    """Économie de la stratégie prédictive selon les hypothèses de coûts."""
    base = base or CostAssumptions()
    rows = []
    for fcost in failure_costs:
        for cval in cycle_values:
            a = CostAssumptions(base.planned_maintenance, fcost, cval, base.lead_time)
            t = compare_strategies(preds, a)
            rows.append({
                "coût panne": fcost,
                "valeur d'un cycle": cval,
                "seuil optimal": t.loc[1, "paramètre"],
                "économie par moteur": t.loc[1, "économie vs intervalle fixe"] / preds["unit"].nunique(),
            })
    return pd.DataFrame(rows)
