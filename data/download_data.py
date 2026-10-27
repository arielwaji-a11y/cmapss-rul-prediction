"""Télécharge le jeu de données NASA C-MAPSS et extrait les fichiers .txt dans data/.

Usage :
    python data/download_data.py

Si le téléchargement échoue (lien déplacé par la NASA), télécharge manuellement
CMAPSSData.zip depuis https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data
et place-le dans data/ : le script l'extraira.

Citation : A. Saxena, K. Goebel, D. Simon et N. Eklund, « Damage Propagation
Modeling for Aircraft Engine Run-to-Failure Simulation », PHM08, Denver, 2008.
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
URLS = [
    "https://data.nasa.gov/docs/legacy/CMAPSSData.zip",
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
]
EXPECTED = [f"{kind}_FD00{i}.txt" for kind in ("train", "test", "RUL") for i in range(1, 5)]


def _extract(zf: zipfile.ZipFile) -> None:
    """Extrait les .txt, y compris depuis une archive imbriquée."""
    for name in zf.namelist():
        base = Path(name).name
        if name.lower().endswith(".zip"):
            _extract(zipfile.ZipFile(io.BytesIO(zf.read(name))))
        elif base in EXPECTED or base.lower().startswith("readme"):
            (DATA_DIR / base).write_bytes(zf.read(name))


def _missing() -> list[str]:
    return [f for f in EXPECTED if not (DATA_DIR / f).exists()]


def main() -> int:
    if not _missing():
        print("Les données sont déjà présentes dans", DATA_DIR)
        return 0

    local = DATA_DIR / "CMAPSSData.zip"
    if local.exists():
        print("Extraction de", local.name)
        _extract(zipfile.ZipFile(local))
    else:
        for url in URLS:
            try:
                print("Téléchargement :", url)
                with urllib.request.urlopen(url, timeout=120) as resp:
                    _extract(zipfile.ZipFile(io.BytesIO(resp.read())))
                if not _missing():
                    break
            except Exception as exc:  # lien mort, réseau, archive invalide
                print("  échec :", exc)

    missing = _missing()
    if missing:
        print("Fichiers manquants :", ", ".join(missing))
        print("Télécharge CMAPSSData.zip manuellement (voir l'en-tête de ce script) et relance.")
        return 1
    print("Données prêtes :", ", ".join(EXPECTED))
    return 0


if __name__ == "__main__":
    sys.exit(main())
