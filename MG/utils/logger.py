#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 13:40:54 2025

@author: petermillington
"""
# utils/logger.py
import os
import json
import shutil
import random
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd


# -----------------------------
# Simple append-only text logger
# -----------------------------
def log_simulation(metadata: List[str], log_path: str = "logs/simulation_log.txt") -> None:
    """
    Append human-readable metadata lines to a single rolling log file.
    Keeps compatibility with existing modules that already use it.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write("\n" + "=" * 60 + "\n")
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n")
        for line in metadata:
            f.write(str(line) + "\n")
        f.write("=" * 60 + "\n")


# -----------------------------
# General-purpose run logger
# -----------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class RunLogger:
    """
    General-purpose, module-agnostic run logger.

    Creates a timestamped run directory under base_save_dir and provides utilities to:
      - log_params(dict) -> params.json
      - log_metrics(dict, step=None) -> metrics.csv (append)
      - log_table(df, name) -> name.csv
      - log_artifact(path, dest_subdir='artifacts') -> copies a file into the run folder
      - save_dict(name, data) -> name.json
      - write_text(name, lines) -> name.txt
      - close() -> stamps end_time in run_info.json

    This can be reused by any module (CompareScaledBinary, phase diagrams, etc.).
    """

    def __init__(
        self,
        base_save_dir: str = "simulation_runs",
        module: Optional[str] = None,
        run_id: Optional[str] = None,
        payoff: Optional[str] = None,
        tags: Optional[List[str]] = None,
        extra_info: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
    ):
        ts = _ts()
        folder_bits = [ts]
        if module:
            folder_bits.append(module)
        if payoff:
            folder_bits.append(payoff)
        if run_id:
            folder_bits.append(run_id)

        folder_name = "_".join(folder_bits)
        self.save_dir = os.path.join(base_save_dir, folder_name)
        os.makedirs(self.save_dir, exist_ok=True)

        self._metrics_path = os.path.join(self.save_dir, "metrics.csv")
        self._run_info_path = os.path.join(self.save_dir, "run_info.json")
        self._metrics_header_written = False
        
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.run_info = {
            "start_time": ts,
            "end_time": None,
            "module": module,
            "payoff": payoff,
            "run_id": run_id,
            "tags": tags or [],
            "extra_info": extra_info or {},
            "seed": seed,
            "save_dir": self.save_dir,
        }
        self._write_run_info()

    # -------- Core writers --------
    def _write_run_info(self) -> None:
        with open(self._run_info_path, "w") as f:
            json.dump(self.run_info, f, indent=2)

    def close(self) -> None:
        self.run_info["end_time"] = _ts()
        self._write_run_info()

    # -------- Logging methods --------
    def log_params(self, params: Dict[str, Any], filename: str = "params.json") -> str:
        path = os.path.join(self.save_dir, filename)
        with open(path, "w") as f:
            json.dump(params, f, indent=2)
        return path

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> str:
        """
        Append a row of metrics to metrics.csv. If step is provided, it's included as a column.
        The CSV header is inferred from the first call and preserved thereafter.
        """
        row = {"step": step} if step is not None else {}
        row.update(metrics)

        df_row = pd.DataFrame([row])
        if not os.path.exists(self._metrics_path):
            df_row.to_csv(self._metrics_path, index=False)
            self._metrics_header_written = True
        else:
            # Ensure all columns stay aligned: reindex existing header if needed
            if not self._metrics_header_written:
                # Load header to align columns
                existing = pd.read_csv(self._metrics_path, nrows=0)
                df_row = df_row.reindex(columns=list(existing.columns), fill_value=None)
                # Also add any new columns at the end
                for c in df_row.columns:
                    if c not in existing.columns:
                        existing[c] = None
                existing.to_csv(self._metrics_path, index=False)
                self._metrics_header_written = True
            df_row.to_csv(self._metrics_path, mode="a", header=False, index=False)
        return self._metrics_path

    def log_table(self, df: pd.DataFrame, name: str) -> str:
        """
        Save a full table as CSV under the run dir.
        """
        if not name.lower().endswith(".csv"):
            name += ".csv"
        path = os.path.join(self.save_dir, name)
        df.to_csv(path, index=False)
        return path

    def log_artifact(self, src_path: str, dest_subdir: str = "artifacts") -> str:
        """
        Copy any file (plots, PDFs, pickles, etc.) into the run folder under dest_subdir.
        """
        dest_dir = os.path.join(self.save_dir, dest_subdir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_path = os.path.join(dest_dir, os.path.basename(src_path))
        shutil.copy2(src_path, dest_path)
        return dest_path

    def save_dict(self, name: str, data: Dict[str, Any]) -> str:
        """
        Save arbitrary dict as JSON (e.g., summary stats).
        """
        if not name.lower().endswith(".json"):
            name += ".json"
        path = os.path.join(self.save_dir, name)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def write_text(self, name: str, lines: List[str]) -> str:
        """
        Save an arbitrary text file (e.g., notes or a human-readable summary).
        """
        if not name.lower().endswith(".txt"):
            name += ".txt"
        path = os.path.join(self.save_dir, name)
        with open(path, "w") as f:
            f.write("\n".join(str(x) for x in lines) + "\n")
        return path

    # -------- Convenience getters --------
    def get_dir(self) -> str:
        return self.save_dir
