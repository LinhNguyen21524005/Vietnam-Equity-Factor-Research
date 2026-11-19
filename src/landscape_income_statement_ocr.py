"""Reusable Landscape Income Statement OCR module.

Pilot assumptions: four right-aligned numeric columns on a landscape page.
Dependencies: pandas, numpy, scikit-learn, pillow, pytesseract, Tesseract OCR.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
from pytesseract import Output
from sklearn.cluster import KMeans

NUMERIC_PATTERN = r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$"


@dataclass
class LandscapeOCRConfig:
    language: str = "vie+eng"
    primary_psm: int = 12
    retry_psm: int = 6
    n_numeric_columns: int = 4
    table_top_fraction: float = 0.31
    table_bottom_fraction: float = 0.88
    numeric_left_fraction: float = 0.47
    row_tolerance_factor: float = 0.65
    min_retry_confidence: float = 40.0

    def validate(self):
        if self.n_numeric_columns < 2:
            raise ValueError("n_numeric_columns must be at least 2.")
        if not 0 <= self.table_top_fraction < self.table_bottom_fraction <= 1:
            raise ValueError("Invalid table vertical fractions.")
        if not 0 <= self.numeric_left_fraction < 1:
            raise ValueError("numeric_left_fraction must be in [0, 1).")


@dataclass
class LandscapeOCRResult:
    tokens: pd.DataFrame
    numeric_tokens: pd.DataFrame
    reconstructed_long: pd.DataFrame
    reconstructed_wide: pd.DataFrame
    column_anchors: pd.DataFrame
    missing_cells: pd.DataFrame
    retry_log: pd.DataFrame


class LandscapeIncomeStatementOCR:
    """OCR a landscape Income Statement and reconstruct its numeric columns."""

    def __init__(
        self,
        config: Optional[LandscapeOCRConfig] = None,
        tesseract_cmd: Optional[str] = None,
    ):
        self.config = config or LandscapeOCRConfig()
        self.config.validate()
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    @staticmethod
    def _load_image(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    def _ocr(self, image, psm):
        df = pytesseract.image_to_data(
            image,
            lang=self.config.language,
            config=f"--oem 3 --psm {psm}",
            output_type=Output.DATAFRAME,
        )
        df = df.dropna(subset=["text"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df = df.loc[(df["text"] != "") & (pd.to_numeric(df["conf"], errors="coerce") >= 0)].copy()
        for c in ["left", "top", "width", "height", "conf"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["left", "top", "width", "height", "conf"])
        df["right"] = df["left"] + df["width"]
        df["x_center"] = df["left"] + df["width"] / 2
        df["y_center"] = df["top"] + df["height"] / 2
        return df.reset_index(drop=True)

    def _numeric_tokens(self, tokens, image_size):
        width, height = image_size
        mask = (
            tokens["text"].str.fullmatch(NUMERIC_PATTERN, na=False)
            & (tokens["y_center"] >= height * self.config.table_top_fraction)
            & (tokens["y_center"] <= height * self.config.table_bottom_fraction)
            & (tokens["right"] >= width * self.config.numeric_left_fraction)
        )
        numeric = tokens.loc[mask].copy()
        if len(numeric) < self.config.n_numeric_columns * 2:
            raise ValueError(
                "Too few numeric tokens. Check page orientation, OCR language, "
                "PSM and table-region fractions."
            )
        return numeric.reset_index(drop=True)

    def _anchors(self, numeric):
        model = KMeans(
            n_clusters=self.config.n_numeric_columns,
            random_state=42,
            n_init=20,
        )
        labels = model.fit_predict(numeric[["right"]].to_numpy())
        centers = model.cluster_centers_.flatten()
        ordered = np.sort(centers)
        numeric = numeric.copy()
        numeric["column"] = numeric["right"].apply(
            lambda x: int(np.argmin(np.abs(ordered - x)))
        )
        summary = (
            numeric.groupby("column")
            .agg(
                right_anchor=("right", "median"),
                min_right=("right", "min"),
                max_right=("right", "max"),
                token_count=("text", "count"),
            )
            .reindex(range(self.config.n_numeric_columns))
            .reset_index()
        )
        return numeric, ordered, summary

    @staticmethod
    def _assign_rows(numeric, tolerance):
        """Cluster by distance to current row median (avoids chaining)."""
        out = numeric.sort_values("y_center").copy()
        ids, row_id, row_ys = [], -1, []
        for y in out["y_center"].astype(float):
            if not row_ys or abs(y - float(np.median(row_ys))) > tolerance:
                row_id += 1
                row_ys = [y]
            else:
                row_ys.append(y)
            ids.append(row_id)
        out["row_id"] = ids
        return out

    @staticmethod
    def _reconstruct(numeric):
        records = []
        for (rid, col), group in numeric.groupby(["row_id", "column"], sort=True):
            group = group.sort_values("left")
            records.append({
                "row_id": int(rid),
                "column": int(col),
                "y_center": float(group["y_center"].median()),
                "value_text": "".join(group["text"].astype(str)),
                "token_count": len(group),
                "mean_confidence": float(group["conf"].mean()),
                "fragments": group["text"].astype(str).tolist(),
                "source": "primary_ocr",
            })
        long = pd.DataFrame(records)
        if long.empty:
            return long, pd.DataFrame()
        wide = long.pivot(index="row_id", columns="column", values="value_text")
        wide.columns = [f"column_{int(c)}" for c in wide.columns]
        return long, wide

    def _missing_cells(self, wide, numeric):
        columns = [f"column_{i}" for i in range(self.config.n_numeric_columns)]
        rows = []
        for rid, row in wide.iterrows():
            present = [c for c in columns if c in wide.columns and pd.notna(row.get(c))]
            if len(present) < self.config.n_numeric_columns - 1:
                continue
            yvals = numeric.loc[numeric["row_id"] == rid, "y_center"]
            if yvals.empty:
                continue
            for col, name in enumerate(columns):
                if name not in wide.columns or pd.isna(row.get(name)):
                    rows.append({"row_id": int(rid), "column": col,
                                 "y_center": float(yvals.median())})
        return pd.DataFrame(rows, columns=["row_id", "column", "y_center"])

    def _retry(self, image, missing, anchors, median_height):
        recovered, logs = [], []
        width, height = image.size
        spacing = float(np.median(np.diff(anchors)))
        half_w = max(45, int(spacing * 0.75))
        half_h = max(18, int(median_height * 2.0))

        for _, cell in missing.iterrows():
            col, y = int(cell["column"]), float(cell["y_center"])
            x_anchor = float(anchors[col])
            # Numeric values are right-aligned near the column anchor.
            box = (
                max(0, int(x_anchor - half_w)),
                max(0, int(y - half_h)),
                min(width, int(x_anchor + 8)),
                min(height, int(y + half_h)),
            )
            crop = image.crop(box)
            retry = self._ocr(crop, self.config.retry_psm)
            candidates = retry.loc[
                retry["text"].str.fullmatch(NUMERIC_PATTERN, na=False)
                & (retry["conf"] >= self.config.min_retry_confidence)
            ].copy()

            if candidates.empty:
                logs.append({"row_id": int(cell["row_id"]), "column": col,
                             "crop_box": box, "result": None,
                             "confidence": None, "status": "not_recovered"})
                continue

            candidates["score"] = (
                (crop.width - candidates["right"]).abs() / max(crop.width, 1)
                + (candidates["y_center"] - crop.height / 2).abs() / max(crop.height, 1)
            )
            best = candidates.sort_values("score").iloc[0]
            text = str(best["text"])
            recovered.append({
                "row_id": int(cell["row_id"]), "column": col,
                "y_center": y, "value_text": text, "token_count": 1,
                "mean_confidence": float(best["conf"]),
                "fragments": [text], "source": "targeted_ocr",
            })
            logs.append({"row_id": int(cell["row_id"]), "column": col,
                         "crop_box": box, "result": text,
                         "confidence": float(best["conf"]), "status": "recovered"})

        return pd.DataFrame(recovered), pd.DataFrame(logs)

    def extract(self, image, retry_missing=True):
        page = self._load_image(image)
        tokens = self._ocr(page, self.config.primary_psm)
        numeric = self._numeric_tokens(tokens, page.size)
        numeric, anchors, anchor_summary = self._anchors(numeric)

        tolerance = max(
            8.0,
            float(numeric["height"].median()) * self.config.row_tolerance_factor,
        )
        numeric = self._assign_rows(numeric, tolerance)

        long, wide = self._reconstruct(numeric)
        missing = self._missing_cells(wide, numeric)
        retry_log = pd.DataFrame()

        if retry_missing and not missing.empty:
            recovered, retry_log = self._retry(
                page, missing, anchors, float(numeric["height"].median())
            )
            if not recovered.empty:
                existing = set(zip(long["row_id"], long["column"]))
                recovered = recovered.loc[
                    ~recovered.apply(
                        lambda r: (int(r["row_id"]), int(r["column"])) in existing,
                        axis=1,
                    )
                ]
                long = pd.concat([long, recovered], ignore_index=True)
                long = long.sort_values(["row_id", "column"]).reset_index(drop=True)
                wide = long.pivot(index="row_id", columns="column", values="value_text")
                wide.columns = [f"column_{int(c)}" for c in wide.columns]
                missing = self._missing_cells(wide, numeric)

        return LandscapeOCRResult(
            tokens=tokens,
            numeric_tokens=numeric,
            reconstructed_long=long,
            reconstructed_wide=wide,
            column_anchors=anchor_summary,
            missing_cells=missing,
            retry_log=retry_log,
        )


def save_result(result, output_dir, prefix="landscape_is"):
    """Save all extraction outputs and diagnostics as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.tokens.to_csv(output_dir / f"{prefix}_all_tokens.csv", index=False)
    result.numeric_tokens.to_csv(output_dir / f"{prefix}_numeric_tokens.csv", index=False)
    result.reconstructed_long.to_csv(output_dir / f"{prefix}_reconstructed_long.csv", index=False)
    result.reconstructed_wide.to_csv(output_dir / f"{prefix}_reconstructed_wide.csv")
    result.column_anchors.to_csv(output_dir / f"{prefix}_column_anchors.csv", index=False)
    result.missing_cells.to_csv(output_dir / f"{prefix}_missing_cells.csv", index=False)
    result.retry_log.to_csv(output_dir / f"{prefix}_retry_log.csv", index=False)
