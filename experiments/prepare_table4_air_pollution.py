from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


POLLUTANTS = ("PM2.5", "PM10", "SO2", "NO2", "CO")
WEATHER_COLUMNS = ("TEMP", "PRES", "DEWP", "RAIN", "WSPM")
WIND_DIRECTIONS = (
    "N",
    "NNE",
    "NE",
    "ENE",
    "E",
    "ESE",
    "SE",
    "SSE",
    "S",
    "SSW",
    "SW",
    "WSW",
    "W",
    "WNW",
    "NW",
    "NNW",
)


@dataclass
class StationData:
    name: str
    timestamps: Tuple[str, ...]
    numeric: Dict[str, np.ndarray]
    wind_direction: Tuple[str, ...]


def parse_float(value: str) -> float:
    value = value.strip()
    if not value or value.lower() in {"na", "nan"}:
        return math.nan
    return float(value)


def read_station_csv(path: Path) -> StationData:
    timestamps: List[str] = []
    wind_direction: List[str] = []
    values = {name: [] for name in (*POLLUTANTS, *WEATHER_COLUMNS)}
    station_name = ""

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "year",
            "month",
            "day",
            "hour",
            "station",
            "wd",
            *POLLUTANTS,
            *WEATHER_COLUMNS,
        }
        missing = sorted(required - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        for row in reader:
            current_station = row["station"].strip()
            if station_name and current_station != station_name:
                raise ValueError(f"{path} contains multiple station names")
            station_name = current_station
            timestamps.append(
                f"{int(row['year']):04d}-{int(row['month']):02d}-"
                f"{int(row['day']):02d}T{int(row['hour']):02d}:00:00"
            )
            for name in values:
                values[name].append(parse_float(row[name]))
            wind_direction.append(row["wd"].strip())

    if not station_name or not timestamps:
        raise ValueError(f"{path} contains no station observations")

    return StationData(
        name=station_name,
        timestamps=tuple(timestamps),
        numeric={
            name: np.asarray(column, dtype=np.float32)
            for name, column in values.items()
        },
        wind_direction=tuple(wind_direction),
    )


def load_stations(raw_dir: Path) -> Dict[str, StationData]:
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No station CSV files found in {raw_dir}")

    stations: Dict[str, StationData] = {}
    reference_timestamps: Tuple[str, ...] | None = None
    for path in paths:
        station = read_station_csv(path)
        if station.name in stations:
            raise ValueError(f"Duplicate station: {station.name}")
        if reference_timestamps is None:
            reference_timestamps = station.timestamps
        elif station.timestamps != reference_timestamps:
            raise ValueError(f"Timestamp mismatch in {path}")
        stations[station.name] = station

    return dict(sorted(stations.items()))


def encode_wind_direction(values: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    direction_to_index = {name: index for index, name in enumerate(WIND_DIRECTIONS)}
    sine = np.full(len(values), np.nan, dtype=np.float32)
    cosine = np.full(len(values), np.nan, dtype=np.float32)

    for index, value in enumerate(values):
        if not value or value.lower() in {"na", "nan"}:
            continue
        if value not in direction_to_index:
            raise ValueError(f"Unknown wind direction: {value}")
        angle = direction_to_index[value] * (2.0 * math.pi / len(WIND_DIRECTIONS))
        sine[index] = math.sin(angle)
        cosine[index] = math.cos(angle)

    return sine, cosine


def build_feature_matrix(
    stations: Dict[str, StationData],
    target_station: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Tuple[str, ...]]:
    if target_station not in stations:
        raise ValueError(
            f"Unknown target station {target_station!r}; choices: {sorted(stations)}"
        )

    feature_columns: List[np.ndarray] = []
    feature_names: List[str] = []
    for station_name, station in stations.items():
        for pollutant in POLLUTANTS:
            feature_columns.append(station.numeric[pollutant])
            feature_names.append(f"{station_name}.{pollutant}")

    target = stations[target_station]
    for weather_name in WEATHER_COLUMNS:
        feature_columns.append(target.numeric[weather_name])
        feature_names.append(f"{target_station}.{weather_name}")

    wind_sine, wind_cosine = encode_wind_direction(target.wind_direction)
    feature_columns.extend((wind_sine, wind_cosine))
    feature_names.extend((f"{target_station}.wd_sin", f"{target_station}.wd_cos"))

    matrix = np.column_stack(feature_columns).astype(np.float32, copy=False)
    raw_target = target.numeric["PM2.5"].copy()
    target_is_observed = np.isfinite(raw_target)
    return matrix, raw_target, target_is_observed, feature_names, target.timestamps


def causal_impute(
    values: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward fill from the past, then fill leading gaps with train medians."""

    if values.ndim != 2:
        raise ValueError("causal_impute expects a two-dimensional matrix")
    if not 0 < train_end <= values.shape[0]:
        raise ValueError("train_end must lie inside the time axis")

    original = values.astype(np.float32, copy=True)
    result = original.copy()
    missing_counts = np.sum(~np.isfinite(original), axis=0).astype(np.int64)
    medians = np.empty(result.shape[1], dtype=np.float32)

    row_indices = np.arange(result.shape[0])
    for column_index in range(result.shape[1]):
        column = result[:, column_index]
        train_values = original[:train_end, column_index]
        median = np.nanmedian(train_values)
        if not np.isfinite(median):
            raise ValueError(f"Feature column {column_index} has no finite training values")
        medians[column_index] = median

        valid = np.isfinite(column)
        previous = np.maximum.accumulate(np.where(valid, row_indices, -1))
        has_previous = previous >= 0
        column[has_previous] = column[previous[has_previous]]
        column[~has_previous] = median

    if not np.isfinite(result).all():
        raise RuntimeError("Imputation left non-finite values")
    return result, medians, missing_counts


def minmax_scale(
    values: np.ndarray,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_values = values[:train_end]
    minimum = np.min(train_values, axis=0)
    maximum = np.max(train_values, axis=0)
    scale = maximum - minimum
    scale = np.where(scale > 0.0, scale, 1.0)
    scaled = (values - minimum) / scale
    return scaled.astype(np.float32), minimum.astype(np.float32), maximum.astype(np.float32)


def make_sequence_splits(
    scaled_features: np.ndarray,
    scaled_target: np.ndarray,
    target_is_observed: np.ndarray,
    sequence_length: int,
    train_end: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length <= 0 or sequence_length >= scaled_features.shape[0]:
        raise ValueError("sequence_length must be between 1 and the number of timestamps")

    windows = np.lib.stride_tricks.sliding_window_view(
        scaled_features,
        window_shape=sequence_length,
        axis=0,
    )[:-1].transpose(0, 2, 1)
    prediction_indices = np.arange(sequence_length, scaled_features.shape[0])
    observed = target_is_observed[prediction_indices]
    train_mask = observed & (prediction_indices < train_end)
    test_mask = observed & (prediction_indices >= train_end)

    x_train = np.ascontiguousarray(windows[train_mask], dtype=np.float32)
    y_train = np.ascontiguousarray(scaled_target[prediction_indices[train_mask]], dtype=np.float32)
    x_test = np.ascontiguousarray(windows[test_mask], dtype=np.float32)
    y_test = np.ascontiguousarray(scaled_target[prediction_indices[test_mask]], dtype=np.float32)
    return (
        x_train,
        y_train,
        x_test,
        y_test,
        prediction_indices[train_mask],
        prediction_indices[test_mask],
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(args: argparse.Namespace) -> Tuple[Path, Path]:
    raw_dir = Path(args.raw_dir)
    output = Path(args.output)
    metadata_path = output.with_suffix(".metadata.json")
    if (output.exists() or metadata_path.exists()) and not args.force:
        raise FileExistsError(f"Output exists; pass --force to replace {output}")
    if not 0.5 <= args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be in [0.5, 1.0)")

    stations = load_stations(raw_dir)
    if args.require_station_count and len(stations) != args.require_station_count:
        raise ValueError(
            f"Expected {args.require_station_count} stations, found {len(stations)}"
        )

    raw_features, raw_target, target_is_observed, feature_names, timestamps = (
        build_feature_matrix(stations, args.target_station)
    )
    if args.require_feature_count and raw_features.shape[1] != args.require_feature_count:
        raise ValueError(
            f"Expected {args.require_feature_count} features, found {raw_features.shape[1]}"
        )

    train_end = int(raw_features.shape[0] * args.train_ratio)
    imputed_features, medians, missing_counts = causal_impute(raw_features, train_end)
    scaled_features, feature_minimum, feature_maximum = minmax_scale(
        imputed_features,
        train_end,
    )

    target_column = feature_names.index(f"{args.target_station}.PM2.5")
    filled_target = imputed_features[:, target_column]
    target_minimum = float(np.min(filled_target[:train_end]))
    target_maximum = float(np.max(filled_target[:train_end]))
    target_range = target_maximum - target_minimum
    if target_range <= 0.0:
        raise ValueError("Training target has zero range")
    scaled_target = ((filled_target - target_minimum) / target_range).astype(np.float32)

    x_train, y_train, x_test, y_test, train_indices, test_indices = make_sequence_splits(
        scaled_features,
        scaled_target,
        target_is_observed,
        args.sequence_length,
        train_end,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        feature_names=np.asarray(feature_names),
    )

    metadata = {
        "protocol": "table4-beijing-multisite-v1",
        "source": "UCI Beijing Multi-Site Air Quality",
        "source_doi": "10.24432/C5RK5G",
        "raw_dir": str(raw_dir.resolve()),
        "output_file": str(output.resolve()),
        "output_sha256": sha256(output),
        "station_order": list(stations),
        "target_station": args.target_station,
        "target": "next-hour PM2.5",
        "sequence_length": args.sequence_length,
        "train_ratio": args.train_ratio,
        "train_end_index": train_end,
        "train_end_timestamp_exclusive": timestamps[train_end],
        "time_range": [timestamps[0], timestamps[-1]],
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "feature_construction": {
            "pollution": "12 stations x PM2.5, PM10, SO2, NO2, CO",
            "weather": f"{args.target_station} TEMP, PRES, DEWP, RAIN, WSPM",
            "wind": f"{args.target_station} wind direction encoded as sine and cosine",
        },
        "imputation": (
            "inputs are forward-filled from past observations; leading gaps use "
            "training-segment medians; windows with missing prediction labels are dropped"
        ),
        "normalization": "min-max fitted on the training time segment only",
        "feature_training_median": medians.tolist(),
        "feature_training_minimum": feature_minimum.tolist(),
        "feature_training_maximum": feature_maximum.tolist(),
        "feature_missing_count": dict(zip(feature_names, missing_counts.tolist())),
        "target_training_minimum": target_minimum,
        "target_training_maximum": target_maximum,
        "target_missing_count": int(np.sum(~target_is_observed)),
        "arrays": {
            "x_train": list(x_train.shape),
            "y_train": list(y_train.shape),
            "x_test": list(x_test.shape),
            "y_test": list(y_test.shape),
        },
        "prediction_boundaries": {
            "first_train": timestamps[int(train_indices[0])],
            "last_train": timestamps[int(train_indices[-1])],
            "first_test": timestamps[int(test_indices[0])],
            "last_test": timestamps[int(test_indices[-1])],
        },
        "paper_dataset_status": (
            "public proxy protocol; the Mapping Networks paper does not identify or "
            "release its original Table 4 air-pollution dataset"
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print("Table 4 air-pollution preprocessing")
    print("=" * 80)
    print(f"stations             = {len(stations)}")
    print(f"timestamps           = {raw_features.shape[0]:,}")
    print(f"features             = {raw_features.shape[1]}")
    print(f"sequence_length      = {args.sequence_length}")
    print(f"x_train / y_train    = {x_train.shape} / {y_train.shape}")
    print(f"x_test / y_test      = {x_test.shape} / {y_test.shape}")
    print(f"missing target rows  = {np.sum(~target_is_observed):,}")
    print(f"saved data           = {output}")
    print(f"saved metadata       = {metadata_path}")
    return output, metadata_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the public proxy dataset for Mapping Networks Table 4"
    )
    parser.add_argument(
        "--raw-dir",
        default="data/beijing_multi_site_air_quality/raw",
    )
    parser.add_argument(
        "--output",
        default="data/beijing_multi_site_air_quality/air_pollution_sequences.npz",
    )
    parser.add_argument("--target-station", default="Aotizhongxin")
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--require-station-count", type=int, default=12)
    parser.add_argument("--require-feature-count", type=int, default=67)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    prepare(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
