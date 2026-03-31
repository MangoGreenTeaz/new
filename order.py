import math
import random
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import polars as pl
from tqdm import tqdm


INPUT_PATH = Path("a_label.csv")
OUTPUT_PATH = Path("a_order.csv")
CHUNK_SIZE = 100_000
TIME_FORMAT = "%Y/%m/%d %H:%M"
RANDOM_SEED = 42
MAX_OFFSET_MINUTES = 5
MAX_GAP_HOURS = 12

ORDER_COLUMN = "order"
OUTPUT_COLUMNS = ["time", "udid", "text", "scene_label", "context", "history_usage", "service_click", ORDER_COLUMN]
REQUIRED_COLUMNS = ["time", "udid", "text", "scene_label"]

CITY_PATTERN = re.compile(r"城市：([^，,]+)")
FROM_CITY_PATTERN = re.compile(r"前\d+分钟从([^，,]+?)(跨城市|市内)")

TRAIN_LABELS = {
    "抵达始发高铁站",
    "高铁站候车",
    "高铁行程途中",
    "抵达终点高铁站",
    "离开终点高铁站",
}

FLIGHT_LABELS = {
    "抵达始发机场",
    "机场内活动",
    "飞机行程途中",
    "抵达终点机场",
    "离开终点机场",
}

HOTEL_LABELS = {
    "酒店办理入住",
    "旅游住宿休息",
}

TOURIST_LABELS = {
    "旅游参观",
    "旅游中途休息",
    "旅游中用餐",
}

ALL_ORDER_LABELS = TRAIN_LABELS | FLIGHT_LABELS | HOTEL_LABELS | TOURIST_LABELS


@dataclass(frozen=True)
class OrderRule:
    name: str
    labels: set[str]
    builder: Callable[[list[dict], random.Random], str]


def get_scene_type(label: str) -> str | None:
    if label in TRAIN_LABELS:
        return "train"
    if label in FLIGHT_LABELS:
        return "flight"
    if label in HOTEL_LABELS:
        return "hotel"
    if label in TOURIST_LABELS:
        return "tourist"
    return None


def build_order_rules() -> list[OrderRule]:
    return [
        OrderRule(name="train", labels=TRAIN_LABELS, builder=build_train_order),
        OrderRule(name="flight", labels=FLIGHT_LABELS, builder=build_flight_order),
        OrderRule(name="hotel", labels=HOTEL_LABELS, builder=build_hotel_order),
        OrderRule(name="tourist", labels=TOURIST_LABELS, builder=build_tourist_order),
    ]


def resolve_order_rule(label: str, rules: list[OrderRule]) -> OrderRule | None:
    for rule in rules:
        if label in rule.labels:
            return rule
    return None


def extract_current_city(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = CITY_PATTERN.search(text)
    return match.group(1).strip() if match else None


def extract_from_city(text: str | None) -> str | None:
    if not isinstance(text, str):
        return None
    match = FROM_CITY_PATTERN.search(text)
    return match.group(1).strip() if match else None


def parse_time(value: str | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.strptime(value, TIME_FORMAT)
    except ValueError:
        return None


def add_random_offset(rng: random.Random, dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    offset = rng.randint(-MAX_OFFSET_MINUTES, MAX_OFFSET_MINUTES)
    return dt + timedelta(minutes=offset)


def format_order(
    order_type: str,
    start_dt: datetime | None,
    end_dt: datetime | None,
    departure_city: str,
    arrival_city: str,
) -> str:
    start_str = start_dt.strftime("%Y/%m/%d %H:%M") if start_dt else ""
    end_str = end_dt.strftime("%Y/%m/%d %H:%M") if end_dt else ""
    return (
        f"订单类型：{order_type}，"
        f"出发时间：{start_str}，"
        f"到达时间：{end_str}，"
        f"出发城市：{departure_city}，"
        f"到达城市：{arrival_city}"
    )


def format_day_order(order_type: str, dt: datetime | None) -> str:
    day_str = dt.strftime("%Y/%m/%d") if dt else ""
    return f"订单类型：{order_type}，时间：{day_str}"


def find_first_city(rows: list[dict], indices: list[int], field: str = "current_city") -> str | None:
    for idx in indices:
        city = rows[idx].get(field)
        if isinstance(city, str) and city:
            return city
    return None


def build_train_order(rows: list[dict], rng: random.Random) -> str:
    waiting_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "高铁站候车"]
    depart_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "抵达始发高铁站"]
    onboard_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "高铁行程途中"]
    arrive_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "抵达终点高铁站"]
    leave_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "离开终点高铁站"]

    left_dt: datetime | None = None
    right_dt: datetime | None = None

    if waiting_indices:
        left_dt = rows[waiting_indices[-1]].get("_parsed_time")
    elif depart_indices:
        left_dt = rows[depart_indices[-1]].get("_parsed_time")

    if onboard_indices:
        right_dt = rows[onboard_indices[0]].get("_parsed_time")
    elif arrive_indices:
        right_dt = rows[arrive_indices[0]].get("_parsed_time")

    if left_dt is not None and right_dt is not None:
        start_dt = left_dt + (right_dt - left_dt) / 2
    elif left_dt is not None:
        start_dt = left_dt
    elif right_dt is not None:
        start_dt = right_dt
    else:
        start_dt = rows[0].get("_parsed_time")

    if arrive_indices:
        end_dt_raw = rows[arrive_indices[0]].get("_parsed_time")
        end_dt = end_dt_raw - timedelta(minutes=1) if end_dt_raw else None
    elif leave_indices:
        end_dt_raw = rows[leave_indices[0]].get("_parsed_time")
        end_dt = end_dt_raw - timedelta(minutes=1) if end_dt_raw else None
    else:
        end_dt = rows[-1].get("_parsed_time")

    start_dt = add_random_offset(rng, start_dt)
    end_dt = add_random_offset(rng, end_dt)

    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_side = sorted(depart_indices + waiting_indices, key=lambda i: rows[i].get("_parsed_time") or datetime.min)
    departure_city = find_first_city(rows, start_side)

    if not departure_city and onboard_indices:
        first_onboard = rows[onboard_indices[0]]
        departure_city = first_onboard.get("from_city") or first_onboard.get("current_city")

    if not departure_city:
        departure_city = rows[0].get("current_city")

    end_side = sorted(arrive_indices + leave_indices, key=lambda i: rows[i].get("_parsed_time") or datetime.min)
    arrival_city = find_first_city(rows, end_side)

    if not arrival_city:
        arrival_city = rows[-1].get("current_city")

    return format_order("火车", start_dt, end_dt, departure_city or "", arrival_city or "")


def build_flight_order(rows: list[dict], rng: random.Random) -> str:
    depart_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "抵达始发机场"]
    activity_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "机场内活动"]
    onboard_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "飞机行程途中"]
    arrive_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "抵达终点机场"]
    leave_indices = [i for i, r in enumerate(rows) if r["scene_label"] == "离开终点机场"]

    if activity_indices:
        raw = rows[activity_indices[-1]].get("_parsed_time")
        start_dt = raw + timedelta(minutes=1) if raw else None
    elif depart_indices:
        raw = rows[depart_indices[-1]].get("_parsed_time")
        start_dt = raw + timedelta(minutes=1) if raw else None
    elif onboard_indices:
        raw = rows[onboard_indices[0]].get("_parsed_time")
        start_dt = raw - timedelta(minutes=1) if raw else None
    else:
        start_dt = rows[0].get("_parsed_time")

    if arrive_indices:
        raw = rows[arrive_indices[0]].get("_parsed_time")
        end_dt = raw - timedelta(minutes=1) if raw else None
    elif leave_indices:
        raw = rows[leave_indices[0]].get("_parsed_time")
        end_dt = raw - timedelta(minutes=1) if raw else None
    else:
        end_dt = rows[-1].get("_parsed_time")

    start_dt = add_random_offset(rng, start_dt)
    end_dt = add_random_offset(rng, end_dt)

    if start_dt is not None and end_dt is not None and start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    start_side = sorted(depart_indices + activity_indices, key=lambda i: rows[i].get("_parsed_time") or datetime.min)
    departure_city = find_first_city(rows, start_side)

    if not departure_city and onboard_indices:
        first_onboard = rows[onboard_indices[0]]
        departure_city = first_onboard.get("from_city") or first_onboard.get("current_city")

    if not departure_city:
        departure_city = rows[0].get("current_city")

    end_side = sorted(arrive_indices + leave_indices, key=lambda i: rows[i].get("_parsed_time") or datetime.min)
    arrival_city = find_first_city(rows, end_side)

    if not arrival_city:
        arrival_city = rows[-1].get("current_city")

    return format_order("飞机", start_dt, end_dt, departure_city or "", arrival_city or "")


def build_hotel_order(rows: list[dict], rng: random.Random) -> str:
    del rng
    start_dt = next((row.get("_parsed_time") for row in rows if row.get("_parsed_time") is not None), None)
    return format_day_order("酒店", start_dt)


def build_tourist_order(rows: list[dict], rng: random.Random) -> str:
    del rng
    start_dt = next((row.get("_parsed_time") for row in rows if row.get("_parsed_time") is not None), None)
    return format_day_order("旅游", start_dt)


def split_segments(
    rows: list[dict],
) -> list[list[dict]]:
    if not rows:
        return []

    segments: list[list[dict]] = []
    current_segment: list[dict] = [rows[0]]
    prev_scene_type = rows[0].get("scene_type")
    prev_time = rows[0].get("_parsed_time")

    for row in rows[1:]:
        scene_type = row.get("scene_type")
        curr_time = row.get("_parsed_time")

        gap_too_large = (
            prev_time is not None
            and curr_time is not None
            and (curr_time - prev_time).total_seconds() > MAX_GAP_HOURS * 3600
        )
        scene_changed = scene_type != prev_scene_type

        if gap_too_large or scene_changed:
            segments.append(current_segment)
            current_segment = [row]
        else:
            current_segment.append(row)

        prev_time = curr_time
        prev_scene_type = scene_type

    if current_segment:
        segments.append(current_segment)

    return segments


def split_segment_by_day(rows: list[dict]) -> list[list[dict]]:
    if not rows:
        return []

    day_segments: list[list[dict]] = []
    current_segment: list[dict] = [rows[0]]
    prev_day = rows[0].get("_parsed_time").date() if rows[0].get("_parsed_time") is not None else None

    for row in rows[1:]:
        parsed_time = row.get("_parsed_time")
        curr_day = parsed_time.date() if parsed_time is not None else prev_day

        if curr_day != prev_day:
            day_segments.append(current_segment)
            current_segment = [row]
        else:
            current_segment.append(row)

        prev_day = curr_day

    if current_segment:
        day_segments.append(current_segment)

    return day_segments


def enrich_rows(rows: list[dict]) -> None:
    for row in rows:
        row["_parsed_time"] = parse_time(row.get("time"))
        row["current_city"] = row.get("city") or extract_current_city(row.get("text"))
        row["from_city"] = extract_from_city(row.get("text"))
        row["scene_type"] = get_scene_type(row.get("scene_label", ""))


def process_user_orders(user_rows: list[dict], rng: random.Random) -> None:
    order_rules = build_order_rules()
    order_rows = [row for row in user_rows if resolve_order_rule(row.get("scene_label", ""), order_rules) is not None]
    if not order_rows:
        return

    segments = split_segments(order_rows)

    for segment in segments:
        if not segment:
            continue

        rule = resolve_order_rule(segment[0].get("scene_label", ""), order_rules)
        if rule is None:
            continue

        if rule.name in {"hotel", "tourist"}:
            day_segments = split_segment_by_day(segment)
            for day_segment in day_segments:
                if not day_segment:
                    continue
                order_str = rule.builder(day_segment, rng)
                for row in day_segment:
                    row[ORDER_COLUMN] = order_str
            continue

        order_str = rule.builder(segment, rng)

        for row in segment:
            row[ORDER_COLUMN] = order_str


def process_batch(batch: pl.DataFrame, rng: random.Random) -> pl.DataFrame:
    if batch.height == 0:
        return batch.with_columns(pl.lit("").alias(ORDER_COLUMN))

    rows = batch.to_dicts()

    for row in rows:
        row[ORDER_COLUMN] = ""

    enrich_rows(rows)

    user_groups: dict[str, list[dict]] = {}
    for row in rows:
        udid = row.get("udid", "")
        if udid not in user_groups:
            user_groups[udid] = []
        user_groups[udid].append(row)

    for user_rows in user_groups.values():
        process_user_orders(user_rows, rng)

    order_values = [row[ORDER_COLUMN] for row in rows]
    result = batch.with_columns(pl.Series(ORDER_COLUMN, order_values))

    available_columns = [col for col in OUTPUT_COLUMNS if col in result.columns]
    return result.select(available_columns)


def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as infile:
        row_count = sum(1 for _ in infile)
    return max(row_count - 1, 0)


def validate_columns(csv_path: Path) -> None:
    header = pl.read_csv(csv_path, n_rows=0)
    missing = [col for col in REQUIRED_COLUMNS if col not in header.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")


def split_tail_user(batch: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if batch.height == 0:
        return batch, None

    if batch.height == 1:
        return batch.slice(0, 0), batch

    last_udid = batch.item(batch.height - 1, "udid")
    tail_mask = pl.col("udid") == pl.lit(last_udid)
    tail_batch = batch.filter(tail_mask)
    ready_batch = batch.filter(~tail_mask)
    return ready_batch, tail_batch


def process_csv(input_path: Path, output_path: Path, chunk_size: int) -> None:
    validate_columns(input_path)

    total_rows = count_data_rows(input_path)
    total_chunks = math.ceil(total_rows / chunk_size) if total_rows else 0

    rng = random.Random(RANDOM_SEED)

    if total_rows == 0:
        empty_schema = {col: pl.Utf8 for col in OUTPUT_COLUMNS}
        pl.DataFrame(schema=empty_schema).write_csv(output_path)
        return

    reader = pl.read_csv_batched(
        input_path,
        batch_size=chunk_size,
        encoding="utf8",
        schema_overrides={"scene_label": pl.Utf8},
    )

    pending_tail: pl.DataFrame | None = None

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        chunk_bar = tqdm(total=total_chunks, desc="Generating orders", unit="chunk")
        wrote_header = False

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            current_batch = batches[0]
            if pending_tail is not None and pending_tail.height > 0:
                current_batch = pl.concat([pending_tail, current_batch], how="vertical_relaxed")

            ready_batch, pending_tail = split_tail_user(current_batch)
            if ready_batch.height > 0:
                transformed = process_batch(ready_batch, rng)
                transformed.write_csv(outfile, include_header=not wrote_header)
                wrote_header = True

            chunk_bar.update(1)

        if pending_tail is not None and pending_tail.height > 0:
            final_transformed = process_batch(pending_tail, rng)
            final_transformed.write_csv(outfile, include_header=not wrote_header)

        chunk_bar.close()


def main() -> None:
    process_csv(INPUT_PATH, OUTPUT_PATH, CHUNK_SIZE)


if __name__ == "__main__":
    main()
