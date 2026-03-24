import math
import re
from pathlib import Path

import polars as pl
from tqdm import tqdm


INPUT_PATH = Path("sample_in.csv")
OUTPUT_PATH = Path("out.csv")
CHUNK_SIZE = 100_000
TIME_FORMAT = "%Y/%m/%d %H:%M"
HOUR_GAP_COLUMN = "hours_since_prev"

APP_KEYWORDS = {
    "app_travel": ["同程旅行", "携程旅行", "去哪儿旅行", "华住会", "飞猪旅行", "美团"],
    "app_takeaway": ["UU跑腿", "美团众包", "蜂鸟众包", "达达骑士版", "闪送员", "美团骑手"],
    "app_goods": [
        "运满满司机",
        "货拉拉司机版",
        "陆运帮司机",
        "满易运司机",
        "成丰货运司机端",
        "运盟司机端",
        "中交智运司机版",
        "货车帮司机",
        "丰湃司机",
        "新赤湾司机",
        "美达司机端",
        "润药司机端",
        "梦驼铃司机帮",
        "智通三千司机APP",
        "狮桥司机",
        "顺丰同城骑士",
        "申行者",
        "滴滴送货司机",
        "货拉拉专送司机",
    ],
    "app_driver": ["滴滴车主", "T3车主", "嘀嗒出租司机", "曹操司机", "优e出租司机", "花小猪司机端", "哈啰车主"],
    "app_work": ["企业微信", "腾讯会议", "飞书", "Welink", "钉钉"],
    "app_map": ["百度地图", "高德地图"],
    "app_ticket": [
        "铁路12306",
        "携程旅行",
        "航旅纵横",
        "飞猪旅行",
        "同程旅行",
        "华住会",
        "美团",
        "航班管家",
        "智行火车票",
        "去哪儿旅行",
        "东方航空",
        "南方航空",
        "四川航空",
        "飞常准业内版",
        "吉祥航空",
        "春秋航空",
        "海南航空",
        "深圳航空",
        "小猪民宿",
        "途家民宿",
        "途牛旅游",
    ],
    "app_ride_hailing": ["网约车"],
}

MOVE_FEATURES = {
    "move_any": "移动",
    "move_fast": "高速移动",
    "move_cross_city": "跨城市",
}

TIME_FEATURES = {
    "time_early_morning": "凌晨",
    "time_morning": "上午",
    "time_afternoon": "下午",
    "time_night": "晚上",
}

REQUIRED_COLUMNS = ["time", "udid", "text"]
CITY_PATTERN = r"城市：([^，]*)"
POI_PATTERN = r"POI：([^，]*)"


def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as infile:
        row_count = sum(1 for _ in infile)

    return max(row_count - 1, 0)


def build_contains_expr(source: pl.Expr, keyword: str) -> pl.Expr:
    return source.str.contains(re.escape(keyword), literal=False)


def build_any_keyword_expr(source: pl.Expr, keywords: list[str]) -> pl.Expr:
    pattern = "|".join(re.escape(keyword) for keyword in keywords)
    return source.str.contains(pattern, literal=False)


def transform_batch(batch: pl.DataFrame) -> pl.DataFrame:
    text_expr = pl.col("text").fill_null("")
    city_expr = text_expr.str.extract(CITY_PATTERN, group_index=1).fill_null("")
    poi_expr = text_expr.str.extract(POI_PATTERN, group_index=1).fill_null("")

    feature_exprs = []

    for column_name, keywords in APP_KEYWORDS.items():
        feature_exprs.append(build_any_keyword_expr(text_expr, keywords).alias(column_name))

    for column_name, keyword in MOVE_FEATURES.items():
        feature_exprs.append(build_contains_expr(text_expr, keyword).alias(column_name))

    for column_name, keyword in TIME_FEATURES.items():
        feature_exprs.append(build_contains_expr(text_expr, keyword).alias(column_name))

    return batch.select(
        [
            pl.col("time"),
            pl.col("udid"),
            pl.col("text"),
            city_expr.alias("city"),
            poi_expr.alias("poi"),
            *feature_exprs,
        ]
    )


def add_hour_gap(batch: pl.DataFrame) -> pl.DataFrame:
    parsed_time_column = "_parsed_time"
    diff_hours_column = "_diff_hours"

    batch_with_time = batch.with_columns(
        pl.col("time").str.strptime(pl.Datetime, format=TIME_FORMAT, strict=True).alias(parsed_time_column)
    )

    diff_hours_expr = (
        (pl.col(parsed_time_column).diff().over("udid").dt.total_seconds() // 3600)
        .fill_null(0)
        .cast(pl.Int64)
    )

    batch_with_gap = batch_with_time.with_columns(diff_hours_expr.alias(diff_hours_column)).with_columns(
        pl.when(pl.col(diff_hours_column) < 0)
        .then(pl.lit(0))
        .otherwise(pl.col(diff_hours_column))
        .alias(HOUR_GAP_COLUMN)
    )

    output_columns = [
        "time",
        "udid",
        "text",
        "city",
        "poi",
        HOUR_GAP_COLUMN,
        *APP_KEYWORDS.keys(),
        *MOVE_FEATURES.keys(),
        *TIME_FEATURES.keys(),
    ]
    return batch_with_gap.select(output_columns)


def validate_columns(csv_path: Path) -> None:
    header = pl.read_csv(csv_path, n_rows=0)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")


def split_tail_user(batch: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if batch.height == 0:
        return batch, None

    if batch.height == 1:
        return batch.slice(0, 0), batch

    last_udid = batch.item(batch.height - 1, "udid")
    suffix_start = batch.height - 1
    while suffix_start > 0 and batch.item(suffix_start - 1, "udid") == last_udid:
        suffix_start -= 1

    return batch.slice(0, suffix_start), batch.slice(suffix_start)


def process_ready_batch(batch: pl.DataFrame) -> pl.DataFrame:
    if batch.height == 0:
        return batch

    transformed = transform_batch(batch)
    return add_hour_gap(transformed)


def process_csv(input_path: Path, output_path: Path, chunk_size: int) -> None:
    validate_columns(input_path)

    total_rows = count_data_rows(input_path)
    total_chunks = math.ceil(total_rows / chunk_size) if total_rows else 0
    output_columns = [
        *REQUIRED_COLUMNS,
        "city",
        "poi",
        HOUR_GAP_COLUMN,
        *APP_KEYWORDS.keys(),
        *MOVE_FEATURES.keys(),
        *TIME_FEATURES.keys(),
    ]

    if total_rows == 0:
        text_columns = set(REQUIRED_COLUMNS + ["city", "poi"])
        empty_schema = {
            column: pl.Utf8 if column in text_columns else (pl.Int64 if column == HOUR_GAP_COLUMN else pl.Boolean)
            for column in output_columns
        }
        pl.DataFrame(schema=empty_schema).select(output_columns).write_csv(output_path)
        return

    reader = pl.read_csv_batched(
        input_path,
        batch_size=chunk_size,
        columns=REQUIRED_COLUMNS,
        encoding="utf8",
        schema_overrides={"time": pl.Utf8, "udid": pl.Utf8, "text": pl.Utf8},
    )

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        chunk_bar = tqdm(total=total_chunks, desc="Processing chunks", unit="chunk")
        wrote_header = False
        pending_tail: pl.DataFrame | None = None

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            current_batch = batches[0]
            if pending_tail is not None and pending_tail.height > 0:
                current_batch = pl.concat([pending_tail, current_batch], how="vertical_relaxed")

            ready_batch, pending_tail = split_tail_user(current_batch)
            if ready_batch.height > 0:
                transformed = process_ready_batch(ready_batch)
                transformed.write_csv(outfile, include_header=not wrote_header)
                wrote_header = True

            chunk_bar.update(1)

        if pending_tail is not None and pending_tail.height > 0:
            final_transformed = process_ready_batch(pending_tail)
            final_transformed.write_csv(outfile, include_header=not wrote_header)

        chunk_bar.close()


def main() -> None:
    process_csv(INPUT_PATH, OUTPUT_PATH, CHUNK_SIZE)


if __name__ == "__main__":
    main()
