import math
from datetime import datetime, time as dt_time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import polars as pl
from tqdm import tqdm


INPUT_PATH = Path("a.csv")
OUTPUT_PATH = Path("a_label.csv")
CHUNK_SIZE = 100_000
TIME_FORMAT = "%Y/%m/%d %H:%M"
SCENE_LABEL_COLUMN = "scene_label"
REQUIRED_COLUMNS = [
    "time",
    "udid",
    "text",
    "context",
    "history_usage",
    "service_click",
    "city",
    "poi",
    "app_takeaway",
    "app_goods",
    "app_map",
    "app_travel",
    "app_ticket",
    "app_driver",
    "app_ride_hailing",
    "app_work",
    "move_any",
    "move_fast",
    "move_cross_city",
    "time_early_morning",
    "time_morning",
    "time_night",
]
CORE_OUTPUT_COLUMNS = [
    "time",
    "udid",
    "text",
    SCENE_LABEL_COLUMN,
    "city",
    "context",
    "history_usage",
    "service_click",
]
SAVE_ALL_COLUMNS = False
TRAIN_START_LABEL = "抵达始发高铁站"
TRAIN_WAITING_LABEL = "高铁站候车"
TRAIN_TRAVEL_LABEL = "高铁行程途中"
TRAIN_ARRIVAL_LABEL = "抵达终点高铁站"
TRAIN_EXIT_LABEL = "离开终点高铁站"
AIR_START_LABEL = "抵达始发机场"
AIR_ACTIVITY_LABEL = "机场内活动"
AIR_TRAVEL_LABEL = "飞机行程途中"
AIR_ARRIVAL_LABEL = "抵达终点机场"
AIR_EXIT_LABEL = "离开终点机场"
SELF_DRIVE_LABEL = "自驾途中"
SELF_DRIVE_SERVICE_LABEL = "服务区休息"
SELF_DRIVE_FUEL_LABEL = "驾驶途中加油、充电"
SELF_DRIVE_PARKING_LABEL = "去停车场停车/取车"
SELF_DRIVE_ARRIVAL_LABEL = "驾车抵达终点"
SELF_DRIVE_POI_KEYWORDS = ["加油站", "电动车充电站", "服务区", "停车场"]
TOURISM_VISIT_LABEL = "旅游参观"
TOURISM_REST_LABEL = "旅游中途休息"
TOURISM_DINING_LABEL = "旅游中用餐"
HOTEL_CHECKIN_LABEL = "酒店办理入住"
HOTEL_REST_LABEL = "旅游住宿休息"
VENUE_VISIT_LABEL = "文化场馆参观"
SHOPPING_LABEL = "旅游中逛街"
CULTURAL_POI_KEYWORDS = ["博物馆", "戏剧", "宗教场所", "图书馆"]
SHOPPING_POI_KEYWORDS = ["购物中心", "店铺", "市场", "购物", "便利店", "娱乐/夜生活"]
OUTDOOR_SPORTS_LABEL = "户外运动"
FAMILY_FUN_LABEL = "亲子游玩"
OUTDOOR_SPORTS_POI_KEYWORDS = ["公园及休憩用地", "体育场", "体育中心", "水上运动中心", "露营地", "高尔夫球场", "游泳馆"]
FAMILY_FUN_POI_KEYWORDS = ["游乐园", "动植物园", "公园及休憩用地"]
RIDE_HAIL_WAIT_LABEL = "等待网约车"
RIDE_HAIL_TRAVEL_LABEL = "乘坐网约车行程中"
RIDE_HAIL_ARRIVAL_LABEL = "网约车到达终点"
WORKER_ACTIVE_LABEL = "快递员/外卖员/网约车司机工作中"
WORKER_REST_LABEL = "快递员/外卖员/网约车司机中途休息"
SUBWAY_START_LABEL = "抵达始发地铁站"
SUBWAY_TRAVEL_LABEL = "乘坐地铁中"
SUBWAY_END_LABEL = "抵达终点地铁站"
COMMUTE_TO_WORK_LABEL = "上班通勤"
COMMUTE_HOME_LABEL = "下班通勤"
TRIP_PLANNING_LABEL = "行程规划"
TRAVEL_IRRELEVANT_LABEL = "出行无关场景"
TRAVEL_IRRELEVANT_POI_KEYWORDS = [
    "高铁站",
    "机场",
    "加油站",
    "电动车充电站",
    "服务区",
    "停车场",
    "旅游景点",
    "酒店旅馆",
    "地铁站",
]


@dataclass(frozen=True)
class SceneRule:
    name: str
    priority: int
    description: str
    processor: Callable[[pl.DataFrame], pl.DataFrame]


def truthy(value: object) -> bool:
    return value is True


def is_meal_time(value: object) -> bool:
    if not isinstance(value, str):
        return False

    parsed_time = datetime.strptime(value, TIME_FORMAT).time()
    return dt_time(11, 0) <= parsed_time <= dt_time(14, 0) or dt_time(17, 0) <= parsed_time <= dt_time(21, 0)


def parse_datetime_value(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None

    return datetime.strptime(value, TIME_FORMAT)


def contains_keyword(value: object, keyword: str) -> bool:
    return isinstance(value, str) and keyword in value


def contains_any_keyword(value: object, keywords: list[str]) -> bool:
    return isinstance(value, str) and any(keyword in value for keyword in keywords)


def is_train_related(row: dict[str, object]) -> bool:
    return (
        truthy(row.get("move_fast"))
        or truthy(row.get("move_cross_city"))
        or truthy(row.get("app_ticket"))
        or contains_keyword(row.get("poi"), "高铁站")
    )


def is_airport_related(row: dict[str, object]) -> bool:
    return (
        truthy(row.get("move_fast"))
        or truthy(row.get("move_cross_city"))
        or truthy(row.get("app_ticket"))
        or contains_keyword(row.get("poi"), "机场")
    )


def is_self_drive_related(row: dict[str, object]) -> bool:
    return (
        truthy(row.get("move_fast"))
        or truthy(row.get("move_cross_city"))
        or contains_any_keyword(row.get("poi"), SELF_DRIVE_POI_KEYWORDS)
    )


def has_gap_exceeding(rows: list[dict[str, object]], threshold: int) -> bool:
    for row in rows:
        hours_since_prev = row.get("hours_since_prev")
        if isinstance(hours_since_prev, int) and hours_since_prev > threshold:
            return True

    return False


def row_gap_exceeds(row: dict[str, object], threshold: int) -> bool:
    hours_since_prev = row.get("hours_since_prev")
    return isinstance(hours_since_prev, int) and hours_since_prev > threshold


def apply_travel_scene(
    user_df: pl.DataFrame,
    *,
    start_keyword: str,
    exclude_keyword: str,
    related_checker: Callable[[dict[str, object]], bool],
    start_label: str,
    middle_before_b_label: str,
    travel_label: str,
    arrival_label: str,
    exit_label: str,
    stop_hours_threshold: int | None = None,
    max_range_length: int | None = None,
    discard_on_gap_hours: int | None = None,
) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        if labels[index] != "" or not contains_keyword(rows[index].get("poi"), start_keyword):
            index += 1
            continue

        start_index = index
        last_related_index = start_index
        has_excluded_poi = False
        has_cross_city = False
        unrelated_streak = 0
        scan_index = start_index

        while scan_index < row_count:
            if scan_index > start_index and labels[scan_index] != "":
                break

            row = rows[scan_index]
            hours_since_prev = row.get("hours_since_prev")
            if (
                stop_hours_threshold is not None
                and scan_index > start_index
                and isinstance(hours_since_prev, int)
                and hours_since_prev > stop_hours_threshold
            ):
                break
            if contains_keyword(row.get("poi"), exclude_keyword):
                has_excluded_poi = True
            if truthy(row.get("move_cross_city")):
                has_cross_city = True

            if related_checker(row):
                last_related_index = scan_index
                unrelated_streak = 0
            else:
                unrelated_streak += 1
                if unrelated_streak >= 3:
                    break

            scan_index += 1

        end_index = last_related_index

        range_rows = rows[start_index : end_index + 1]
        range_length = end_index - start_index + 1

        if has_excluded_poi or not has_cross_city:
            index = end_index + 1
            continue
        if max_range_length is not None and range_length > max_range_length:
            index = end_index + 1
            continue
        if discard_on_gap_hours is not None and has_gap_exceeding(range_rows, discard_on_gap_hours):
            index = end_index + 1
            continue

        b_index = next(
            (
                candidate_index
                for candidate_index in range(start_index + 1, end_index + 1)
                if truthy(rows[candidate_index].get("move_fast"))
            ),
            None,
        )
        final_city = rows[end_index].get("city") or ""
        c_index = next(
            (
                candidate_index
                for candidate_index in range(start_index, end_index + 1)
                if rows[candidate_index].get("city") == final_city
            ),
            None,
        )

        if b_index is None or not isinstance(final_city, str) or final_city == "":
            index = end_index + 1
            continue

        if c_index is None or c_index <= b_index:
            c_index = next(
                (
                    candidate_index
                    for candidate_index in range(b_index + 1, end_index + 1)
                    if rows[candidate_index].get("city") == final_city
                ),
                None,
            )

        if c_index is None:
            index = end_index + 1
            continue

        labels[start_index] = start_label

        for label_index in range(start_index + 1, b_index):
            if labels[label_index] == "":
                labels[label_index] = middle_before_b_label

        if labels[b_index] == "":
            labels[b_index] = travel_label

        for label_index in range(b_index + 1, c_index):
            if labels[label_index] == "":
                labels[label_index] = travel_label

        if labels[c_index] == "":
            labels[c_index] = arrival_label

        for label_index in range(c_index + 1, end_index + 1):
            if labels[label_index] == "":
                labels[label_index] = exit_label

        index = end_index + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_high_speed_rail_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_travel_scene(
        user_df,
        start_keyword="高铁站",
        exclude_keyword="机场",
        related_checker=is_train_related,
        start_label=TRAIN_START_LABEL,
        middle_before_b_label=TRAIN_WAITING_LABEL,
        travel_label=TRAIN_TRAVEL_LABEL,
        arrival_label=TRAIN_ARRIVAL_LABEL,
        exit_label=TRAIN_EXIT_LABEL,
        stop_hours_threshold=12,
    )


def process_airport_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_travel_scene(
        user_df,
        start_keyword="机场",
        exclude_keyword="高铁站",
        related_checker=is_airport_related,
        start_label=AIR_START_LABEL,
        middle_before_b_label=AIR_ACTIVITY_LABEL,
        travel_label=AIR_TRAVEL_LABEL,
        arrival_label=AIR_ARRIVAL_LABEL,
        exit_label=AIR_EXIT_LABEL,
        max_range_length=15,
        discard_on_gap_hours=12,
    )


def classify_self_drive_label(row: dict[str, object], is_last_row: bool) -> str:
    poi_value = row.get("poi")

    if is_last_row:
        if contains_keyword(poi_value, "停车场"):
            return SELF_DRIVE_PARKING_LABEL
        return SELF_DRIVE_ARRIVAL_LABEL

    if contains_keyword(poi_value, "服务区"):
        return SELF_DRIVE_SERVICE_LABEL
    if contains_keyword(poi_value, "加油站") or contains_keyword(poi_value, "电动车充电站"):
        return SELF_DRIVE_FUEL_LABEL
    if contains_keyword(poi_value, "停车场"):
        return SELF_DRIVE_PARKING_LABEL
    return SELF_DRIVE_LABEL


def has_nearby_fast_move(rows: list[dict[str, object]], current_index: int) -> bool:
    start_index = max(0, current_index - 3)
    end_index = min(len(rows), current_index + 4)

    for neighbor_index in range(start_index, end_index):
        if truthy(rows[neighbor_index].get("move_fast")):
            return True

    return False


def find_self_drive_boundary(
    rows: list[dict[str, object]],
    labels: list[str],
    anchor_index: int,
    step: int,
) -> int:
    boundary_index = anchor_index
    unrelated_streak = 0
    scan_index = anchor_index

    while 0 <= scan_index < len(rows):
        if scan_index != anchor_index and labels[scan_index] != "":
            break
        hours_since_prev = rows[scan_index].get("hours_since_prev")
        if scan_index != anchor_index and isinstance(hours_since_prev, int) and hours_since_prev > 3:
            break

        if is_self_drive_related(rows[scan_index]):
            boundary_index = scan_index
            unrelated_streak = 0
        else:
            unrelated_streak += 1
            if unrelated_streak >= 3:
                break

        scan_index += step

    return boundary_index


def process_self_drive_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        row = rows[index]
        is_anchor = truthy(row.get("move_fast")) and truthy(row.get("move_cross_city"))
        if labels[index] != "" or not is_anchor:
            index += 1
            continue

        start_index = find_self_drive_boundary(rows, labels, index, -1)
        end_index = find_self_drive_boundary(rows, labels, index, 1)
        scene_rows = rows[start_index : end_index + 1]

        if not any(truthy(scene_row.get("move_cross_city")) for scene_row in scene_rows):
            index = end_index + 1
            continue
        if any(
            contains_keyword(scene_row.get("poi"), "高铁站") or contains_keyword(scene_row.get("poi"), "机场")
            for scene_row in scene_rows
        ):
            index = end_index + 1
            continue

        for label_index in range(start_index, end_index + 1):
            if labels[label_index] == "":
                labels[label_index] = classify_self_drive_label(
                    rows[label_index],
                    is_last_row=label_index == end_index,
                )

        index = end_index + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def apply_point_scene(
    user_df: pl.DataFrame,
    *,
    poi_keywords: list[str],
    label_value: str,
) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if not contains_any_keyword(row.get("poi"), poi_keywords):
            continue
        if not has_nearby_fast_move(rows, index):
            continue

        labels[index] = label_value

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_service_area_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_point_scene(
        user_df,
        poi_keywords=["服务区"],
        label_value=SELF_DRIVE_SERVICE_LABEL,
    )


def process_fuel_charge_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_point_scene(
        user_df,
        poi_keywords=["加油站", "电动车充电站"],
        label_value=SELF_DRIVE_FUEL_LABEL,
    )


def process_parking_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_point_scene(
        user_df,
        poi_keywords=["停车场"],
        label_value=SELF_DRIVE_PARKING_LABEL,
    )


def process_tourism_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        row = rows[index]
        is_anchor = (
            labels[index] == ""
            and contains_keyword(row.get("poi"), "旅游景点")
            and truthy(row.get("app_travel"))
            and not truthy(row.get("time_early_morning"))
        )
        if not is_anchor:
            index += 1
            continue

        window_end = min(row_count, index + 11)
        scenic_count = 0
        for scan_index in range(index, window_end):
            scan_row = rows[scan_index]
            if scan_index > index and truthy(scan_row.get("time_early_morning")):
                break
            if contains_keyword(scan_row.get("poi"), "旅游景点"):
                scenic_count += 1

        if scenic_count < 2:
            index += 1
            continue

        last_spot_index = index
        unrelated_streak = 0
        scan_index = index
        while scan_index < row_count:
            if scan_index > index and labels[scan_index] != "":
                break

            scan_row = rows[scan_index]
            if scan_index > index and truthy(scan_row.get("time_early_morning")):
                break

            has_spot = contains_keyword(scan_row.get("poi"), "旅游景点")
            has_travel_app = truthy(scan_row.get("app_travel"))
            if has_spot or has_travel_app:
                last_spot_index = scan_index
                unrelated_streak = 0
            else:
                unrelated_streak += 1
                if unrelated_streak >= 5:
                    break

            scan_index += 1

        for label_index in range(index, last_spot_index + 1):
            if labels[label_index] != "":
                continue
            current_row = rows[label_index]
            if contains_keyword(current_row.get("poi"), "餐厅") and not truthy(current_row.get("move_any")) and is_meal_time(current_row.get("time")):
                labels[label_index] = TOURISM_DINING_LABEL
                continue

            if not contains_keyword(current_row.get("poi"), "旅游景点"):
                continue

            if truthy(current_row.get("move_any")):
                labels[label_index] = TOURISM_VISIT_LABEL
            else:
                labels[label_index] = TOURISM_REST_LABEL

        index = last_spot_index + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_hotel_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        if not contains_keyword(rows[index].get("poi"), "酒店旅馆"):
            index += 1
            continue

        start_index = index
        last_hotel_index = index
        hotel_poi_count = 0
        unrelated_streak = 0
        scan_index = index

        while scan_index < row_count:
            if contains_keyword(rows[scan_index].get("poi"), "酒店旅馆"):
                last_hotel_index = scan_index
                hotel_poi_count += 1
                unrelated_streak = 0
            else:
                unrelated_streak += 1
                if unrelated_streak >= 10:
                    break

            scan_index += 1

        range_rows = rows[start_index : last_hotel_index + 1]
        has_discard_feature = any(
            truthy(scene_row.get("move_cross_city"))
            or truthy(scene_row.get("move_fast"))
            or contains_keyword(scene_row.get("poi"), "机场")
            or contains_keyword(scene_row.get("poi"), "高铁站")
            or contains_keyword(scene_row.get("poi"), "旅游景点")
            or truthy(scene_row.get("app_map"))
            or truthy(scene_row.get("app_travel"))
            for scene_row in range_rows
        )

        if hotel_poi_count <= 1 or has_discard_feature:
            index = last_hotel_index + 1
            continue

        previous_hotel_city = next(
            (
                rows[previous_index].get("city")
                for previous_index in range(start_index - 1, -1, -1)
                if contains_keyword(rows[previous_index].get("poi"), "酒店旅馆")
            ),
            None,
        )
        first_hotel_city = rows[start_index].get("city")
        first_hotel_labeled = False
        for label_index in range(start_index, last_hotel_index + 1):
            if not contains_keyword(rows[label_index].get("poi"), "酒店旅馆"):
                continue
            if labels[label_index] != "":
                continue

            if not first_hotel_labeled:
                if isinstance(previous_hotel_city, str) and previous_hotel_city != "" and previous_hotel_city == first_hotel_city:
                    labels[label_index] = HOTEL_REST_LABEL
                else:
                    labels[label_index] = HOTEL_CHECKIN_LABEL
                first_hotel_labeled = True
            else:
                labels[label_index] = HOTEL_REST_LABEL

        index = last_hotel_index + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def has_consecutive_category_poi(
    rows: list[dict[str, object]],
    start_index: int,
    end_index: int,
    keywords: list[str],
) -> bool:
    consecutive_count = 0

    for index in range(start_index, end_index):
        if contains_any_keyword(rows[index].get("poi"), keywords):
            consecutive_count += 1
            if consecutive_count >= 2:
                return True
        else:
            consecutive_count = 0

    return False


def count_category_poi(
    rows: list[dict[str, object]],
    start_index: int,
    end_index: int,
    keywords: list[str],
) -> int:
    return sum(
        1
        for index in range(start_index, end_index)
        if contains_any_keyword(rows[index].get("poi"), keywords)
    )


def process_cultural_venue_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if truthy(row.get("time_early_morning")):
            continue
        if not contains_any_keyword(row.get("poi"), CULTURAL_POI_KEYWORDS):
            continue

        start_index = max(0, index - 5)
        end_index = min(len(rows), index + 6)
        has_consecutive = has_consecutive_category_poi(rows, start_index, end_index, CULTURAL_POI_KEYWORDS)
        poi_count = count_category_poi(rows, start_index, end_index, CULTURAL_POI_KEYWORDS)

        if has_consecutive or poi_count >= 3:
            labels[index] = VENUE_VISIT_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def apply_range_poi_scene(
    user_df: pl.DataFrame,
    *,
    poi_keywords: list[str],
    label_value: str,
    allow_app_feature: bool = False,
) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        row = rows[index]
        if labels[index] != "":
            index += 1
            continue
        if truthy(row.get("time_early_morning")):
            index += 1
            continue
        if not contains_any_keyword(row.get("poi"), poi_keywords):
            index += 1
            continue

        start_index = index
        last_related_index = index
        unrelated_streak = 0
        scan_index = index

        while scan_index < row_count:
            if labels[scan_index] != "" and scan_index != start_index:
                break
            current_row = rows[scan_index]
            has_related_poi = contains_any_keyword(current_row.get("poi"), poi_keywords)
            has_related_app = allow_app_feature and truthy(current_row.get("app_travel"))

            if has_related_poi or has_related_app:
                last_related_index = scan_index
                unrelated_streak = 0
            else:
                unrelated_streak += 1
                if unrelated_streak >= 3:
                    break

            scan_index += 1

        range_rows = rows[start_index : last_related_index + 1]
        poi_count = sum(1 for scene_row in range_rows if contains_any_keyword(scene_row.get("poi"), poi_keywords))
        has_consecutive = has_consecutive_category_poi(rows, start_index, last_related_index + 1, poi_keywords)

        range_start_time = parse_datetime_value(rows[start_index].get("time"))
        range_end_time = parse_datetime_value(rows[last_related_index].get("time"))
        exceeds_duration = (
            range_start_time is not None
            and range_end_time is not None
            and (range_end_time - range_start_time).total_seconds() > 12 * 3600
        )

        has_required_app = True
        if allow_app_feature:
            has_required_app = any(truthy(scene_row.get("app_travel")) for scene_row in range_rows)

        if not has_consecutive or poi_count < 3 or exceeds_duration or not has_required_app:
            index = last_related_index + 1
            continue

        for label_index in range(start_index, last_related_index + 1):
            if labels[label_index] != "":
                continue
            if truthy(rows[label_index].get("time_early_morning")):
                continue
            if contains_any_keyword(rows[label_index].get("poi"), poi_keywords):
                labels[label_index] = label_value

        index = last_related_index + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_outdoor_sports_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_range_poi_scene(
        user_df,
        poi_keywords=OUTDOOR_SPORTS_POI_KEYWORDS,
        label_value=OUTDOOR_SPORTS_LABEL,
    )


def process_family_fun_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_range_poi_scene(
        user_df,
        poi_keywords=FAMILY_FUN_POI_KEYWORDS,
        label_value=FAMILY_FUN_LABEL,
    )


def process_shopping_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    return apply_range_poi_scene(
        user_df,
        poi_keywords=SHOPPING_POI_KEYWORDS,
        label_value=SHOPPING_LABEL,
        allow_app_feature=True,
    )


def process_ride_hailing_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        row = rows[index]
        is_candidate = labels[index] == "" and truthy(row.get("app_ride_hailing"))
        if not is_candidate:
            index += 1
            continue

        if truthy(row.get("move_any")):
            index += 1
            continue

        next_labeled_index = next(
            (
                candidate_index
                for candidate_index in range(index + 1, row_count)
                if labels[candidate_index] != ""
            ),
            row_count,
        )
        next_gap_index = next(
            (
                candidate_index
                for candidate_index in range(index + 1, next_labeled_index)
                if row_gap_exceeds(rows[candidate_index], 3)
            ),
            next_labeled_index,
        )
        next_labeled_index = min(next_labeled_index, next_gap_index)
        window_end = min(index + 6, next_labeled_index)
        if not any(truthy(rows[candidate_index].get("move_any")) for candidate_index in range(index + 1, window_end)):
            index += 1
            continue

        first_move = next(
            (
                candidate_index
                for candidate_index in range(index, next_labeled_index)
                if truthy(rows[candidate_index].get("move_any"))
            ),
            None,
        )
        if first_move is None or first_move == index:
            index += 1
            continue

        last_move = first_move
        scan_index = first_move + 1
        while scan_index < next_labeled_index and truthy(rows[scan_index].get("move_any")):
            last_move = scan_index
            scan_index += 1

        for label_index in range(index, first_move):
            if labels[label_index] == "":
                labels[label_index] = RIDE_HAIL_WAIT_LABEL

        for label_index in range(first_move, last_move):
            if labels[label_index] == "":
                labels[label_index] = RIDE_HAIL_TRAVEL_LABEL

        if labels[last_move] == "":
            labels[last_move] = RIDE_HAIL_ARRIVAL_LABEL

        index = last_move + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def has_worker_app(row: dict[str, object]) -> bool:
    return truthy(row.get("app_takeaway")) or truthy(row.get("app_goods")) or truthy(row.get("app_driver"))


def process_worker_active_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if not has_worker_app(row):
            continue
        if not truthy(row.get("move_any")):
            continue

        labels[index] = WORKER_ACTIVE_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_worker_rest_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if truthy(row.get("move_any")):
            continue

        start_index = max(0, index - 5)
        end_index = min(len(rows), index + 6)
        if any(has_worker_app(rows[neighbor_index]) for neighbor_index in range(start_index, end_index) if neighbor_index != index):
            labels[index] = WORKER_REST_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_subway_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    row_count = len(rows)
    index = 0

    while index < row_count:
        row = rows[index]
        is_candidate = labels[index] == "" and contains_keyword(row.get("poi"), "地铁站") and not truthy(row.get("move_any"))
        if not is_candidate:
            index += 1
            continue

        last_move = None
        no_feature_streak = 0
        scan_index = index + 1

        while scan_index < row_count:
            if labels[scan_index] != "":
                break

            current_row = rows[scan_index]
            has_subway_poi = contains_keyword(current_row.get("poi"), "地铁站")
            has_move = truthy(current_row.get("move_any"))

            if has_subway_poi or has_move:
                no_feature_streak = 0
                if has_move:
                    last_move = scan_index
            else:
                no_feature_streak += 1
                if no_feature_streak >= 2:
                    break

            scan_index += 1

        if last_move is None:
            index += 1
            continue

        range_end = last_move
        b_index = next(
            (
                candidate_index
                for candidate_index in range(range_end, index, -1)
                if contains_keyword(rows[candidate_index].get("poi"), "地铁站")
            ),
            None,
        )
        subway_poi_count = sum(
            1
            for candidate_index in range(index, range_end + 1)
            if contains_keyword(rows[candidate_index].get("poi"), "地铁站")
        )
        range_start_time = parse_datetime_value(rows[index].get("time"))
        range_end_time = parse_datetime_value(rows[range_end].get("time"))
        exceeds_duration = (
            range_start_time is not None
            and range_end_time is not None
            and (range_end_time - range_start_time).total_seconds() > 3 * 3600
        )
        has_cross_city = any(
            truthy(rows[candidate_index].get("move_cross_city"))
            for candidate_index in range(index, range_end + 1)
        )
        has_move_between = any(
            truthy(rows[candidate_index].get("move_any"))
            for candidate_index in range(index + 1, (b_index or index) + 1)
        )

        if b_index is None or b_index == index or subway_poi_count < 2 or exceeds_duration or has_cross_city or not has_move_between:
            index = range_end + 1
            continue

        labels[index] = SUBWAY_START_LABEL
        for label_index in range(index + 1, b_index):
            if labels[label_index] == "":
                labels[label_index] = SUBWAY_TRAVEL_LABEL

        if labels[b_index] == "":
            labels[b_index] = SUBWAY_END_LABEL

        index = range_end + 1

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_commuting_to_work_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if not truthy(row.get("app_work")):
            continue
        if not truthy(row.get("move_any")):
            continue
        if not truthy(row.get("time_morning")):
            continue

        labels[index] = COMMUTE_TO_WORK_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_commuting_home_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue
        if not truthy(row.get("app_work")):
            continue
        if not truthy(row.get("move_any")):
            continue
        if not truthy(row.get("time_night")):
            continue

        labels[index] = COMMUTE_HOME_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_trip_planning_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]

    for index, row in enumerate(rows):
        if labels[index] != "":
            continue

        current_is_map = truthy(row.get("app_map"))
        current_is_ticket = truthy(row.get("app_ticket"))
        if not current_is_map and not current_is_ticket:
            continue

        start_index = max(0, index - 4)
        window_rows = rows[start_index : index + 1]
        has_map_in_window = any(truthy(window_row.get("app_map")) for window_row in window_rows)
        has_ticket_in_window = any(truthy(window_row.get("app_ticket")) for window_row in window_rows)

        if (current_is_map and has_ticket_in_window) or (current_is_ticket and has_map_in_window):
            labels[index] = TRIP_PLANNING_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def process_travel_irrelevant_scene(user_df: pl.DataFrame) -> pl.DataFrame:
    if user_df.height == 0:
        return user_df

    rows = user_df.to_dicts()
    initial_labels = [str(row.get(SCENE_LABEL_COLUMN) or "") for row in rows]
    labels = initial_labels.copy()

    for index, row in enumerate(rows):
        if initial_labels[index] != "":
            continue

        start_index = max(0, index - 5)
        end_index = min(len(rows), index + 6)
        window_rows = rows[start_index:end_index]
        window_labels = initial_labels[start_index:end_index]

        if any(label != "" for label in window_labels):
            continue
        if any(
            contains_any_keyword(window_row.get("poi"), TRAVEL_IRRELEVANT_POI_KEYWORDS)
            for window_row in window_rows
        ):
            continue
        if any(
            truthy(window_row.get("move_cross_city")) or truthy(window_row.get("move_fast"))
            for window_row in window_rows
        ):
            continue
        if any(
            truthy(window_row.get("app_travel"))
            or truthy(window_row.get("app_map"))
            or truthy(window_row.get("app_ticket"))
            for window_row in window_rows
        ):
            continue

        labels[index] = TRAVEL_IRRELEVANT_LABEL

    return user_df.with_columns(pl.Series(SCENE_LABEL_COLUMN, labels))


def build_scene_rules() -> list[SceneRule]:
    return [
        SceneRule(
            name="high_speed_rail",
            priority=10,
            description="高铁场景：基于高铁站POI、购票、高速移动、跨城市等特征标注高铁出行阶段。",
            processor=process_high_speed_rail_scene,
        ),
        SceneRule(
            name="airport",
            priority=20,
            description="飞机场景：基于机场POI、购票、高速移动、跨城市等特征标注飞机出行阶段。",
            processor=process_airport_scene,
        ),
        SceneRule(
            name="worker_active",
            priority=30,
            description="快递员/外卖员/网约车司机工作中：职业相关app出现且当前条存在移动。",
            processor=process_worker_active_scene,
        ),
        SceneRule(
            name="worker_rest",
            priority=40,
            description="快递员/外卖员/网约车司机中途休息：当前条无移动，且前后五条合计窗口内出现职业相关app。",
            processor=process_worker_rest_scene,
        ),
        SceneRule(
            name="subway_travel",
            priority=50,
            description="地铁出行场景：从地铁站且未移动的起点开始，结合高速移动窗口和双非移动停止信号识别地铁行程。",
            processor=process_subway_scene,
        ),
        SceneRule(
            name="tourism",
            priority=60,
            description="旅游场景：基于旅游景点POI、旅游app、移动和凌晨边界识别旅游参观与中途休息。",
            processor=process_tourism_scene,
        ),
        SceneRule(
            name="hotel_stay",
            priority=70,
            description="酒店场景：基于酒店旅馆POI和邻近旅游上下文标注办理入住与住宿休息。",
            processor=process_hotel_scene,
        ),
        SceneRule(
            name="cultural_venue_visit",
            priority=80,
            description="文化场馆参观：当前行为文化场馆POI，且前后五条满足连续或累计密度要求。",
            processor=process_cultural_venue_scene,
        ),
        SceneRule(
            name="shopping",
            priority=90,
            description="逛街：当前行为购物类POI且有移动，前后各三条邻近记录中至少再出现一次同类POI。",
            processor=process_shopping_scene,
        ),
        SceneRule(
            name="outdoor_sports",
            priority=100,
            description="户外运动：当前行为户外运动类POI，且前后五条满足连续两条或累计三次以上同类POI。",
            processor=process_outdoor_sports_scene,
        ),
        SceneRule(
            name="family_fun",
            priority=110,
            description="亲子游玩：当前行为亲子游玩类POI，且前后五条满足连续两条或累计三次以上同类POI。",
            processor=process_family_fun_scene,
        ),
        SceneRule(
            name="ride_hailing_passenger",
            priority=120,
            description="网约车乘客出行：从网约车关键词且未移动的起点开始，识别等待、乘坐和到达阶段。",
            processor=process_ride_hailing_scene,
        ),
        SceneRule(
            name="self_drive",
            priority=130,
            description="自驾场景：基于高速移动、跨城市与道路设施POI识别自驾范围，并按沿途活动优先级标注。",
            processor=process_self_drive_scene,
        ),
        SceneRule(
            name="service_area_rest",
            priority=140,
            description="服务区休息：当前行出现服务区POI，且前后六条邻近记录中至少一次高速移动。",
            processor=process_service_area_scene,
        ),
        SceneRule(
            name="fuel_charge_stop",
            priority=150,
            description="驾驶途中加油、充电：当前行出现加油站或电动车充电站POI，且前后六条邻近记录中至少一次高速移动。",
            processor=process_fuel_charge_scene,
        ),
        SceneRule(
            name="parking_pickup_dropoff",
            priority=160,
            description="去停车场停车/取车：当前行出现停车场POI，且前后六条邻近记录中至少一次高速移动。",
            processor=process_parking_scene,
        ),
        SceneRule(
            name="commuting_to_work",
            priority=170,
            description="上班通勤：工作类App出现、存在移动且时间为上午。",
            processor=process_commuting_to_work_scene,
        ),
        SceneRule(
            name="commuting_home",
            priority=180,
            description="下班通勤：工作类App出现、存在移动且时间为晚上。",
            processor=process_commuting_home_scene,
        ),
        SceneRule(
            name="trip_planning",
            priority=190,
            description="行程规划：地图类App与票务类App在最近五条记录内形成近邻组合。",
            processor=process_trip_planning_scene,
        ),
        SceneRule(
            name="travel_irrelevant",
            priority=200,
            description="出行无关场景：当前条及前后五条均未标注，且窗口内不出现交通设施POI、跨城/高速移动和出行相关App。",
            processor=process_travel_irrelevant_scene,
        )
    ]


def count_data_rows(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8", newline="") as infile:
        row_count = sum(1 for _ in infile)

    return max(row_count - 1, 0)


def validate_columns(csv_path: Path) -> list[str]:
    header = pl.read_csv(csv_path, n_rows=0)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing}")

    return header.columns


def ensure_scene_label_column(batch: pl.DataFrame) -> pl.DataFrame:
    if SCENE_LABEL_COLUMN in batch.columns:
        return batch.with_columns(pl.col(SCENE_LABEL_COLUMN).fill_null(""))

    return batch.with_columns(pl.lit("").alias(SCENE_LABEL_COLUMN))


def build_output_columns(batch_columns: list[str], save_all_columns: bool) -> list[str]:
    if not save_all_columns:
        return [column for column in CORE_OUTPUT_COLUMNS if column in batch_columns]

    ordered_columns = [column for column in batch_columns if column != SCENE_LABEL_COLUMN]
    if "text" in ordered_columns:
        text_index = ordered_columns.index("text")
        return [
            *ordered_columns[: text_index + 1],
            SCENE_LABEL_COLUMN,
            *ordered_columns[text_index + 1 :],
        ]

    return [*ordered_columns, SCENE_LABEL_COLUMN]


def process_user_batch(user_batch: pl.DataFrame, scene_rules: list[SceneRule]) -> pl.DataFrame:
    labeled_user_batch = ensure_scene_label_column(user_batch)
    for scene_rule in sorted(scene_rules, key=lambda rule: rule.priority):
        labeled_user_batch = scene_rule.processor(labeled_user_batch)
    return labeled_user_batch


def split_tail_user(batch: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    if batch.height == 0:
        return batch, None

    last_udid = batch.item(batch.height - 1, "udid")
    if batch.height == 1:
        return batch.slice(0, 0), batch

    tail_mask = pl.col("udid") == pl.lit(last_udid)
    tail_batch = batch.filter(tail_mask)
    ready_batch = batch.filter(~tail_mask)
    return ready_batch, tail_batch


def process_ready_batch(
    batch: pl.DataFrame,
    scene_rules: list[SceneRule],
    save_all_columns: bool,
) -> pl.DataFrame:
    if batch.height == 0:
        empty_columns = build_output_columns([*batch.columns, SCENE_LABEL_COLUMN], save_all_columns)
        empty_schema = {column: pl.Utf8 for column in empty_columns}
        return pl.DataFrame(schema=empty_schema).select(empty_columns)

    processed_groups = [
        process_user_batch(user_group, scene_rules)
        for user_group in batch.partition_by("udid", maintain_order=True)
    ]
    combined_batch = pl.concat(processed_groups, how="vertical")
    output_columns = build_output_columns(combined_batch.columns, save_all_columns)
    return combined_batch.select(output_columns)


def write_empty_output(
    output_path: Path,
    input_columns: list[str],
    save_all_columns: bool,
) -> None:
    output_columns = build_output_columns([*input_columns, SCENE_LABEL_COLUMN], save_all_columns)
    empty_schema = {column: pl.Utf8 for column in output_columns}
    pl.DataFrame(schema=empty_schema).select(output_columns).write_csv(output_path)


def process_csv(
    input_path: Path,
    output_path: Path,
    chunk_size: int,
    save_all_columns: bool = SAVE_ALL_COLUMNS,
) -> None:
    input_columns = validate_columns(input_path)
    total_rows = count_data_rows(input_path)
    total_chunks = math.ceil(total_rows / chunk_size) if total_rows else 0
    scene_rules = build_scene_rules()

    if total_rows == 0:
        write_empty_output(output_path, input_columns, save_all_columns)
        return

    reader = pl.read_csv_batched(
        input_path,
        batch_size=chunk_size,
        encoding="utf8",
    )

    pending_tail: pl.DataFrame | None = None

    with output_path.open("w", encoding="utf-8", newline="") as outfile:
        chunk_bar = tqdm(total=total_chunks, desc="Labeling scenes", unit="chunk")
        wrote_header = False

        while True:
            batches = reader.next_batches(1)
            if not batches:
                break

            current_batch = ensure_scene_label_column(batches[0])
            if pending_tail is not None and pending_tail.height > 0:
                current_batch = pl.concat([pending_tail, current_batch], how="vertical_relaxed")

            ready_batch, pending_tail = split_tail_user(current_batch)
            if ready_batch.height > 0:
                transformed = process_ready_batch(ready_batch, scene_rules, save_all_columns)
                transformed.write_csv(outfile, include_header=not wrote_header)
                wrote_header = True

            chunk_bar.update(1)

        final_batch = pending_tail if pending_tail is not None else pl.DataFrame(schema={column: pl.Utf8 for column in input_columns})
        final_transformed = process_ready_batch(final_batch, scene_rules, save_all_columns)
        if final_transformed.height > 0 or not wrote_header:
            final_transformed.write_csv(outfile, include_header=not wrote_header)

        chunk_bar.close()


def main() -> None:
    process_csv(INPUT_PATH, OUTPUT_PATH, CHUNK_SIZE, SAVE_ALL_COLUMNS)


if __name__ == "__main__":
    main()
