import pandas as pd
import re
import random
from datetime import timedelta

INPUT_FILE = "extracted_row.csv"
OUTPUT_FILE = "extracted_row_with_order.csv"

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

TOURIST_ATTRACTION_LABELS = {
    "旅游参观",
    "旅游中途休息",
}


def get_scene_type(label):
    if label in TRAIN_LABELS:
        return "train"
    if label in FLIGHT_LABELS:
        return "flight"
    if label in HOTEL_LABELS:
        return "hotel"
    if label in TOURIST_ATTRACTION_LABELS:
        return "tourist"
    return None


def extract_current_city(text):
    if pd.isna(text):
        return None
    m = re.search(r'城市：([^，,]+)', str(text))
    return m.group(1).strip() if m else None


def extract_from_city(text):
    if pd.isna(text):
        return None
    m = re.search(r'前\d+分钟从([^，,]+?)(跨城市|市内)', str(text))
    return m.group(1).strip() if m else None


def add_random_offset(dt, max_minutes=5):
    if pd.isna(dt):
        return dt
    offset = random.randint(-max_minutes, max_minutes)
    return dt + timedelta(minutes=offset)


def format_full_order(order_type, start_dt, end_dt, departure_city, arrival_city):
    start_date = start_dt.strftime("%Y/%m/%d") if pd.notna(start_dt) else ""
    start_time = start_dt.strftime("%H:%M:%S") if pd.notna(start_dt) else ""
    end_date = end_dt.strftime("%Y/%m/%d") if pd.notna(end_dt) else ""
    end_time = end_dt.strftime("%H:%M:%S") if pd.notna(end_dt) else ""

    departure_city = departure_city or ""
    arrival_city = arrival_city or ""

    return (
        f"orderType={order_type};"
        f"startDate={start_date};"
        f"startTime={start_time};"
        f"endDate={end_date};"
        f"endTime={end_time};"
        f"departureCity={departure_city};"
        f"arrivalCity={arrival_city}"
    )


def format_simple_order(order_type, start_dt):
    start_date = start_dt.strftime("%Y/%m/%d") if pd.notna(start_dt) else ""
    return f"orderType={order_type};startDate={start_date}"


def build_train_order(seg):
    seg = seg.sort_values("time_dt").copy()

    waiting = seg[seg["scene_label"] == "高铁站候车"]
    depart_station = seg[seg["scene_label"] == "抵达始发高铁站"]
    onboard = seg[seg["scene_label"] == "高铁行程途中"]
    arrive_station = seg[seg["scene_label"] == "抵达终点高铁站"]
    leave_station = seg[seg["scene_label"] == "离开终点高铁站"]

    left_start_row = None
    right_start_row = None

    if not waiting.empty:
        left_start_row = waiting.iloc[-1]
    elif not depart_station.empty:
        left_start_row = depart_station.iloc[-1]

    if not onboard.empty:
        right_start_row = onboard.iloc[0]
    elif not arrive_station.empty:
        right_start_row = arrive_station.iloc[0]

    if left_start_row is not None and right_start_row is not None:
        start_dt = left_start_row["time_dt"] + (right_start_row["time_dt"] - left_start_row["time_dt"]) / 2
    elif left_start_row is not None:
        start_dt = left_start_row["time_dt"]
    elif right_start_row is not None:
        start_dt = right_start_row["time_dt"]
    else:
        start_dt = seg.iloc[0]["time_dt"]

    if not arrive_station.empty:
        end_dt = arrive_station.iloc[0]["time_dt"] - timedelta(minutes=1)
    elif not leave_station.empty:
        end_dt = leave_station.iloc[0]["time_dt"] - timedelta(minutes=1)
    else:
        end_dt = seg.iloc[-1]["time_dt"]

    start_dt = add_random_offset(start_dt, max_minutes=5)
    end_dt = add_random_offset(end_dt, max_minutes=5)

    if pd.notna(start_dt) and pd.notna(end_dt) and start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    departure_city = None
    start_side = pd.concat([depart_station, waiting]).sort_values("time_dt")
    for _, row in start_side.iterrows():
        city = row["current_city"]
        if city:
            departure_city = city
            break

    if not departure_city and not onboard.empty:
        departure_city = onboard.iloc[0]["from_city"] or onboard.iloc[0]["current_city"]

    if not departure_city:
        departure_city = seg.iloc[0]["current_city"]

    arrival_city = None
    end_side = pd.concat([arrive_station, leave_station]).sort_values("time_dt")
    for _, row in end_side.iterrows():
        city = row["current_city"]
        if city:
            arrival_city = city
            break

    if not arrival_city:
        arrival_city = seg.iloc[-1]["current_city"]

    return format_full_order("火车订单", start_dt, end_dt, departure_city, arrival_city)

def build_flight_order(seg):
    seg = seg.sort_values("time_dt").copy()

    depart_airport = seg[seg["scene_label"] == "抵达始发机场"]
    airport_activity = seg[seg["scene_label"] == "机场内活动"]
    onboard = seg[seg["scene_label"] == "飞机行程途中"]
    arrive_airport = seg[seg["scene_label"] == "抵达终点机场"]
    leave_airport = seg[seg["scene_label"] == "离开终点机场"]

    if not airport_activity.empty:
        start_dt = airport_activity.iloc[-1]["time_dt"] + timedelta(minutes=1)
    elif not depart_airport.empty:
        start_dt = depart_airport.iloc[-1]["time_dt"] + timedelta(minutes=1)
    elif not onboard.empty:
        start_dt = onboard.iloc[0]["time_dt"] - timedelta(minutes=1)
    else:
        start_dt = seg.iloc[0]["time_dt"]

    if not arrive_airport.empty:
        end_dt = arrive_airport.iloc[0]["time_dt"] - timedelta(minutes=1)
    elif not leave_airport.empty:
        end_dt = leave_airport.iloc[0]["time_dt"] - timedelta(minutes=1)
    else:
        end_dt = seg.iloc[-1]["time_dt"]

    start_dt = add_random_offset(start_dt, max_minutes=5)
    end_dt = add_random_offset(end_dt, max_minutes=5)

    if pd.notna(start_dt) and pd.notna(end_dt) and start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    departure_city = None
    start_side = pd.concat([depart_airport, airport_activity]).sort_values("time_dt")
    for _, row in start_side.iterrows():
        city = row["current_city"]
        if city:
            departure_city = city
            break

    if not departure_city and not onboard.empty:
        departure_city = onboard.iloc[0]["from_city"] or onboard.iloc[0]["current_city"]

    if not departure_city:
        departure_city = seg.iloc[0]["current_city"]

    arrival_city = None
    end_side = pd.concat([arrive_airport, leave_airport]).sort_values("time_dt")
    for _, row in end_side.iterrows():
        city = row["current_city"]
        if city:
            arrival_city = city
            break

    if not arrival_city:
        arrival_city = seg.iloc[-1]["current_city"]

    return format_full_order("飞机订单", start_dt, end_dt, departure_city, arrival_city)


def build_hotel_order(seg):
    seg = seg.sort_values("time_dt").copy()
    start_dt = seg.iloc[0]["time_dt"]
    return format_simple_order("酒店订单", start_dt), start_dt.strftime("%Y/%m/%d")


def build_tourist_order(day_seg):
    day_seg = day_seg.sort_values("time_dt").copy()
    start_dt = day_seg.iloc[0]["time_dt"]
    return format_simple_order("旅游景点订单", start_dt), start_dt.strftime("%Y/%m/%d")


def split_segments(group, max_gap_hours=12):
    group = group.sort_values("time_dt").copy()
    segments = []

    current_idx = []
    prev_time = None
    prev_scene_type = None

    for idx, row in group.iterrows():
        scene_type = row["scene_type"]
        curr_time = row["time_dt"]

        if not current_idx:
            current_idx = [idx]
        else:
            gap_too_large = (curr_time - prev_time).total_seconds() > max_gap_hours * 3600
            scene_changed = scene_type != prev_scene_type

            if gap_too_large or scene_changed:
                segments.append(group.loc[current_idx])
                current_idx = [idx]
            else:
                current_idx.append(idx)

        prev_time = curr_time
        prev_scene_type = scene_type

    if current_idx:
        segments.append(group.loc[current_idx])

    return segments


def split_tourist_segment_by_day(seg):
    """
    将一个旅游连续片段按日期拆分。
    返回: [(date_str, day_seg), ...]
    """
    seg = seg.sort_values("time_dt").copy()
    seg["date_str"] = seg["time_dt"].dt.strftime("%Y/%m/%d")

    result = []
    for date_str, day_seg in seg.groupby("date_str", sort=True):
        result.append((date_str, day_seg.sort_values("time_dt")))
    return result


def main():
    # random.seed(42)

    df = pd.read_csv(INPUT_FILE)

    required_cols = {"time", "udid", "text", "scene_label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV必须包含列: {required_cols}")

    df["time_dt"] = pd.to_datetime(df["time"], format="%Y/%m/%d %H:%M", errors="coerce")
    df["scene_type"] = df["scene_label"].apply(get_scene_type)
    df["current_city"] = df["text"].apply(extract_current_city)
    df["from_city"] = df["text"].apply(extract_from_city)
    df["order"] = ""

    valid_df = df[df["scene_type"].notna()].copy()

    generated_hotel_days = set()    # {(udid, 'YYYY/MM/DD')}
    generated_tourist_days = set()  # {(udid, 'YYYY/MM/DD')}

    for udid, group in valid_df.groupby("udid", sort=False):
        group = group.sort_values("time_dt")
        segments = split_segments(group, max_gap_hours=12)

        for seg in segments:
            if seg.empty:
                continue

            scene_type = seg.iloc[0]["scene_type"]

            if scene_type == "train":
                first_idx = seg.index[0]
                order_str = build_train_order(seg)
                df.at[first_idx, "order"] = order_str

            elif scene_type == "flight":
                first_idx = seg.index[0]
                order_str = build_flight_order(seg)
                df.at[first_idx, "order"] = order_str

            elif scene_type == "hotel":
                first_idx = seg.index[0]
                order_str, start_date = build_hotel_order(seg)
                key = (udid, start_date)
                if key not in generated_hotel_days:
                    df.at[first_idx, "order"] = order_str
                    generated_hotel_days.add(key)

            elif scene_type == "tourist":
                # 旅游连续片段跨天时，按天拆分生成多个订单
                day_segments = split_tourist_segment_by_day(seg)

                for date_str, day_seg in day_segments:
                    key = (udid, date_str)
                    if key in generated_tourist_days:
                        continue

                    first_idx = day_seg.index[0]
                    order_str, _ = build_tourist_order(day_seg)
                    df.at[first_idx, "order"] = order_str
                    generated_tourist_days.add(key)

    df = df.drop(columns=["time_dt", "scene_type", "current_city", "from_city"], errors="ignore")
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"处理完成，输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
