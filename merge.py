import os
import csv
import re
from datetime import datetime
import polars as pl
from collections import deque
from tqdm import tqdm


def _parse_order_time(order_text: str, field_name: str):
    """从 order 文本中解析出发/到达时间，失败返回 None。"""
    if not order_text:
        return None

    pattern = rf"{re.escape(field_name)}：(\d{{4}}/\d{{2}}/\d{{2}} \d{{2}}:\d{{2}})"
    match = re.search(pattern, order_text)
    if not match:
        return None

    try:
        return datetime.strptime(match.group(1), "%Y/%m/%d %H:%M")
    except ValueError:
        return None


def _build_time_hint(time_val: str, order_text: str, time_window_minutes: int = 120):
    """根据 time 与 order 中的出发/到达时间生成提示文案。"""
    if not time_val or not order_text:
        return ""

    try:
        current_time = datetime.strptime(time_val, "%Y/%m/%d %H:%M")
    except ValueError:
        return ""

    times = {}
    for field_name, arrive_word in (("出发时间", "出发"), ("到达时间", "抵达")):
        target_time = _parse_order_time(order_text, field_name)
        if target_time is not None:
            times[arrive_word] = target_time

    if not times:
        return ""

    show_hints = False
    if time_window_minutes == 0:
        show_hints = True
    else:
        for tgt_time in times.values():
            delta_minutes = int(abs((current_time - tgt_time).total_seconds()) // 60)
            if delta_minutes <= time_window_minutes:
                show_hints = True
                break

    if not show_hints:
        return ""

    hints = []
    for arrive_word in ("出发", "抵达"):
        if arrive_word in times:
            target_time = times[arrive_word]
            sign = "+" if target_time >= current_time else "-"
            abs_delta = int(abs((target_time - current_time).total_seconds()) // 60)
            hints.append(f"距{arrive_word}时间{sign}{abs_delta}分钟")

    return "，".join(hints)


def process_and_merge_final_streaming_polars(
    input_file: str,
    output_file: str,
    n: int = 10,
    batch_size: int = 200_000,
    time_window_minutes: int = 120,
):
    """
    使用 Polars 分块处理并流式写出，将 label 列统一为 scene_label。

    数据假设：
    1. 已按 udid, time 排好序
    2. 同一个 udid 的记录连续出现

MERGED_TEXT 规则：
[current]当前text[order]当前order[/order][/current]
+ 依次追加 [previous-1]...[/previous-1] 到 [previous-n]...[/previous-n]
（仅限同 udid 的历史记录）

    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径
        n: 历史窗口大小
        batch_size: 每批读取行数
    """
    if n < 1:
        raise ValueError("n 必须 >= 1")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"未找到输入文件: {input_file}")

    output_cols = [
        "time",
        "udid",
        "scene_label",
        "MERGED_TEXT",
        "context",
        "history_usage",
        "service_click",
    ]

    print("开始使用 Polars 分块读取并拼接（不排序，流式写出）...")

    # 跨 chunk 保留状态
    last_udid = None
    prev_texts = deque(maxlen=n)

    # 先删掉旧文件，避免追加到历史结果后面
    if os.path.exists(output_file):
        os.remove(output_file)

    # 用 csv writer 做真正的流式写出
    with open(output_file, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(output_cols)

        input_columns = pl.read_csv(input_file, n_rows=0).columns
        schema_overrides = {
            col: pl.Utf8
            for col in [
                "time",
                "udid",
                "scene_label",
                "text",
                "order",
                "context",
                "history_usage",
                "service_click",
            ]
            if col in input_columns
        }

        # 使用新 API 分批读取，兼容缺失的旧字段
        batches = pl.scan_csv(
            input_file,
            has_header=True,
            ignore_errors=True,
            schema_overrides=schema_overrides,
        ).collect_batches(chunk_size=batch_size, maintain_order=True)

        pbar = tqdm(desc="处理batch", unit="batch")

        for batch in batches:

            optional_cols = ["context", "history_usage", "service_click"]
            for col in optional_cols:
                if col not in batch.columns:
                    batch = batch.with_columns(pl.lit("").alias(col))

            # 空值统一处理
            batch = batch.with_columns([
                pl.col("time").fill_null(""),
                pl.col("udid").fill_null(""),
                pl.col("scene_label").fill_null(""),
                pl.col("text").fill_null(""),
                pl.col("order").fill_null(""),
                pl.col("context").fill_null(""),
                pl.col("history_usage").fill_null(""),
                pl.col("service_click").fill_null(""),
            ])

            # 转成行迭代需要的形式
            rows = batch.iter_rows(named=True)

            out_rows = []

            for row in rows:
                time_val = row["time"]
                udid = row["udid"]
                scene_label = row["scene_label"]
                text = row["text"]
                order = row["order"]
                context = row["context"]
                history_usage = row["history_usage"]
                service_click = row["service_click"]

                # udid 切换时清空历史窗口
                if udid != last_udid:
                    prev_texts.clear()
                    last_udid = udid

                cur_text = text or ""
                cur_order = order or ""

                time_hint = _build_time_hint(time_val, cur_order, time_window_minutes)

                merged = f"[current]{cur_text}"
                if cur_order != "":
                    merged += f"[order]{cur_order}[/order]"
                    if time_hint:
                        merged += time_hint
                merged += "[/current]"

                if prev_texts:
                    for i, ptxt in enumerate(reversed(prev_texts), start=1):
                        merged += f" [previous-{i}]{ptxt}[/previous-{i}]"

                out_rows.append([
                    time_val,
                    udid,
                    scene_label,
                    merged,
                    context,
                    history_usage,
                    service_click,
                ])

                prev_texts.append(cur_text)

            writer.writerows(out_rows)
            pbar.update(1)

        pbar.close()

    print(f"✅ 处理成功！结果已保存至: {output_file}")


# 运行配置
config = {
    "input_file": "../data/单框架戏剧替换_test1_feature_label.csv",
    "output_file": "../data/单框架戏剧替换_test1_muban_merged.csv",
    "n": 10,
    "batch_size": 200000,
    "time_window_minutes": 120,
}

if __name__ == "__main__":
    if os.path.exists(config["input_file"]):
        process_and_merge_final_streaming_polars(**config)
    else:
        print(f"错误：未找到输入文件 {config['input_file']}")
