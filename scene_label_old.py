import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
import os
import numpy as np
from multiprocessing import Pool, cpu_count

# ============================================================
# 特征抽取（升级版：支持多POI切割、App记录识别、时间段识别）
# ============================================================
def extract_features(text):
    if not isinstance(text, str):
        text = str(text)

    # 1. 提取城市
    city_match = re.search(r'城市：(\w+市|\w+自治州|\w+盟|\w+县)', text)
    city = city_match.group(1) if city_match else None

    # 2. 提取 POI 列表 (逻辑：找到 "POI：" 开头,直到遇到中文逗号 "," 或字符串结束)
    # 示例文本："POI：高铁站、地铁站,..."
    poi_match = re.search(r'POI：([^,]+)', text)
    poi_set = set()
    if poi_match:
        poi_content = poi_match.group(1)
        # 按顿号切割
        poi_items = [item.strip() for item in poi_content.split('、')]
        poi_set = set(poi_items)

    # 3. 提取应用记录 & 判断App
    # 逻辑：找到 "应用记录：" 开头,直到遇到中文逗号 "," 或字符串结束
    app_match = re.search(r'应用记录：([^,]+)', text)
    app_text = app_match.group(1) if app_match else ""
    
    # 定义相关 App 的关键词库
    travel_app_keywords = ['同程旅行', '携程旅行', '去哪儿旅行', '华住会', '飞猪旅行', '美团']
    has_travel_app = any(keyword in app_text for keyword in travel_app_keywords)
    
    take_out_app_keywords = ['UU跑腿','美团众包','蜂鸟众包','达达骑士版','闪送员','美团骑手']
    has_take_out_app = any(keyword in app_text for keyword in take_out_app_keywords)
    
    goods_app_keywords =  ['运满满司机','货拉拉司机版','陆运帮司机','满易运司机','成丰货运司机端','运盟司机端','中交智运司机版','货车帮司机','丰湃司机','新赤湾司机','美达司机端','润药司机端','梦驼铃司机帮','智通三千司机APP','狮桥司机','顺丰同城骑士','申行者','滴滴快递/货物配送司机','货拉拉专送司机']
    has_goods_app = any(keyword in app_text for keyword in goods_app_keywords)

    driver_app_keywords = ['滴滴车主','T3车主','嘀嗒出租司机','曹操司机','优e出租司机','花小猪司机端','哈啰车主']
    has_driver_app = any(keyword in app_text for keyword in driver_app_keywords)

    work_app_keywords = ['企业微信','腾讯会议','飞书','Welink','钉钉']
    has_work_app = any(keyword in app_text for keyword in work_app_keywords)

    map_app_keywords = ['百度地图', '高德地图']
    has_map_app = any(keyword in app_text for keyword in map_app_keywords)

    ticket_app_keywords = ['铁路12306', '携程旅行', '航旅纵横', '飞猪旅行', '同程旅行', '华住会', '美团', '航班管家', '智行火车票', '去哪儿旅行', '东方航空', '南方航空', '四川航空', '飞常准业内版', '吉祥航空', '春秋航空', '海南航空', '深圳航空', '小猪民宿', '途家民宿', '途牛旅游']
    has_ticket_app = any(keyword in app_text for keyword in ticket_app_keywords)

    return pd.Series({
        'city': city,
        
        # 基础移动特征
        'has_fast_move': '高速移动' in text,
        'is_cross_city': '跨城市' in text,
        'has_any_move': '移动' in text,
        'has_ride_keyword': '网约车' in text,

        # POI 特征
        'has_poi_hotel': '酒店旅馆' in poi_set,
        'has_poi_gt': '高铁站' in poi_set,
        'has_poi_air': '机场' in poi_set,
        'has_poi_metro': '地铁站' in poi_set,
        'has_poi_spot': '旅游景点' in poi_set,

        # App 特征
        'has_travel_app': has_travel_app,
        'has_take_out_app': has_take_out_app,
        'has_goods_app': has_goods_app,
        'has_driver_app': has_driver_app,
        'has_work_app': has_work_app,
        'has_map_app': has_map_app,
        'has_ticket_app': has_ticket_app,

        # 时间段特征
        'is_early_morning': '凌晨' in text,
        'is_morning': '上午' in text,
        'is_afternoon': '下午' in text,
        'is_night': '晚上' in text
    })


# ============================================================
# 场景范围 / 关键节点识别
# ============================================================
def find_scene_range(df_user, poi_type):
    poi_col = 'has_poi_gt' if poi_type == 'gt' else 'has_poi_air'
    clues = df_user[poi_col] | df_user['has_fast_move'] | df_user['is_cross_city']

    first_poi = df_user[df_user[poi_col]].index.min()
    if pd.isna(first_poi):
        return -1, -1, None

    start_index = df_user.index.min()
    scan_idx = first_poi
    while scan_idx >= df_user.index.min():
        row = df_user.loc[scan_idx]
        if not (row[poi_col] or row['has_fast_move'] or row['is_cross_city']):
            start_index = scan_idx + 1
            break
        scan_idx -= 1

    end_index = df_user.index.max()
    max_idx = df_user.index.max()
    for i in range(start_index, max_idx + 1):
        if i + 9 <= max_idx:
            if not clues.loc[i:i+9].any():
                end_index = i - 1
                break

    if end_index < start_index:
        end_index = max_idx

    return start_index, end_index, poi_col


# ============================================================
#  高铁节点识别
# ============================================================
def find_critical_nodes_gt(df_range, start_index, end_index):
    A, C, E = None, None, None
    df_slice = df_range.loc[start_index:end_index]

    A_series = df_slice[df_slice['has_poi_gt']].index
    if not A_series.empty:
        A = A_series.min()

    if A is not None:
        C_series = df_slice.loc[A+1:]['has_fast_move'] | df_slice.loc[A+1:]['is_cross_city']
        if C_series.any():
            C = C_series[C_series].index.min()

    if C is not None:
        last_city = df_slice['city'].iloc[-1]
        city_slice = df_slice.loc[C:][df_slice.loc[C:]['city'] == last_city]
        if not city_slice.empty:
            E_candidate = city_slice['has_fast_move'] | city_slice['is_cross_city']
            if E_candidate.any():
                E = E_candidate[E_candidate].index.max()
            else:
                E = city_slice.index.min()

    if A is None:
        return None, None, None

    if C is not None and C < A:
        C = None
    if E is not None and E < (C or A):
        E = None

    return A, C, E


# ============================================================
#  机场节点识别
# ============================================================
def find_critical_nodes_air(df_range, start_index, end_index):
    A, E, L = None, None, None
    df_slice = df_range.loc[start_index:end_index]

    A_series = df_slice[df_slice['has_poi_air']].index
    if not A_series.empty:
        A = A_series.min()

    if A is not None:
        L_series = df_slice[df_slice['has_poi_air']].index
        L = L_series.max() if not L_series.empty else None

        last_city = df_slice['city'].iloc[-1]
        E_candidate = df_slice[df_slice['city'] == last_city].index.min()
        if not pd.isna(E_candidate):
            E_series = df_slice.loc[E_candidate:][df_slice.loc[E_candidate:]['is_cross_city']].index
            if not E_series.empty:
                E = E_series.min()

    if A is None:
        return None, None, None

    if E is not None and E < A:
        E = None
    if L is not None and L < A:
        L = None

    return A, E, L


# ============================================================
# 高铁 标注
# ============================================================
def label_gt_scene(df_local, start_index_local, end_index_local):
    if df_local.loc[start_index_local:end_index_local, 'has_poi_air'].any():
        return df_local

    A_local, C_local, E_local = find_critical_nodes_gt(df_local, start_index_local, end_index_local)
    if A_local is None:
        return df_local

    if df_local.at[A_local, 'scene_label'] == '':
        df_local.at[A_local, 'scene_label'] = '抵达始发高铁站'

    if C_local is not None and A_local + 1 <= C_local - 1:
        for i in range(A_local + 1, C_local):
            if df_local.at[i, 'scene_label'] == '':
                df_local.at[i, 'scene_label'] = '在高铁站候车'

    start_D = C_local if C_local is not None else A_local + 1
    end_D = E_local - 1 if E_local is not None else end_index_local
    if start_D <= end_D:
        for i in range(start_D, end_D + 1):
            if df_local.at[i, 'scene_label'] == '':
                df_local.at[i, 'scene_label'] = '高铁行程途中'

    if E_local is not None and E_local <= end_index_local:
        if df_local.at[E_local, 'scene_label'] == '':
            df_local.at[E_local, 'scene_label'] = '抵达终点高铁站'
        if E_local + 1 <= end_index_local:
            for i in range(E_local + 1, end_index_local + 1):
                if df_local.at[i, 'scene_label'] == '':
                    df_local.at[i, 'scene_label'] = '离开终点高铁站'

    return df_local


# ============================================================
# 机场 标注
# ============================================================
def label_air_scene(df_local, start_index_local, end_index_local):
    if df_local.loc[start_index_local:end_index_local, 'has_poi_gt'].any():
        return df_local

    A_local, E_local, L_local = find_critical_nodes_air(df_local, start_index_local, end_index_local)
    if A_local is None:
        return df_local

    if df_local.at[A_local, 'scene_label'] == '':
        df_local.at[A_local, 'scene_label'] = '抵达始发机场'

    end_activity = E_local - 1 if E_local is not None else end_index_local
    if A_local + 1 <= end_activity:
        for i in range(A_local + 1, end_activity + 1):
            if df_local.at[i, 'scene_label'] == '':
                df_local.at[i, 'scene_label'] = '机场内活动'

    if E_local is not None and E_local <= end_index_local:
        if df_local.at[E_local, 'scene_label'] == '':
            df_local.at[E_local, 'scene_label'] = '抵达终点机场'
        if L_local is not None and E_local + 1 <= L_local:
            for i in range(E_local + 1, L_local + 1):
                if df_local.at[i, 'scene_label'] == '':
                    df_local.at[i, 'scene_label'] = '离开终点机场'

    return df_local


# ============================================================
# 旅游参观 标注
# ============================================================
def label_tour_visiting(df_local):
    df = df_local
    n = len(df)
    i = 0
    
    while i < n:
        # 1. 寻找锚点：必须包含旅游景点POI,且未被标注,且当前不是凌晨
        if df.at[i, 'scene_label'] != '' or not df.at[i, 'has_poi_spot'] or df.at[i, 'is_early_morning']:
            i += 1
            continue

        # 2. 验证密度
        # 规则：往下扫描10条 或 扫描到凌晨停止
        # 要求：范围内至少2条包含旅游景点POI (包括锚点本身)
        check_limit = min(i + 10, n)
        poi_count = 0
        
        temp_idx = i
        while temp_idx < check_limit:
            # 遇到凌晨,立即停止计数扫描
            if df.at[temp_idx, 'is_early_morning']:
                break
            
            if df.at[temp_idx, 'has_poi_spot']:
                poi_count += 1
            temp_idx += 1
        
        # 如果密度不足2条,视为噪点,跳过
        if poi_count < 2:
            i += 1
            continue

        # 3. 确定结束边界 (延伸逻辑)
        # 规则：一直往下扫,直到 "连续5条无POI" 或 "扫描到凌晨停止"
        current_scan = i
        consecutive_no_spot = 0
        last_spot_idx = i  # 记录最后一次出现POI的位置
        
        while current_scan < n:
            # 终止条件A：遇到凌晨
            if df.at[current_scan, 'is_early_morning']:
                break
            
            # 状态更新
            if df.at[current_scan, 'has_poi_spot']:
                consecutive_no_spot = 0
                last_spot_idx = current_scan # 更新边界
            else:
                consecutive_no_spot += 1
            
            # 终止条件B：连续5条无景点POI
            if consecutive_no_spot >= 5:
                break
            
            current_scan += 1
        
        # 4. 执行标注
        # 范围：从 [第一条出现POI的锚点 i] 到 [最后一次出现POI的位置 last_spot_idx]
        # 策略：闭区间标注,填补中间空隙
        for k in range(i, last_spot_idx + 1):
            if df.at[k, 'scene_label'] == '' and df.at[k, 'has_poi_spot'] and df.at[k, 'has_any_move']:
                df.at[k, 'scene_label'] = '旅游参观'
        
        # 5. 更新主循环游标,跳过已处理区域
        i = last_spot_idx + 1

    return df



# ============================================================
# 旅游中途休息 
# ============================================================
def label_tour_mid_rest(df_local):
    df = df_local
    n = len(df)

    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'is_early_morning']:
                continue
            if df.at[i, 'has_any_move']:
                continue

            start_lookback = max(0, i - 10)
            window = df.iloc[start_lookback:i]
            if window.empty:
                continue

            if window['has_poi_spot'].any():
                df.at[i, 'scene_label'] = '旅游中途休息'

    return df

# ============================================================
# 旅游住宿休息 标注
# ============================================================
def label_tour_accommodation(df_local):
    df = df_local
    n = len(df)
    
    # 遍历每一行
    for i in range(n):
        # 触发条件：当前行未标注 且 时间为凌晨
        if df.at[i, 'scene_label'] == '' and df.at[i, 'is_early_morning']:
            
            # 定义回溯窗口：前10条
            start_lookback = max(0, i - 10)
            window = df.iloc[start_lookback:i] # 切片不包含 i 本身
            
            if window.empty:
                continue

            # 判断条件：前10条中至少出现过一次 (POI：旅游景点) 或者 (旅游app使用记录)
            has_spot_history = window['has_poi_spot'].any()
            has_app_history = window['has_travel_app'].any()
            
            if has_spot_history or has_app_history:
                df.at[i, 'scene_label'] = '旅游住宿休息'
                
    return df


# ============================================================
# 行程规划 标注
# ============================================================
def label_itinerary_planning(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            curr_map = df.at[i, 'has_map_app']
            curr_ticket = df.at[i, 'has_ticket_app']
            
            if curr_map or curr_ticket:
                start = max(0, i - 4)
                window = df.iloc[start : i + 1]
                
                win_map = window['has_map_app'].any()
                win_ticket = window['has_ticket_app'].any()
                
                match = False
                if curr_map and win_ticket:
                    match = True
                elif curr_ticket and win_map:
                    match = True
                    
                if match:
                    df.at[i, 'scene_label'] = '行程规划'
                    
    return df


# ============================================================
# 网约车标注
# ============================================================
def label_ride_scene_with_existing(df_local):
    df = df_local
    n = len(df)
    cur_idx = 0
    while cur_idx < n:
        cand = df[(df.index >= cur_idx) & (df['has_ride_keyword']) & (df['scene_label'] == '')].index
        if cand.empty:
            break
        A = cand.min()
        if df.at[A, 'has_any_move']:
            cur_idx = A + 1
            continue

        labeled_after = df[(df.index > A) & (df['scene_label'] != '')].index
        next_labeled_idx = labeled_after.min() if not labeled_after.empty else n

        window_end = min(A + 5, next_labeled_idx - 1, n - 1)
        if window_end < A:
            cur_idx = A + 1
            continue

        window = df.loc[A:window_end]
        if not window['has_any_move'].any():
            cur_idx = A + 1
            continue

        possible_moves = df[(df.index >= A) & (df.index < next_labeled_idx) & (df['has_any_move'])].index
        if possible_moves.empty:
            cur_idx = A + 1
            continue
        first_move = possible_moves.min()

        last_move = first_move
        i = first_move + 1
        while i < next_labeled_idx and i < n:
            if df.at[i, 'has_any_move']:
                if i == last_move + 1:
                    last_move = i
                else:
                    break
            else:
                break
            i += 1

        if first_move == A:
            cur_idx = A + 1
            continue

        if A <= first_move - 1:
            for j in range(A, first_move):
                if df.at[j, 'scene_label'] == '':
                    df.at[j, 'scene_label'] = '等待网约车'
        if first_move <= last_move - 1:
            for j in range(first_move, last_move):
                if df.at[j, 'scene_label'] == '':
                    df.at[j, 'scene_label'] = '乘坐网约车行程中'
        if df.at[last_move, 'scene_label'] == '':
            df.at[last_move, 'scene_label'] = '网约车到达终点'

        cur_idx = last_move + 1

    return df


# ============================================================
# 地铁标注
# ============================================================
def label_metro_with_existing(df_local):
    df = df_local
    n = len(df)
    cur_idx = 0

    while cur_idx < n:
        cand = df[(df.index >= cur_idx) & (df['has_poi_metro']) & (~df['has_any_move']) & (df['scene_label'] == '')].index
        if cand.empty:
            break
        A = cand.min()

        labeled_after = df[(df.index > A) & (df['scene_label'] != '')].index
        next_labeled_idx = labeled_after.min() if not labeled_after.empty else n

        window_end = min(A + 5, next_labeled_idx - 1, n - 1)
        if window_end < A:
            cur_idx = A + 1
            continue
        window = df.loc[A:window_end]
        if not window['has_fast_move'].any():
            cur_idx = A + 1
            continue

        i = A + 1
        stop_found = False
        stop_idx_first_nonmove = -1
        while i < next_labeled_idx and i < n:
            if (i + 1 < next_labeled_idx and i + 1 < n) and (not df.at[i, 'has_any_move']) and (not df.at[i + 1, 'has_any_move']):
                stop_found = True
                stop_idx_first_nonmove = i
                break
            i += 1

        if not stop_found:
            moves_after_A = df[(df.index >= A) & (df.index < next_labeled_idx) & (df['has_any_move'])].index
            if moves_after_A.empty:
                cur_idx = A + 1
                continue
            last_move = moves_after_A.max()
        else:
            search_idx = stop_idx_first_nonmove - 1
            last_move = None
            while search_idx >= A:
                if df.at[search_idx, 'has_any_move']:
                    last_move = search_idx
                    break
                search_idx -= 1
            if last_move is None:
                cur_idx = A + 1
                continue

        back_pois = df[(df.index <= last_move) & (df['has_poi_metro']) & (df['scene_label'] == '')].index
        if back_pois.empty:
            cur_idx = A + 1
            continue
        end_poi = back_pois.max()

        if end_poi == A:
            cur_idx = A + 1
            continue

        if not df.loc[A+1:last_move]['has_any_move'].any():
            cur_idx = A + 1
            continue

        if df.at[A, 'scene_label'] == '':
            df.at[A, 'scene_label'] = '抵达始发地铁站'
        if df.at[end_poi, 'scene_label'] == '':
            df.at[end_poi, 'scene_label'] = '抵达终点地铁站'

        for j in range(A + 1, last_move + 1):
            if j == end_poi:
                continue
            if df.at[j, 'scene_label'] == '' and df.at[j, 'has_any_move']:
                df.at[j, 'scene_label'] = '乘坐地铁中'

        cur_idx = last_move + 1

    return df


# ============================================================
# 外卖配送中 标注
# ============================================================
def label_take_out_delivery(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'has_take_out_app'] and df.at[i, 'has_any_move']:
                df.at[i, 'scene_label'] = '外卖配送中'
                
    return df


# ============================================================
# 外卖配送途中休息 标注
# ============================================================
def label_take_out_rest(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条不处于移动
            if not df.at[i, 'has_any_move']:
                # 前10条至少有一条有take_outapp特征
                start_lookback = max(0, i - 10)
                window = df.iloc[start_lookback:i]
                
                if window.empty:
                    continue
                    
                if window['has_take_out_app'].any():
                    df.at[i, 'scene_label'] = '外卖配送途中休息'
                    
    return df


# ============================================================
# 快递/货物配送接单等待 标注
# ============================================================
def label_goods_order_wait(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条无移动 且 本条不处于凌晨
            if not df.at[i, 'has_any_move'] and not df.at[i, 'is_early_morning']:
                # 前五条有goodsapp特征
                start = max(0, i - 5)
                # 前五条不包含本条
                window = df.iloc[start:i]
                
                if not window.empty and window['has_goods_app'].any():
                    df.at[i, 'scene_label'] = '快递/货物配送接单等待'
                    
    return df


# ============================================================
# 快递/货物配送途中 标注
# ============================================================
def label_goods_delivery_en_route(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'has_goods_app'] and df.at[i, 'has_any_move'] and not df.at[i, 'is_early_morning']:
                df.at[i, 'scene_label'] = '快递/货物配送途中'
                
    return df


# ============================================================
# 快递/货物配送配送住宿休息 标注
# ============================================================
def label_goods_delivery_accommodation_rest(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条不处于移动 且 本条处于凌晨
            if not df.at[i, 'has_any_move'] and df.at[i, 'is_early_morning']:
                # 前十条至少有一条有goodsapp特征
                start_lookback = max(0, i - 10)
                window = df.iloc[start_lookback:i]
                
                if window.empty:
                    continue
                    
                if window['has_goods_app'].any():
                    df.at[i, 'scene_label'] = '快递/货物配送配送中途休息'
                    
    return df


# ============================================================
# 网约车接单等待 标注
# ============================================================
def label_ride_hailing_order_wait(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条不处于移动 且 本条不处于凌晨
            if not df.at[i, 'has_any_move'] and not df.at[i, 'is_early_morning']:
                # 前五条至少有一条有driverapp特征
                start_lookback = max(0, i - 5)
                window = df.iloc[start_lookback:i]
                
                if window.empty:
                    continue
                    
                if window['has_driver_app'].any():
                    df.at[i, 'scene_label'] = '网约车接单等待'
                    
    return df


# ============================================================
# 网约车司机工作中 标注
# ============================================================
def label_ride_hailing_working(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'has_driver_app'] and df.at[i, 'has_any_move'] and not df.at[i, 'is_early_morning']:
                df.at[i, 'scene_label'] = '网约车司机工作中'
                
    return df


# ============================================================
# 上班通勤 标注
# ============================================================
def label_commuting_to_work(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'has_work_app'] and df.at[i, 'has_any_move'] and df.at[i, 'is_morning']:
                df.at[i, 'scene_label'] = '上班通勤'
                
    return df


# ============================================================
# 下班通勤 标注
# ============================================================
def label_commuting_home(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            if df.at[i, 'has_work_app'] and df.at[i, 'has_any_move'] and df.at[i, 'is_night']:
                df.at[i, 'scene_label'] = '下班通勤'
                
    return df


# ============================================================
# 自驾途中 (新增)
# ============================================================
def label_self_driving_en_route(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条处于高速移动
            if df.at[i, 'has_fast_move']:
                start_lookback = max(0, i - 5)
                # 前五条 (不含本条)
                prev_window = df.iloc[start_lookback:i]
                
                # 前五条及本条 (含本条)
                full_window = df.iloc[start_lookback:i+1]
                
                # 前五条有跨市
                cond_cross = prev_window['is_cross_city'].any() if not prev_window.empty else False
                
                # 前五条及本条无高铁机场POI
                has_gt = full_window['has_poi_gt'].any()
                has_air = full_window['has_poi_air'].any()
                
                if cond_cross and (not has_gt) and (not has_air):
                    df.at[i, 'scene_label'] = '自驾途中'
                    
    return df


# ============================================================
# 驾车抵达终点 (新增)
# ============================================================
def label_driving_arrival(df_local):
    df = df_local
    n = len(df)
    
    for i in range(n):
        if df.at[i, 'scene_label'] == '':
            # 本条处于高速移动 且 本条处于跨城市
            if df.at[i, 'has_fast_move'] and df.at[i, 'is_cross_city']:
                # 本条无高铁机场POI
                if not df.at[i, 'has_poi_gt'] and not df.at[i, 'has_poi_air']:
                    start_lookback = max(0, i - 8)
                    prev_window = df.iloc[start_lookback:i]
                    
                    if not prev_window.empty and prev_window['is_cross_city'].any():
                        df.at[i, 'scene_label'] = '驾车抵达终点'
                        
    return df


# ============================================================
# 单用户整体排程：GT -> AIR -> TOUR -> RIDE -> METRO
# ============================================================
def label_scene_combined_all(df_user):
    df_local = df_user.reset_index(drop=True).copy()
    if 'scene_label' not in df_local.columns:
        df_local['scene_label'] = ''

    # 1. 高铁
    s_gt, e_gt, _ = find_scene_range(df_local, 'gt')
    if s_gt != -1:
        df_local = label_gt_scene(df_local, s_gt, e_gt)

    # 2. 机场
    s_air, e_air, _ = find_scene_range(df_local, 'air')
    if s_air != -1:
        df_local = label_air_scene(df_local, s_air, e_air)

    # 4. 网约车
    df_local = label_ride_scene_with_existing(df_local)

    # 5. 地铁
    df_local = label_metro_with_existing(df_local)
    
    # 3. 行程规划 (新增, 优先级在机场之后)
    df_local = label_itinerary_planning(df_local)
    
    # 6. 外卖配送中
    df_local = label_take_out_delivery(df_local)
    
    # 7. 外卖配送途中休息
    df_local = label_take_out_rest(df_local)
    
    # 11. 快递/货物配送接单等待
    df_local = label_goods_order_wait(df_local)
    
    # 12. 快递/货物配送途中
    df_local = label_goods_delivery_en_route(df_local)

    # 13. 快递/货物配送配送住宿休息
    df_local = label_goods_delivery_accommodation_rest(df_local)

    # 14. 网约车接单等待
    df_local = label_ride_hailing_order_wait(df_local)

    # 15. 网约车司机工作中
    df_local = label_ride_hailing_working(df_local)

    # 16. 上班通勤
    df_local = label_commuting_to_work(df_local)

    # 17. 下班通勤
    df_local = label_commuting_home(df_local)

    # 18. 自驾途中
    df_local = label_self_driving_en_route(df_local)

    # 19. 驾车抵达终点
    df_local = label_driving_arrival(df_local)
    
    # 8. 旅游参观
    df_local = label_tour_visiting(df_local)

    # 9. 旅游住宿休息
    df_local = label_tour_accommodation(df_local)
    
    # 10. 旅游中途休息
    df_local = label_tour_mid_rest(df_local)

    return df_local


# ============================================================
# 并行处理子函数
# ============================================================
def process_sub_chunk(df_chunk):
    # 特征提取
    features = df_chunk['text'].apply(extract_features)
    df_chunk = pd.concat([df_chunk, features], axis=1)
    
    # 标签处理
    df_chunk['scene_label'] = ''
    
    # 使用显式迭代代替 apply 以避免 DeprecationWarning，并确保 udid 列保留
    results = []
    for _, group in df_chunk.groupby('udid'):
        results.append(label_scene_combined_all(group))
        
    if results:
        return pd.concat(results, ignore_index=True)
    else:
        return df_chunk


# ============================================================
# 大数据处理（chunk）
# ============================================================
def process_large_data_four_scenes(input_csv_path, output_csv_path, chunk_size=500000):

    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    output_cols = ['time', 'udid','scene_label', 'text', 'context', 'history_usage', 'service_click']
    cached_df = pd.DataFrame()

    reader = pd.read_csv(
        input_csv_path,
        chunksize=chunk_size,
        usecols=['time', 'udid', 'text', 'service_click', 'context', 'history_usage'],
        iterator=True,
        encoding='utf-8'
    )

    is_first_chunk = True
    
    # 启动并行进程池
    cpu_cores = cpu_count()
    print(f"启动并行处理，使用核心数：{cpu_cores}")

    with Pool(processes=cpu_cores) as pool:
        with tqdm( desc="处理进度", unit="行") as pbar:
            for chunk in reader:
                pbar.update(len(chunk))
                current_df = pd.concat([cached_df, chunk], ignore_index=True)
                
                # 处理用户ID被切断的情况
                last_udid = current_df['udid'].iloc[-1]
                # 找到该用户在当前chunk中第一次出现的索引
                split_idx = current_df[current_df['udid'] == last_udid].index.min()

                process_df = current_df.iloc[:split_idx].copy()
                cached_df = current_df.iloc[split_idx:].copy()

                if process_df.empty:
                    continue
                
                # --- 并行拆分任务 ---
                unique_udids = process_df['udid'].unique()
                if len(unique_udids) == 0:
                    continue
                
                # 将udid均匀分给各个核
                udid_splits = np.array_split(unique_udids, cpu_cores)
                sub_dfs = []
                for ids in udid_splits:
                    if len(ids) > 0:
                        # 筛选并从 process_df 中提取
                        sub_df = process_df[process_df['udid'].isin(ids)].copy()
                        if not sub_df.empty:
                            sub_dfs.append(sub_df)
                
                if sub_dfs:
                    # 并发执行
                    results = pool.map(process_sub_chunk, sub_dfs)
                    # 合并结果
                    df_labeled = pd.concat(results)
                    
                    df_final_chunk = df_labeled[output_cols]
                    df_final_chunk.to_csv(output_csv_path, mode='a', header=is_first_chunk, index=False, encoding='utf-8')
                    is_first_chunk = False

        # 处理最后缓存的剩余数据
        if not cached_df.empty:
            unique_udids = cached_df['udid'].unique()
            if len(unique_udids) > 0:
                udid_splits = np.array_split(unique_udids, cpu_cores)
                sub_dfs = []
                for ids in udid_splits:
                    if len(ids) > 0:
                        sub_df = cached_df[cached_df['udid'].isin(ids)].copy()
                        if not sub_df.empty:
                            sub_dfs.append(sub_df)
                
                if sub_dfs:
                    results = pool.map(process_sub_chunk, sub_dfs)
                    df_labeled = pd.concat(results)
                    df_final_chunk = df_labeled[output_cols]
                    df_final_chunk.to_csv(output_csv_path, mode='a', header=is_first_chunk, index=False, encoding='utf-8')

    print(f"\n处理完成,保存到：{output_csv_path}")


# 入口示例
if __name__ == "__main__":
    input_csv = "./单框架元旦七天语义化_ab.csv":d   
    output_csv = "../data/单框架元旦七天语义化ab_muban.csv"
    process_large_data_four_scenes(input_csv, output_csv, chunk_size=100000)
    