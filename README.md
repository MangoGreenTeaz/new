# 项目说明

## 目录

- [项目概览](#项目概览)
- [推荐使用流程](#推荐使用流程)
- [当前使用中的文件](#当前使用中的文件)
  - [feature.py](#featurepy)
  - [scene_label.py](#scene_labelpy)
  - [order.py](#orderpy)
  - [feature_output_schema.md](#feature_output_schemamd)
  - [scene_rules.md](#scene_rulesmd)
  - [order_rules.md](#order_rulesmd)
- [文档跳转](#文档跳转)

## 项目概览

本项目包含三步核心处理流程：

1. 使用 `feature.py` 对原始文本数据做特征提取，生成结构化特征数据。
2. 使用 `scene_label.py` 在特征数据基础上进行场景识别与 `scene_label` 标注。
3. 使用 `order.py` 从场景标注结果中提取高铁/飞机订单信息，生成 `order` 字段。

## 推荐使用流程

1. 先运行 `feature.py`
   - 输入原始 `in.csv`
   - 输出特征化后的 `out.csv`
2. 再运行 `scene_label.py`
   - 输入 `out.csv`
   - 输出带场景标签的 `scene_out.csv`
3. 最后运行 `order.py`
   - 输入 `scene_out.csv`（即 `scene_label.py` 的输出）
   - 输出带订单信息的 `a_order.csv`

## 当前使用中的文件

### `feature.py`

作用：

- 读取原始文本数据
- 提取 `city`、`poi`、App 特征、移动特征、时间特征
- 生成后续场景标注所需的结构化输入数据

配套说明文档：[`feature_output_schema.md`](./feature_output_schema.md)

### `scene_label.py`

作用：

- 读取 `feature.py` 输出的特征数据
- 按优先级依次执行场景规则
- 生成 `scene_label` 标注结果

配套说明文档：[`scene_rules.md`](./scene_rules.md)

### `order.py`

作用：

- 读取 `scene_label.py` 输出的场景标注数据
- 从高铁和飞机场景中提取订单信息（出发/到达时间、出发/到达城市）
- 生成结构化的 `order` 字段

配套说明文档：[`order_rules.md`](./order_rules.md)

### `feature_output_schema.md`

作用：

- 说明 `feature.py` 输出数据的字段结构
- 说明每个字段的含义、类型、生成规则

快速跳转：[`feature_output_schema.md`](./feature_output_schema.md)

### `scene_rules.md`

作用：

- 维护 `scene_label.py` 中所有场景规则的人类可读说明
- 记录场景优先级、判定逻辑、依赖字段和标签含义

快速跳转：[`scene_rules.md`](./scene_rules.md)

### `order_rules.md`

作用：

- 维护 `order.py` 中订单生成规则的人类可读说明
- 记录订单类型、生成逻辑、字段提取规则和输出格式

快速跳转：[`order_rules.md`](./order_rules.md)

## 文档跳转

- 跳转到特征输出字段说明：[`feature_output_schema.md`](./feature_output_schema.md)
- 跳转到场景规则说明：[`scene_rules.md`](./scene_rules.md)
- 跳转到订单生成规则说明：[`order_rules.md`](./order_rules.md)
