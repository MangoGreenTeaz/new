# 项目说明

## 第一步：特征提取

- [`feature.py`](./feature.py)：从原始文本中抽取 `city`、`poi`、App、移动和时间等结构化特征，是后续所有规则的输入基础。它把非结构化内容整理成可直接用于场景识别的数据列。
- [`feature_rules.md`](./feature_rules.md)：说明 `feature.py` 的输出字段、含义和生成方式。适合在新增字段或排查特征缺失时查看。
- 默认输入输出：`data.csv` → `data_feature.csv`

## 第二步：场景标注

- [`scene_label.py`](./scene_label.py)：基于特征数据做场景识别，为每条记录写入 `scene_label`。规则按优先级依次执行，适合先看整体场景，再看具体子场景。
- [`scene_rules.md`](./scene_rules.md)：记录 `scene_label.py` 的规则细节、优先级和判定条件。这里是修改或理解场景逻辑时的对照说明。
- 默认输入输出：`data_feature.csv` → `data_feature_label.csv`

## 第三步：订单提取

- [`order.py`](./order.py)：从高铁、飞机、酒店和旅游相关场景中提取订单信息，生成结构化 `order` 字段。它关注的是出发/到达时间、城市以及酒店和旅游的日期型订单。
- [`order_rules.md`](./order_rules.md)：说明 `order.py` 的订单提取规则、场景分类、字段含义和输出格式。适合在调整订单识别逻辑前先参考。
- 默认输入输出：`data_feature_label.csv` → `data_feature_label_order.csv`

## 第四步：文本合并

- [`merge.py`](./merge.py)：将当前 `text`、`order`、时间提示和历史文本拼成 `MERGED_TEXT`。它主要用于训练或推理前的上下文增强，把单条记录扩展成更完整的输入。
- [`merge_rules.md`](./merge_rules.md)：说明 `merge.py` 的拼接顺序、时间提示规则和历史窗口逻辑。适合确认最终拼接结果为什么长成这样。
- 默认输入输出：`data_feature_label_order.csv` → `data_feature_label_order_merge.csv`
