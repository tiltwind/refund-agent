"""Step 3 · 过滤 —— 进索引之前先用元数据把候选池砍到可控规模。

**这一步只做硬约束，不做任何相关性判断。** 两者混在一起是很多检索事故的
起点：相关性是可以商量的（排后面一点无非是精度损失），硬约束不行
（一条已废止的条款排在第 8 名，和排在第 1 名一样是错的）。

具体到这里，只有两类硬约束：

1. **生效日期**。已生效订单适用下单时的规则版本，所以必须在检索前排除尚未
   生效和已废止的版本 —— **不能指望模型从检索结果里事后辨别哪条还有效**。
   政策改版时新增版本而非原地覆盖（旧版 expire_date 置为改版日），
   旧版本留在库里可查，靠的就是这个过滤把它挡在候选池外。
2. **层级范围**。由路由给出（route.py），是「往哪一层倾斜」的执行面。

**时间不做软过滤，freshness 也不参与重排。** 政策是常青内容 ——
一条 2024 年生效、至今未改的条款不会因为「旧」而变得不适用。给它加时间衰减
只会让长期有效的核心条款（P02）输给刚发布的边缘规则。「哪一版有效」是正确性
判据，交给上面的硬过滤；它不是排序信号。
"""

from datetime import date


def build_filter(layers: list[str], today: str | None = None) -> str:
    """生成 Milvus 的 filter 表达式。

    日期用 ISO 字符串比较 —— `YYYY-MM-DD` 的字典序就是时间序，
    不需要 Milvus 支持日期类型。
    """
    day = today or date.today().isoformat()
    layer_list = ", ".join(f'"{x}"' for x in layers)
    return (
        f'effective_date <= "{day}" and expire_date > "{day}" '
        f"and layer in [{layer_list}]"
    )
