"""切片的数据模型。

只有 `Chunk` 会入库。父块**不单独存储** —— 它由「同一 parent_id 下的全部子块
按 chunk_index 拼接」还原（services/rag/pipeline/assemble.py）。这样做的前提是
子块之间 overlap = 0（段落边界即语义边界，见 policy.py），拼接能精确还原原文。

代价换来的是：没有第二个 collection、没有旁路的 KV 存储、没有「父块存了但子块
删了」这类不一致 —— 单一事实源仍然只有一个 Milvus collection。
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocMeta:
    """一篇政策文档的 frontmatter，逐字段进 Milvus 的标量列。

    这些字段不进 embedding（除了 title 与 tags 会被拼进块头），
    它们的用途是**过滤**与**重排加权** —— 生效日期决定条款还算不算数，
    authority_level 决定法规与平台规则冲突时谁说了算。
    """

    doc_id: str
    title: str
    layer: str
    """law（法律法规）| platform（平台政策）。路由的主要依据 —— 答复消费者优先
    召回 platform，判断平台条款是否有效时才召回 law。"""
    category: str
    doc_type: str
    authority: str
    """发布主体：法规是发布机关，平台政策是 publisher。"""
    effective_date: str
    expire_date: str
    authority_level: int
    """效力位阶：1 法律，2 行政法规与部门规章，3 平台规则。数字越小效力越高。"""
    version: str
    retrieval_scope: str
    tags: tuple[str, ...]
    source_path: str
    """仓库内相对路径，答复里做引用定位用。"""


@dataclass
class Chunk:
    """检索单元（子块）。"""

    chunk_id: str
    parent_id: str
    chunk_index: int
    """在父块内的序号，装配时按它排序拼回父块。"""
    doc: DocMeta
    section_path: tuple[str, ...]
    """标题路径，不含文档大标题（那是 doc.title）。"""
    body: str
    """原文片段。**喂给模型的是它**，块头不进上下文（块头信息由 section 名承载）。"""
    kind: str = "text"
    """text | table | code —— 表格与代码是原子块，任何情况下不再切分。"""
    parent_seq: int = 0
    """父块在文档内的顺序号，装配时判断两个父块是否相邻。"""

    @property
    def header(self) -> str:
        """块头 —— 拼在正文前一起参与 embedding 与 BM25。

        这是投入产出比最高的一处优化，比换切分算法便宜得多：正文被切成
        「3.1 无理由退货要求商品未拆封、不影响二次销售」这一段后，它不再带有
        「哪篇文档、第几条」的信息；块头补回文档标题与标题路径，
        「拆封了还能不能无理由退」这个问题才能稳定命中它。

        **只放这两行**。生效日期、tags 这类**文档级常量**故意不进块头：
        同一篇文档的每个块都带上它们，对文档内部的区分度是零，却会稀释短块
        （法规层的条文块正文常常只有 100 token，再加 90 token 的常量块头，
        向量就被这堆重复内容带偏了）。它们各自去该去的地方 —— 生效日期进标量
        字段参与硬过滤，tags 进标量字段供人工排查，都不需要挤进向量。
        """
        lines = [f"【文档】{self.doc.doc_id} {self.doc.title}"]
        if self.section_path:
            lines.append(f"【路径】{' > '.join(self.section_path)}")
        return "\n".join(lines)

    @property
    def text(self) -> str:
        """入库的可检索文本 = 块头 + 正文。dense 与 BM25 索引的都是它。"""
        return f"{self.header}\n\n{self.body}"

    @property
    def section(self) -> str:
        """人读的来源名，答复里引用时展示。"""
        path = " > ".join(self.section_path)
        return f"{self.doc.doc_id} {self.doc.title}" + (f" > {path}" if path else "")


@dataclass
class Section:
    """切分过程的中间态：一个标题路径下的直接正文。不入库。"""

    seq: int
    path: tuple[str, ...]
    text: str
    blocks: list["Block"] = field(default_factory=list)


@dataclass
class Block:
    """段落级单元：一个自然段、一张表、一段代码。"""

    kind: str
    text: str

    @property
    def atomic(self) -> bool:
        """表格与代码不可切分。

        半张表比没有表更糟：表头在 A 块、数据在 B 块，两个块单独看都会给出
        与完整规则相反的结论。P02 第六条那张审核顺序表就是典型 ——
        只截到第 3 行会得出「高风险账户一律拒绝」，而完整表的第 5.5 条说的是
        「高风险只改变处理通道，不预先否定诉求」。
        """
        return self.kind in ("table", "code")
