import json
import os
import re
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv
import fitz  # PyMuPDF
import docx


# 加载环境变量
load_dotenv()

# 页面配置
st.set_page_config(
    page_title="TradeInsight AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义 CSS 样式：检索分数颜色、复制按钮等
st.markdown(
    """
    <style>
    .score-high   { color: #22c55e; font-weight: 600; }  /* 绿色 >0.8 */
    .score-mid    { color: #eab308; font-weight: 600; }   /* 黄色 0.6-0.8 */
    .score-low    { color: #94a3b8; font-weight: 500; }   /* 灰色 <0.6 */
    </style>
    """,
    unsafe_allow_html=True,
)

# 文档截断长度
RETRIEVAL_SNIPPET_LEN = 200


def highlight_keywords(text: str, query: str) -> str:
    """在文档片段中高亮用户查询的关键词（Markdown 粗体）。"""
    if not text or not query:
        return text or ""
    keywords = [kw for kw in query.split() if kw.strip()]
    for kw in keywords:
        pattern = re.compile(f"({re.escape(kw)})", re.IGNORECASE)
        text = pattern.sub(r"**\1**", text)
    return text


# ==================== 初始化 ====================
def init_session_state() -> None:
    """初始化 Streamlit session state。"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False

    if "retrieval_history" not in st.session_state:
        st.session_state.retrieval_history = []  # 每次查询的检索质量，用于趋势展示

    if "last_retrieval_for_export" not in st.session_state:
        st.session_state.last_retrieval_for_export = None  # 最近一次检索详情，用于导出报告


def init_rag_engine() -> bool:
    """初始化 RAG 引擎。"""
    if st.session_state.rag_engine is None:
        try:
            # 动态导入（避免在没有安装依赖时报错）
            from rag_engine import RAGEngine

            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                st.error("⚠️ 未找到 DeepSeek API 密钥！请在 .env 文件中配置 DEEPSEEK_API_KEY")
                return False

            st.session_state.rag_engine = RAGEngine(api_key=api_key)
            return True
        except Exception as e:  # noqa: BLE001
            st.error(f"❌ RAG 引擎初始化失败：{e}")
            return False
    return True


# ==================== 侧边栏 ====================
def render_sidebar() -> None:
    """渲染侧边栏。"""
    with st.sidebar:
        st.title("📚 知识库管理")

        # 显示知识库状态
        if st.session_state.rag_engine:
            doc_count = st.session_state.rag_engine.get_collection_count()
            st.metric("已加载文档数", doc_count)

        st.divider()

        # 重排模型开关
        st.subheader("🧠 高级设置")
        st.checkbox(
            "使用重排模型（FlagEmbedding）优化排序",
            value=st.session_state.get("use_reranker", False),
            key="use_reranker",
            help="勾选后会在混合检索结果之上，再用 bge-reranker 做一次精排，效果更好但会更慢。",
        )

        # 文件上传
        st.subheader("📤 上传文档")
        uploaded_file = st.file_uploader(
            "支持 TXT, MD, PDF, DOCX 格式",
            type=["txt", "md", "pdf", "docx"],
            help="上传交易教程、策略文档等文本文件（纯文本、Markdown、PDF、Word）",
        )

        if uploaded_file is not None:
            if st.button("🚀 处理并添加到知识库", type="primary"):
                process_uploaded_file(uploaded_file)

        st.divider()

        # 快速添加示例文档
        st.subheader("⚡ 快速测试")
        if st.button("添加示例文档"):
            add_sample_documents()

        st.divider()

        # 清空知识库
        st.subheader("🗑️ 管理")
        if st.button("清空知识库", type="secondary"):
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_collection()
            st.session_state.chat_history = []
            st.session_state.documents_loaded = False
            st.session_state.retrieval_history = []
            st.success("✅ 知识库已清空")
            st.experimental_rerun()

        st.divider()

        # 📈 检索历史对比
        render_retrieval_history_trend()

        st.divider()

        # 📥 导出检索报告
        st.subheader("📥 导出检索报告")
        last = st.session_state.get("last_retrieval_for_export")
        if last:
            report_md = generate_report(last, "md")
            report_json = generate_report(last, "json")
            st.download_button(
                label="下载 Markdown 报告",
                data=report_md,
                file_name=f"search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            st.download_button(
                label="下载 JSON 报告",
                data=report_json,
                file_name=f"search_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )
        else:
            st.caption("请先进行一次提问，即可导出最近一次检索详情。")


def _record_retrieval_quality(
    query: str,
    result: dict,
    reranker_used: bool,
) -> None:
    """记录本次查询的检索质量到 retrieval_history。"""
    history = st.session_state.get("retrieval_history", [])
    sources = result.get("sources") or []
    final_count = len(sources)
    rerank_list = (result.get("debug") or {}).get("rerank") or []
    if rerank_list:
        scores = [float(r.get("score") or 0) for r in rerank_list]
        avg_score = sum(scores) / len(scores) if scores else None
        max_score = max(scores) if scores else None
    else:
        avg_score = max_score = None

    entry = {
        "query": query,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "final_count": final_count,
        "avg_score": avg_score,
        "max_score": max_score,
        "reranker_used": reranker_used,
    }
    history.append(entry)
    # 只保留最近 50 条
    st.session_state.retrieval_history = history[-50:]


def generate_report(search_results: dict, fmt: str) -> str:
    """
    根据检索详情生成可导出的报告内容。
    search_results 需包含: query, answer, ts, sources, retrieved_docs, debug(可选)
    fmt: "md" | "json"
    """
    query = search_results.get("query", "")
    answer = search_results.get("answer", "")
    ts = search_results.get("ts", "")
    sources = search_results.get("sources", [])
    retrieved_docs = search_results.get("retrieved_docs", [])
    debug = search_results.get("debug") or {}

    if fmt == "json":
        # 可序列化结构（避免非基本类型）
        payload = {
            "query": query,
            "answer": answer,
            "export_time": ts,
            "num_sources": len(sources),
            "sources": [
                {**meta, "snippet": _truncate_doc(doc, RETRIEVAL_SNIPPET_LEN)}
                for meta, doc in zip(sources, retrieved_docs)
            ],
            "debug": {
                k: [
                    {"doc": (x.get("doc") or "")[:500], "meta": x.get("meta"), "score": x.get("score")}
                    for x in (v or [])
                ]
                for k, v in debug.items()
                if isinstance(v, list)
            },
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    # Markdown
    lines = [
        "# 检索报告",
        "",
        f"**导出时间**：{ts}",
        "",
        "## 用户问题",
        "",
        query,
        "",
        "## 答案",
        "",
        answer,
        "",
        "## 检索统计",
        "",
        f"- 命中片段数：{len(sources)}",
        "",
        "## 引用片段",
        "",
    ]
    for i, (meta, doc) in enumerate(zip(sources, retrieved_docs), 1):
        src = meta.get("source", "未知")
        chunk_id = meta.get("chunk_id", "")
        snippet = _truncate_doc(doc, RETRIEVAL_SNIPPET_LEN)
        lines.append(f"### {i}. {src} (chunk {chunk_id})")
        lines.append("")
        lines.append(snippet)
        lines.append("")
    return "\n".join(lines)


def render_retrieval_history_trend() -> None:
    """展示检索历史对比与趋势（表格 + 折线图）。"""
    history = st.session_state.get("retrieval_history", [])
    if not history:
        st.subheader("📈 检索历史对比")
        st.caption("暂无记录，提问后将在此展示每次检索质量与趋势。")
        return

    st.subheader("📈 检索历史对比")
    # 表格：时间、问题(截断)、命中数、平均相关性、最高相关性、重排
    max_query_len = 20
    rows = []
    for i, e in enumerate(history):
        q = (e.get("query") or "")[:max_query_len]
        if len(e.get("query") or "") > max_query_len:
            q += "…"
        rows.append({
            "序号": i + 1,
            "时间": e.get("ts", "")[-8:],  # HH:MM:SS
            "问题": q,
            "🎯 命中数": e.get("final_count", 0),
            "⭐ 平均相关性": f"{e['avg_score']:.2f}" if e.get("avg_score") is not None else "—",
            "⭐ 最高相关性": f"{e['max_score']:.2f}" if e.get("max_score") is not None else "—",
            "重排": "是" if e.get("reranker_used") else "否",
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # 趋势：有 avg_score 的条目画折线图
    scores_avail = [e for e in history if e.get("avg_score") is not None]
    if len(scores_avail) >= 2:
        st.caption("📉 平均相关性趋势（仅含使用重排的查询）")
        st.line_chart({"⭐ 平均相关性": [e["avg_score"] for e in scores_avail]})


def _extract_text_from_file(uploaded_file) -> str:
    """根据文件类型抽取纯文本内容。"""
    filename: str = uploaded_file.name
    suffix: str = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    def read_text_like(encoding: str = "utf-8") -> str:
        data = uploaded_file.read()
        try:
            return data.decode(encoding)
        except Exception:
            # 回退 latin-1，避免因编码报错
            return data.decode("latin-1", errors="ignore")

    if suffix in {"txt", "md"}:
        return read_text_like()

    if suffix == "pdf":
        # 使用 PyMuPDF 读取 PDF 内容
        data = uploaded_file.read()
        doc = fitz.open(stream=data, filetype="pdf")
        texts = []
        for page in doc:
            try:
                texts.append(page.get_text() or "")
            except Exception:
                continue
        return "\n\n".join(texts)

    if suffix == "docx":
        # 注意：docx.Document 需要一个二进制文件对象
        doc_obj = docx.Document(uploaded_file)
        return "\n\n".join(p.text for p in doc_obj.paragraphs)

    raise ValueError(f"暂不支持的文件格式：{suffix}")


def process_uploaded_file(uploaded_file) -> None:
    """处理上传的文件并添加到知识库。"""
    try:
        # 读取并解析文件内容（根据不同格式提取文本）
        content = _extract_text_from_file(uploaded_file)

        # 添加到知识库
        with st.spinner("正在处理文档……"):
            count = st.session_state.rag_engine.add_text_file(
                content,
                filename=uploaded_file.name,
            )

        st.success(f"✅ 成功添加 {count} 个文档块！")
        st.session_state.documents_loaded = True
        st.experimental_rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ 文件处理失败：{e}")


def add_sample_documents() -> None:
    """添加示例文档。"""
    sample_docs = [
        (
            "Fair Value Gap (FVG) 是 ICT 交易理论中的核心概念。"
            "它指的是价格快速移动时留下的未成交区域，在 K 线图上表现为三根 K 线之间的缺口。"
            "FVG 通常会被价格回补，因此可以作为潜在的支撑或阻力位。"
            "在上涨趋势中，FVG 可以作为买入区域；在下跌趋势中，FVG 可以作为卖出区域。"
        ),
        (
            "Order Block（订单块）是机构订单集中的价格区域。"
            "它通常是趋势反转前的最后一根反向 K 线。"
            "例如，在上涨趋势中，Order Block 是最后一根下跌 K 线；"
            "在下跌趋势中，Order Block 是最后一根上涨 K 线。"
            "机构会在这些区域建立大量仓位，因此价格回到 Order Block 时往往会产生强烈的反应。"
        ),
        (
            "Liquidity（流动性）是市场运动的燃料。机构需要流动性来执行大额订单。"
            "流动性主要存在于两个地方：明显高点之上（买方止损）和明显低点之下（卖方止损）。"
            "市场经常会扫荡这些流动性区域，触发散户的止损单，然后反向运动。"
            "理解流动性的概念可以帮助交易者预判市场的下一个目标。"
        ),
    ]

    try:
        count = st.session_state.rag_engine.add_documents(
            sample_docs,
            metadatas=[
                {"source": "ICT 教程", "topic": "Fair Value Gap"},
                {"source": "ICT 教程", "topic": "Order Block"},
                {"source": "ICT 教程", "topic": "Liquidity"},
            ],
        )
        st.success(f"✅ 成功添加 {count} 个示例文档！")
        st.session_state.documents_loaded = True
        st.experimental_rerun()
    except Exception as e:  # noqa: BLE001
        st.error(f"❌ 添加失败：{e}")


# ==================== 检索可视化 ====================
# 颜色编码：🟢 优秀 (>0.8)  🟡 良好 (0.6-0.8)  ⚪ 一般 (<0.6)
def _score_emoji(score: float) -> tuple:
    """返回 (颜色 emoji, 星级字符串)。"""
    if score >= 0.8:
        return ("🟢", "⭐⭐⭐")  # 优秀
    if score >= 0.6:
        return ("🟡", "⭐⭐")   # 良好
    return ("⚪", "⭐")         # 一般


def _score_bar(score: float, width: int = 20) -> str:
    """生成条形可视化：████████████████░░░░ 0.95"""
    filled = round(min(1.0, max(0.0, score)) * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"{bar} {score:.2f}"


def _truncate_doc(doc: str, max_len: int = 200) -> str:
    """文档内容截断到 max_len 字，避免太长。"""
    if not doc:
        return ""
    doc = doc.strip()
    if len(doc) <= max_len:
        return doc
    return doc[:max_len].rstrip() + "…"


def render_retrieval_visualization(
    sources: list,
    retrieved_docs: list,
    debug: dict,
    query: str = "",
) -> None:
    """
    用 expander + metric + progress + 截断 + 复制，展示检索结果与统计。
    query 用于在文档片段中高亮用户关键词。
    """
    if not sources or not retrieved_docs:
        return

    # 是否有 rerank 分数（用于统计和进度条）
    rerank_list = (debug or {}).get("rerank") or []
    has_scores = len(rerank_list) >= len(retrieved_docs)
    if has_scores and rerank_list:
        scores = [float(r.get("score") or 0) for r in rerank_list[: len(retrieved_docs)]]
        avg_score = sum(scores) / len(scores) if scores else None
        max_score = max(scores) if scores else None
    else:
        scores = [None] * len(retrieved_docs)
        avg_score = max_score = None

    with st.expander("🔍 检索详情", expanded=True):
        # 统计数据：st.metric（emoji 增强可读性）
        n = len(sources)
        cols = st.columns(3)
        cols[0].metric("🎯 命中片段数", n)
        if avg_score is not None:
            cols[1].metric("⭐ 平均相关性", f"{avg_score:.2f}")
        if max_score is not None:
            cols[2].metric("⭐ 最高相关性", f"{max_score:.2f}")

        st.markdown("---")
        st.markdown("**📄 文档引用（可复制）**")

        for i, (source, doc) in enumerate(zip(sources, retrieved_docs)):
            score = scores[i] if i < len(scores) else None
            meta = source if isinstance(source, dict) else {}
            src_label = meta.get("source", "未知")
            chunk_id = meta.get("chunk_id")
            chunk_info = f" (chunk {chunk_id})" if chunk_id is not None else ""

            # 相关性：条形 + 分数 + 颜色 emoji + 星级
            if score is not None:
                color_emoji, stars = _score_emoji(score)
                bar_str = _score_bar(score)
                st.markdown(
                    f"**📄 文档{i + 1}** {src_label}{chunk_info}  \n"
                    f"`{bar_str}` {color_emoji} {stars}"
                )
                st.progress(min(1.0, max(0.0, score)))
            else:
                st.markdown(f"**📄 文档{i + 1}** {src_label}{chunk_info}")

            # 文档截断 200 字 + 关键词高亮 + 复制（st.code 自带复制按钮）
            snippet = _truncate_doc(doc, RETRIEVAL_SNIPPET_LEN)
            if query:
                st.markdown(highlight_keywords(snippet, query))
            st.code(snippet, language=None)
            st.caption("↑ 点击代码块右侧图标可复制")
            st.markdown("")


# ==================== 主界面 ====================
def render_main_area() -> None:
    """渲染主界面。"""
    # 标题
    st.markdown(
        """
        <h1 style="margin-bottom: 0.2rem;">🚀 TradeInsight AI</h1>
        <p style="color: #888;">智能交易知识助手 - 基于 RAG 技术的专业问答系统</p>
        """,
        unsafe_allow_html=True,
    )

    # 检查是否已加载文档
    if not st.session_state.documents_loaded and st.session_state.rag_engine:
        doc_count = st.session_state.rag_engine.get_collection_count()
        if doc_count == 0:
            st.info("👈 请先在左侧上传文档或添加示例文档")
            return

    # 显示对话历史
    render_chat_history()

    # 问答输入框
    render_chat_input()


def render_chat_history() -> None:
    """渲染对话历史。"""
    last_user_question = ""
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            last_user_question = message.get("content", "")
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("**📊 答案**")
                st.write(message["content"])

        # 显示检索来源（混合检索的文档片段，含关键词高亮）
        if "sources" in message and message["sources"]:
            with st.expander("🔍 检索过程（混合检索：向量 + BM25）"):
                for i, (source, doc) in enumerate(
                    zip(message["sources"], message["retrieved_docs"]),
                    start=1,
                ):
                    src = source.get("source", "未知")
                    chunk_id = source.get("chunk_id")
                    chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                    snippet = _truncate_doc(doc, RETRIEVAL_SNIPPET_LEN)
                    highlighted = highlight_keywords(snippet, last_user_question)
                    st.markdown(
                        f"**来源 {i}: {src} {chunk_info}**\n\n{highlighted}",
                        unsafe_allow_html=True,
                    )

        # 可视化：统计 + 相关性进度条 + 截断 + 复制（含关键词高亮）
        if "sources" in message and message["sources"]:
            render_retrieval_visualization(
                message["sources"],
                message["retrieved_docs"],
                message.get("debug"),
                query=last_user_question,
            )

        # 显示更详细的检索思考过程（向量 / BM25 / 融合）
        debug = message.get("debug")
        if debug:
            with st.expander("🧠 思考过程明细（向量检索 / BM25 / 融合排序）"):
                for key, label in [
                    ("vector", "🔍 向量检索 TopK"),
                    ("bm25", "🔍 BM25 检索 TopK"),
                    ("fused", "🎯 RRF 融合 TopK（最终喂给大模型）"),
                ]:
                    items = debug.get(key) or []
                    if not items:
                        continue
                    st.markdown(f"**{label}**")
                    for idx, item in enumerate(items, start=1):
                        meta = item.get("meta") or {}
                        src = meta.get("source", "未知")
                        chunk_id = meta.get("chunk_id")
                        chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                        doc = (item.get("doc") or "")[:200]
                        doc_hl = highlight_keywords(doc, last_user_question)
                        st.markdown(
                            f"- **{idx}. {src} {chunk_info}**\n\n{doc_hl}...",
                            unsafe_allow_html=True,
                        )
                    st.markdown("---")


def render_chat_input() -> None:
    """渲染问答输入框。"""
    # 使用 chat_input 组件
    user_question = st.chat_input("请输入你的问题……")

    if not user_question:
        return

    # 添加用户消息到历史
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.write(user_question)

    # 调用 RAG 引擎获取答案
    with st.chat_message("assistant", avatar="🤖"):
        use_reranker = st.session_state.get("use_reranker", False)

        try:
            # 如果勾选了重排模型且尚未加载，则先展示专门的加载提示
            if use_reranker and not st.session_state.get("reranker_loaded", False):
                with st.spinner("正在加载 Reranker 模型...(首次加载可能较慢)"):
                    ok = st.session_state.rag_engine.ensure_reranker_loaded()
                st.session_state.reranker_loaded = ok

            # 正式思考阶段
            with st.spinner("正在思考……"):
                result = st.session_state.rag_engine.ask(
                    user_question,
                    use_reranker=use_reranker,
                )

        except Exception as e:  # noqa: BLE001
            st.error(f"❌ 出错了：{e}")
            return

        # 📊 答案
        st.markdown("**📊 答案**")
        st.write(result["answer"])

        # 显示检索来源（混合检索的文档片段，含关键词高亮）
        if result.get("sources"):
            with st.expander("🔍 检索过程（混合检索：向量 + BM25）"):
                for i, (source, doc) in enumerate(
                    zip(result["sources"], result["retrieved_docs"]),
                    start=1,
                ):
                    src = source.get("source", "未知")
                    chunk_id = source.get("chunk_id")
                    chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                    snippet = _truncate_doc(doc, RETRIEVAL_SNIPPET_LEN)
                    highlighted = highlight_keywords(snippet, user_question)
                    st.markdown(
                        f"**来源 {i}: {src} {chunk_info}**\n\n{highlighted}",
                        unsafe_allow_html=True,
                    )

        # 可视化：统计 + 相关性进度条 + 截断 + 复制（含关键词高亮）
        if result.get("sources"):
            render_retrieval_visualization(
                result["sources"],
                result["retrieved_docs"],
                result.get("debug"),
                query=user_question,
            )

        # 显示更详细的检索思考过程（向量 / BM25 / 融合）
        debug = result.get("debug")
        if debug:
            with st.expander("🧠 思考过程明细（向量检索 / BM25 / 融合排序）"):
                for key, label in [
                    ("vector", "🔍 向量检索 TopK"),
                    ("bm25", "🔍 BM25 检索 TopK"),
                    ("fused", "🎯 RRF 融合 TopK（最终喂给大模型）"),
                ]:
                    items = debug.get(key) or []
                    if not items:
                        continue
                    st.markdown(f"**{label}**")
                    for idx, item in enumerate(items, start=1):
                        meta = item.get("meta") or {}
                        src = meta.get("source", "未知")
                        chunk_id = meta.get("chunk_id")
                        chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                        doc = (item.get("doc") or "")[:200]
                        doc_hl = highlight_keywords(doc, user_question)
                        st.markdown(
                            f"- **{idx}. {src} {chunk_info}**\n\n{doc_hl}...",
                            unsafe_allow_html=True,
                        )
                st.markdown("---")

        # 记录本次检索质量，用于历史对比与趋势
        _record_retrieval_quality(user_question, result, use_reranker)

        # 保存最近一次检索详情，供导出报告使用
        st.session_state.last_retrieval_for_export = {
            "query": user_question,
            "answer": result.get("answer", ""),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "sources": result.get("sources", []),
            "retrieved_docs": result.get("retrieved_docs", []),
            "debug": result.get("debug"),
        }

        # 添加助手消息到历史
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", []),
                "retrieved_docs": result.get("retrieved_docs", []),
                "debug": result.get("debug"),
            }
        )


# ==================== 主程序 ====================
def main() -> None:
    """主程序入口。"""
    # 初始化
    init_session_state()

    # 初始化 RAG 引擎
    if not init_rag_engine():
        st.stop()

    # 渲染界面
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
