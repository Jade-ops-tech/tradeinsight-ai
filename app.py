import os

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

# 自定义 CSS 样式（目前留空，你可以在这里放全局样式）
st.markdown(
    """
    <style>
    /* 这里可以写自定义 CSS，例如：
    .main {
        background-color: #0b1020;
        color: #ffffff;
    }
    */
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================== 初始化 ====================
def init_session_state() -> None:
    """初始化 Streamlit session state。"""
    if "rag_engine" not in st.session_state:
        st.session_state.rag_engine = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


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
            st.success("✅ 知识库已清空")
            st.experimental_rerun()


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
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.write(message["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message["content"])

        # 显示检索来源
        if "sources" in message and message["sources"]:
            with st.expander("📚 查看检索来源"):
                for i, (source, doc) in enumerate(
                    zip(message["sources"], message["retrieved_docs"]),
                    start=1,
                ):
                    src = source.get("source", "未知")
                    chunk_id = source.get("chunk_id")
                    chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                    st.markdown(
                        f"**来源 {i}: {src} {chunk_info}**\n\n"
                        f"{doc[:200]}...",
                        unsafe_allow_html=True,
                    )


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
        with st.spinner("正在思考……"):
            try:
                result = st.session_state.rag_engine.ask(user_question)

                # 显示答案
                st.write(result["answer"])

                # 显示检索来源
                if result.get("sources"):
                    with st.expander("📚 查看检索来源"):
                        for i, (source, doc) in enumerate(
                            zip(result["sources"], result["retrieved_docs"]),
                            start=1,
                        ):
                            src = source.get("source", "未知")
                            chunk_id = source.get("chunk_id")
                            chunk_info = f"(chunk {chunk_id})" if chunk_id is not None else ""
                            st.markdown(
                                f"**来源 {i}: {src} {chunk_info}**\n\n"
                                f"{doc[:200]}...",
                                unsafe_allow_html=True,
                            )

                # 添加助手消息到历史
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result.get("sources", []),
                        "retrieved_docs": result.get("retrieved_docs", []),
                    }
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"❌ 出错了：{e}")


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
