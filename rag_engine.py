import os
from typing import Dict, List, Optional, Tuple

import chromadb
from openai import OpenAI


class RAGEngine:
    """RAG 引擎类 - 封装完整的 RAG 流程（ChromaDB + DeepSeek Chat）。"""

    def __init__(self, api_key: Optional[str] = None, collection_name: str = "trade_knowledge") -> None:
        """
        初始化 RAG 引擎。

        Args:
            api_key: DeepSeek API 密钥，如果不提供则从环境变量读取。
            collection_name: ChromaDB 集合名称。
        """
        # 1. 初始化 DeepSeek 客户端
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("请提供 DeepSeek API 密钥或设置 DEEPSEEK_API_KEY 环境变量")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com",
        )

        # 2. 初始化 ChromaDB（使用内存或默认配置）
        self.chroma_client = chromadb.Client()
        self.collection_name = collection_name

        # 尝试获取已存在的集合，如果不存在则创建新的
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=collection_name)

    def add_documents(self, documents: List[str], metadatas: Optional[List[Dict]] = None) -> int:
        """
        添加文档到知识库。

        Args:
            documents: 文档内容列表。
            metadatas: 文档元数据列表（可选）。

        Returns:
            添加的文档数量。
        """
        if not documents:
            return 0

        # 生成文档 ID（简单递增，如果需要更健壮可以改成 uuid）
        doc_ids = [f"doc_{i}" for i in range(len(documents))]

        # 如果没有提供元数据，使用默认值
        if metadatas is None:
            metadatas = [{"source": f"document_{i}"} for i in range(len(documents))]

        # 添加到 ChromaDB
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=doc_ids,
        )

        return len(documents)

    def add_text_file(self, file_content: str, filename: str = "uploaded_file") -> int:
        """
        添加文本文件到知识库（自动分块）。

        Args:
            file_content: 文件内容。
            filename: 文件名。

        Returns:
            添加的文档块数量。
        """
        # 简单分块策略：按段落分割
        chunks = self._split_text(file_content)

        # 为每个块添加元数据
        metadatas = [{"source": filename, "chunk_id": i} for i in range(len(chunks))]

        return self.add_documents(chunks, metadatas)

    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        文本分块策略。

        Args:
            text: 原始文本。
            chunk_size: 每块的最大字符数。
            overlap: 相邻块之间的重叠字符数（当前未使用，后续可扩展）。

        Returns:
            分块后的文本列表。
        """
        # 先按段落分割（以空行作为段落分隔）
        paragraphs = text.split("\n\n")

        chunks: List[str] = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # 如果当前块加上新段落不超过限制，就合并
            if len(current_chunk) + len(para) <= chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
            else:
                # 否则保存当前块，开始新块
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para

        # 添加最后一块
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def search(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        语义检索。

        Args:
            query: 查询问题。
            n_results: 返回结果数量。

        Returns:
            (检索到的文档列表, 元数据列表)。
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        return documents, metadatas

    def ask(self, question: str, n_results: int = 3) -> Dict:
        """
        RAG 问答 - 核心方法。

        Args:
            question: 用户问题。
            n_results: 检索文档数量。

        Returns:
            包含答案、检索结果等信息的字典。
        """
        # 1. 检索相关文档
        retrieved_docs, metadatas = self.search(question, n_results)

        if not retrieved_docs:
            return {
                "answer": "抱歉，我的知识库中还没有相关信息。请先上传一些文档。",
                "sources": [],
                "retrieved_docs": [],
            }

        # 2. 构建上下文
        context = "\n\n---\n\n".join(retrieved_docs)

        # 3. 构建 Prompt
        prompt = (
            "你是一个专业的交易知识助手。请基于以下参考资料回答用户的问题。\n\n"
            "参考资料：\n"
            f"{context}\n\n"
            f"用户问题：{question}\n\n"
            "请基于参考资料给出准确、专业的回答。如果参考资料中没有相关信息，请明确说明。"
        )

        # 4. 调用 LLM 生成答案
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的交易知识助手，擅长解释 ICT、SMC 等交易理论。",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # 5. 返回结果
        return {
            "answer": answer,
            "sources": metadatas,
            "retrieved_docs": retrieved_docs,
        }

    def get_collection_count(self) -> int:
        """获取知识库中的文档数量。"""
        return self.collection.count()

    def clear_collection(self) -> None:
        """清空知识库（重新创建集合）。"""
        # 删除旧集合
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            # 如果不存在也没关系
            pass

        # 创建新集合
        self.collection = self.chroma_client.create_collection(name=self.collection_name)


def create_rag_engine(api_key: Optional[str] = None) -> RAGEngine:
    """便捷函数：创建一个 RAG 引擎实例。"""
    return RAGEngine(api_key=api_key)


if __name__ == "__main__":
    # 简单自测代码：从环境读取密钥，构建引擎并做一次问答
    from dotenv import load_dotenv

    load_dotenv()

    engine = RAGEngine()

    test_docs = [
        "Fair Value Gap (FVG) 是 ICT 理论中的核心概念，指的是价格快速移动时留下的未成交区域。",
        "Order Block 是机构订单集中的区域，通常是趋势反转前的最后一根反向 K 线。",
    ]

    count = engine.add_documents(test_docs)
    print(f"✅ 已添加 {count} 个文档")

    result = engine.ask("什么是 Fair Value Gap?")
    print("问题：什么是 Fair Value Gap?")
    print("答案：", result["answer"])
    print("来源数量：", len(result["sources"]))

