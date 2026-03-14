import os
from typing import Any, Dict, List, Optional, Tuple, Union

import chromadb
import jieba
from openai import OpenAI
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker


# RAG 问答使用的 System Prompt 常量
SYSTEM_PROMPT = (
    "你是一个专业的交易知识助手，擅长解释 ICT、SMC 等交易理论。"
    "请严格基于用户提供的参考资料回答问题；若参考资料中无相关信息，请明确说明。"
    "回答应简洁、准确、有条理。"
)


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

        # 2. 初始化 ChromaDB（使用本地持久化存储，避免默认的远程 tenant 配置）
        # 持久化目录放在项目的 data/chroma 文件夹下
        base_dir = os.path.dirname(os.path.abspath(__file__))
        persist_dir = os.path.join(base_dir, "data", "chroma")
        os.makedirs(persist_dir, exist_ok=True)

        # 使用 PersistentClient，而不是默认的 Client（默认现在会尝试连远程 tenant）
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        self.collection_name = collection_name

        # 尝试获取已存在的集合，如果不存在则创建新的
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.chroma_client.create_collection(name=collection_name)

        # 3. 初始化 BM25 相关结构（只在内存中，不持久化）
        self._bm25_corpus: List[str] = []          # 原始文档文本列表
        self._bm25_metadatas: List[Dict] = []      # 与 corpus 对齐的元数据列表
        self._bm25_tokenized: List[List[str]] = [] # 分词后的文档
        self._bm25_model: Optional[BM25Okapi] = None

        # 4. 重排模型（FlagEmbedding）懒加载句柄
        self._reranker: Optional[FlagReranker] = None

    # ==================== 公共接口 ====================

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

        # 同步更新 BM25 语料和索引
        self._bm25_corpus.extend(documents)
        self._bm25_metadatas.extend(metadatas)
        tokenized_new = [self._tokenize(text) for text in documents]
        self._bm25_tokenized.extend(tokenized_new)

        if self._bm25_tokenized:
            self._bm25_model = BM25Okapi(self._bm25_tokenized)

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

    # ==================== 混合检索（向量 + BM25） ====================

    def hybrid_search(
        self,
        query: str,
        n_results: int = 3,
        use_reranker: bool = False,
        return_metadata: bool = False,
    ) -> Union[Tuple[List[str], List[Dict], Dict], Dict[str, Any]]:
        """
        混合检索接口：向量检索 + BM25 检索，使用 RRF 进行结果融合。

        Args:
            query: 查询问题。
            n_results: 返回结果数量（top_k）。
            use_reranker: 是否使用重排模型对融合结果精排。
            return_metadata: 是否返回检索元数据（search_type、reranker_used、total_candidates 等）。

        Returns:
            - 当 return_metadata=False：(fused_docs, fused_metas, debug_info)，与原有行为一致。
            - 当 return_metadata=True：{
                "results": [{"doc": ..., "meta": ..., "score": ... or None}, ...],
                "metadata": {
                    "search_type": "hybrid" | "vector",
                    "reranker_used": bool,
                    "total_candidates": int,
                    "final_count": int,
                    "avg_score": float | None,
                    "max_score": float | None,
                },
                "debug_info": debug_info,
              }
        """
        debug_info: Dict[str, List] = {"vector": [], "bm25": [], "fused": [], "rerank": []}
        total_candidates = 0
        search_type = "vector"
        reranker_used = False
        rerank_items: List[Dict] = []

        # 如果 BM25 尚未构建，则退化为纯向量检索
        if not self._bm25_model or not self._bm25_corpus:
            docs, metas = self.search(query, n_results)
            total_candidates = len(docs)
            debug_info["vector"] = [{"doc": d, "meta": m} for d, m in zip(docs, metas)]
            debug_info["fused"] = debug_info["vector"]
            fused_docs, fused_metas = docs, metas
        else:
            search_type = "hybrid"
            vector_docs, vector_metas = self._vector_search(query, n_results * 3)
            bm25_docs, bm25_metas = self._bm25_search(query, n_results * 3)
            total_candidates = len(vector_docs) + len(bm25_docs)

            fused_docs, fused_metas = self._rrf_fusion(
                list(zip(vector_docs, vector_metas)),
                list(zip(bm25_docs, bm25_metas)),
                n_results=n_results,
            )

            debug_info["vector"] = [{"doc": d, "meta": m} for d, m in zip(vector_docs, vector_metas)]
            debug_info["bm25"] = [{"doc": d, "meta": m} for d, m in zip(bm25_docs, bm25_metas)]
            debug_info["fused"] = [{"doc": d, "meta": m} for d, m in zip(fused_docs, fused_metas)]

            # 可选：使用重排模型对融合结果做二次排序
            if use_reranker:
                fused_docs, fused_metas, rerank_items = self._rerank(
                    query, fused_docs, fused_metas
                )
                debug_info["rerank"] = rerank_items
                reranker_used = len(rerank_items) > 0

        # 构建统一 results 列表：每项为 {"doc", "meta", "score"}
        if rerank_items:
            results = [
                {"doc": r["doc"], "meta": r["meta"], "score": r["score"]}
                for r in rerank_items
            ]
        else:
            results = [
                {"doc": d, "meta": m, "score": None}
                for d, m in zip(fused_docs, fused_metas)
            ]

        if return_metadata:
            scores = [r["score"] for r in results if r["score"] is not None]
            metadata: Dict[str, Any] = {
                "search_type": search_type,
                "reranker_used": reranker_used,
                "total_candidates": total_candidates,
                "final_count": len(results),
                "avg_score": float(sum(scores)) / len(scores) if scores else None,
                "max_score": max(scores) if scores else None,
            }
            return {
                "results": results,
                "metadata": metadata,
                "debug_info": debug_info,
            }

        return fused_docs, fused_metas, debug_info

    # ==================== 内部方法：BM25 & RRF ====================

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        中英文混合分词：
        - 先统一为小写
        - 使用 jieba 做中文切分，同时对英文/数字片段也能较好处理
        """
        lowered = text.lower()
        return [tok.strip() for tok in jieba.cut(lowered) if tok.strip()]

    def _vector_search(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        内部向量检索封装，便于与 BM25 结果做融合。
        实际上调用的就是现有的 self.search。
        """
        return self.search(query, n_results)

    def _bm25_search(self, query: str, n_results: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        仅使用 BM25 的检索。
        """
        if not self._bm25_model or not self._bm25_corpus:
            return [], []

        query_tokens = self._tokenize(query)
        scores = self._bm25_model.get_scores(query_tokens)

        # 根据得分排序，取前 n_results 个索引
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:n_results]

        docs = [self._bm25_corpus[i] for i in ranked_indices]
        metas = [self._bm25_metadatas[i] for i in ranked_indices]
        return docs, metas

    def _load_reranker(self) -> Optional[FlagReranker]:
        """
        懒加载重排模型（FlagEmbedding）。
        只在首次需要时加载，避免占用过多显存/内存。
        """
        if self._reranker is not None:
            return self._reranker

        try:
            # 这里选择 BAAI/bge-reranker-base，你可以根据需要改成 large/m3 等
            self._reranker = FlagReranker(
                "BAAI/bge-reranker-base",
                use_fp16=True,
            )
        except Exception:
            # 加载失败则返回 None，上层逻辑会自动跳过重排
            self._reranker = None

        return self._reranker

    def ensure_reranker_loaded(self) -> bool:
        """
        对外暴露的便捷方法：确保重排模型已加载。
        返回 True 表示可用，False 表示加载失败或未安装依赖。
        """
        return self._load_reranker() is not None

    def _rerank(
        self,
        query: str,
        docs: List[str],
        metas: List[Dict],
    ) -> Tuple[List[str], List[Dict], List[Dict]]:
        """
        使用 FlagEmbedding 对候选文档进行重排。

        Returns:
            rerank_docs, rerank_metas, rerank_items
            其中 rerank_items 中的每一项为:
            {"doc": doc, "meta": meta, "score": score}
        """
        reranker = self._load_reranker()
        if reranker is None or not docs:
            return docs, metas, []

        pairs = [[query, d] for d in docs]
        try:
            scores = reranker.compute_score(pairs)
        except Exception:
            return docs, metas, []

        # 根据 score 从大到小排序
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: float(scores[i]),
            reverse=True,
        )

        rerank_docs: List[str] = []
        rerank_metas: List[Dict] = []
        rerank_items: List[Dict] = []
        for i in ranked_indices:
            d = docs[i]
            m = metas[i]
            s = float(scores[i])
            rerank_docs.append(d)
            rerank_metas.append(m)
            rerank_items.append({"doc": d, "meta": m, "score": s})

        return rerank_docs, rerank_metas, rerank_items

    @staticmethod
    def _rrf_fusion(
        vector_results: List[Tuple[str, Dict]],
        bm25_results: List[Tuple[str, Dict]],
        n_results: int = 3,
        k: int = 60,
    ) -> Tuple[List[str], List[Dict]]:
        """
        RRF（Reciprocal Rank Fusion）结果融合。

        Args:
            vector_results: [(doc, meta), ...]，按向量相似度排序。
            bm25_results:   [(doc, meta), ...]，按 BM25 得分排序。
            n_results: 返回数量。
            k: RRF 参数，默认 60。
        """
        from json import dumps

        def _key(doc: str, meta: Dict) -> str:
            # 通过 文本 + 元数据 的 JSON 串作为 key，近似唯一标识一个 chunk
            return dumps({"doc": doc, "meta": meta}, ensure_ascii=False, sort_keys=True)

        scores: Dict[str, float] = {}
        payloads: Dict[str, Tuple[str, Dict]] = {}

        # 向量结果打分
        for rank, (doc, meta) in enumerate(vector_results, start=1):
            key = _key(doc, meta)
            payloads[key] = (doc, meta)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

        # BM25 结果打分
        for rank, (doc, meta) in enumerate(bm25_results, start=1):
            key = _key(doc, meta)
            payloads[key] = (doc, meta)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

        # 按融合分数排序取前 n 个
        ranked_keys = sorted(scores.keys(), key=lambda kk: scores[kk], reverse=True)[:n_results]

        docs: List[str] = []
        metas: List[Dict] = []
        for kk in ranked_keys:
            doc, meta = payloads[kk]
            docs.append(doc)
            metas.append(meta)

        return docs, metas

    # ==================== Prompt 构建 ====================

    @staticmethod
    def _build_user_prompt(question: str, context: str) -> str:
        """
        根据用户问题和检索到的参考资料构建 User Prompt。

        Args:
            question: 用户问题。
            context: 检索得到的文档片段拼接成的上下文（如用 "\\n\\n---\\n\\n" 分隔）。

        Returns:
            完整的 user 消息内容。
        """
        return (
            "请基于以下参考资料回答用户的问题。\n\n"
            "【参考资料】\n"
            f"{context}\n\n"
            "【用户问题】\n"
            f"{question}\n\n"
            "请基于参考资料给出准确、专业的回答。如果参考资料中没有相关信息，请明确说明。"
        )

    def ask(self, question: str, n_results: int = 3, use_reranker: bool = False) -> Dict:
        """
        RAG 问答 - 核心方法。

        Args:
            question: 用户问题。
            n_results: 检索文档数量。

        Returns:
            包含答案、检索结果等信息的字典。
        """
        # 1. 检索相关文档（使用混合检索：向量 + BM25），并记录检索细节
        retrieved_docs, metadatas, debug_info = self.hybrid_search(
            question,
            n_results,
            use_reranker=use_reranker,
        )

        if not retrieved_docs:
            return {
                "answer": "抱歉，我的知识库中还没有相关信息。请先上传一些文档。",
                "sources": [],
                "retrieved_docs": [],
                "debug": debug_info,
            }

        # 2. 构建上下文与 User Prompt
        context = "\n\n---\n\n".join(retrieved_docs)
        user_prompt = self._build_user_prompt(question, context)

        # 3. 调用 LLM 生成答案（使用 SYSTEM_PROMPT + _build_user_prompt）
        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        # 4. 返回结果
        return {
            "answer": answer,
            "sources": metadatas,
            "retrieved_docs": retrieved_docs,
            "debug": debug_info,
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

