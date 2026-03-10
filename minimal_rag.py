from openai import OpenAI
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions

# 加载环境变量
load_dotenv()

# DeepSeek 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 初始化 ChromaDB（使用默认 Embedding）
print("正在初始化 ChromaDB...")
chroma_client = chromadb.Client()

# 使用 ChromaDB 默认的 Embedding 函数
default_ef = embedding_functions.DefaultEmbeddingFunction()

# 创建 Collection（指定 embedding_function）
collection = chroma_client.create_collection(
    name="wyckoff_knowledge",
    embedding_function=default_ef
)
print("✅ ChromaDB 初始化完成")

# 知识库
knowledge_base = [
    "威科夫理论由理查德·威科夫在20世纪初提出，是一套完整的技术分析方法。",
    "威科夫市场循环包括四个阶段：积累阶段、上涨阶段、派发阶段、下跌阶段。",
    "积累阶段是大资金在底部区域悄悄买入的过程，价格横盘震荡。",
    "上涨阶段是价格突破积累区后的趋势性上涨，成交量放大。",
    "派发阶段是大资金在顶部区域悄悄卖出的过程，价格再次横盘。",
    "下跌阶段是价格跌破派发区后的趋势性下跌。"
]

# 存储知识（ChromaDB 自动处理 Embedding）
print("\n正在存储知识库...")
collection.add(
    ids=[f"doc_{i}" for i in range(len(knowledge_base))],
    documents=knowledge_base  # 直接传文本，ChromaDB 自动转向量
)
print(f"✅ 已存储 {len(knowledge_base)} 条知识")

# 检索函数
def search_knowledge(query, top_k=2):
    """
    根据问题检索最相关的知识
    """
    results = collection.query(
        query_texts=[query],  # 直接传文本，ChromaDB 自动转向量
        n_results=top_k
    )
    return results['documents'][0]

# RAG 问答函数
def rag_query(question):
    """
    RAG 完整流程：检索 + 生成
    """
    print(f"\n🔍 问题: {question}")
    
    # 1. 检索
    relevant_docs = search_knowledge(question, top_k=2)
    
    print("\n📚 检索到的相关知识:")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"  {i}. {doc}")
    
    # 2. 构造提示词
    context = "\n".join(relevant_docs)
    prompt = f"""请根据以下参考资料回答问题。如果参考资料中没有相关信息，请说"我不知道"。

参考资料：
{context}

问题：{question}

回答："""
    
    # 3. 调用 LLM 生成答案
    print("\n🤖 生成回答中...")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    print(f"\n✅ 回答: {answer}")
    
    return answer

# 测试
if __name__ == "__main__":
    questions = [
        "威科夫的四个阶段是什么？",
        "什么是积累阶段？",
        "区块链技术是什么？"  # 故意问一个不相关的
    ]
    
    for q in questions:
        rag_query(q)
        print("\n" + "="*60 + "\n")