import os
import getpass
from typing import List, Annotated, TypedDict
from typing_extensions import TypedDict

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API Key: ")

class RAGState(TypedDict):
    question: str
    retrieved_docs: List[Document]
    context: str
    answer: str
    messages: List


class KnowledgeBaseQA:
    def __init__(self, docs_path: str = "./documents", collection_name: str = "knowledge_base"):
        self.docs_path = docs_path
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        if os.path.exists(self.docs_path):
            self._load_documents()
        else:
            print(f"Directory {self.docs_path} does not exist. Creating...")
            os.makedirs(self.docs_path, exist_ok=True)
            print(f"Created {self.docs_path}. Please add documents and call load_documents().")

    def _load_documents(self):
        print(f"Loading documents from {self.docs_path}...")
        all_docs = []

        if os.path.exists(self.docs_path):
            for file in os.listdir(self.docs_path):
                file_path = os.path.join(self.docs_path, file)
                if file.endswith('.txt'):
                    loader = TextLoader(file_path)
                elif file.endswith('.pdf'):
                    loader = PDFLoader(file_path)
                else:
                    continue

                docs = loader.load()
                all_docs.extend(docs)
                print(f"Loaded {len(docs)} documents from {file}")

        if all_docs:
            print(f"Splitting {len(all_docs)} documents into chunks...")
            chunks = self.text_splitter.split_documents(all_docs)
            print(f"Created {len(chunks)} chunks")

            print(f"Creating vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                persist_directory="./chroma_db"
            )
            print("Vector store created and persisted")
        else:
            print("No documents found")

    def load_documents(self):
        self._load_documents()

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Please load documents first.")

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(query)
        return docs

    def format_context(self, docs: List[Document]) -> str:
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            context_parts.append(f"[文档{i}](来源: {source})\n{doc.page_content}\n")
        return "\n---\n".join(context_parts)

    def build_graph(self):
        workflow = StateGraph(RAGState)

        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate_answer", self._generate_answer_node)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def _retrieve_node(self, state: RAGState) -> dict:
        question = state["question"]
        docs = self.retrieve(question)
        context = self.format_context(docs)
        return {"retrieved_docs": docs, "context": context}

    def _generate_answer_node(self, state: RAGState) -> dict:
        question = state["question"]
        context = state["context"]

        system_prompt = f"""你是一个专业的知识库问答助手。你的任务是根据提供的上下文信息来回答用户的问题。

重要规则：
1. 只使用提供的上下文信息来回答问题
2. 如果上下文中没有相关信息，请明确说明"根据当前知识库没有找到相关信息"
3. 在回答时引用相关文档来源
4. 回答要清晰、准确、有条理

上下文信息：
{context}
"""

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ])

        messages = [HumanMessage(content=question), response]
        return {"answer": response.content, "messages": messages}

    def ask(self, question: str, thread_id: str = "default") -> dict:
        if not self.vectorstore:
            return {"error": "请先加载文档到知识库"}

        graph = self.build_graph()
        config = {"configurable": {"thread_id": thread_id}}

        result = graph.invoke({"question": question}, config)

        return {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": [
                {"source": doc.metadata.get('source', 'Unknown'), "content": doc.page_content[:200] + "..."}
                for doc in result.get("retrieved_docs", [])
            ]
        }


def create_sample_documents():
    os.makedirs("./documents", exist_ok=True)

    sample_text = """人工智能（AI）基础知识

第一章：机器学习概述

机器学习是人工智能的一个分支，它使计算机能够从数据中学习并改进性能。主要类型包括：
1. 监督学习：使用标注数据进行训练
2. 无监督学习：从无标注数据中发现模式
3. 强化学习：通过奖励机制学习决策策略

第二章：深度学习基础

深度学习使用多层神经网络来学习数据的层次表示。
- 卷积神经网络（CNN）：用于图像处理
- 循环神经网络（RNN）：用于序列数据
- Transformer：用于自然语言处理

第三章：自然语言处理

NLP使计算机能够理解、解释和生成人类语言。
主要任务包括：
- 文本分类
- 命名实体识别
- 机器翻译
- 问答系统

第四章：LangChain框架

LangChain是一个用于构建LLM应用的框架。
核心组件：
- Chains：链接多个组件
- Agents：自主决策和执行
- Memory：维护对话状态
- RAG：检索增强生成
"""

    with open("./documents/ai_intro.txt", "w", encoding="utf-8") as f:
        f.write(sample_text)

    print("Sample document created at ./documents/ai_intro.txt")


if __name__ == "__main__":
    print("=" * 60)
    print("本地知识库问答系统")
    print("=" * 60)

    create_sample_documents()

    qa_system = KnowledgeBaseQA(docs_path="./documents")

    questions = [
        "什么是机器学习？",
        "深度学习有哪些主要类型？",
        "LangChain的核心组件有哪些？",
        "自然语言处理的主要任务是什么？"
    ]

    for i, q in enumerate(questions, 1):
        print(f"\n问题 {i}: {q}")
        print("-" * 40)
        result = qa_system.ask(q, thread_id=f"q{i}")
        print(f"回答: {result['answer']}")
        print(f"参考文档数: {len(result['sources'])}")
        print()

    print("=" * 60)
    print("问答演示完成")
    print("=" * 60)