import os
import json
import tempfile
import streamlit as st

from langchain_community.document_loaders import (
    TextLoader, PDFMinerLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.documents import Document

# ------------------------ #
# Configs
# ------------------------ #
GEMINI_API_KEY = "AIzaSyBUAYGkxXkMxQ_bQ6mqgwqCmEwieBRtD8c"  # <-- Thay bằng API KEY của bạn
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

DB_FOLDER = "faiss_index"

st.set_page_config(page_title="RAG FAISS + Gemini", page_icon="🔍", layout="wide")
st.title("🔍 RAG App with Gemini")

# ------------------------ #
# Functions
# ------------------------ #

def load_uploaded_documents(uploaded_files):
    """Load nhiều tài liệu upload"""
    documents = []
    loaders = {
        ".txt": TextLoader,
        ".pdf": PDFMinerLoader,
        ".docx": UnstructuredWordDocumentLoader,
        ".md": UnstructuredMarkdownLoader
    }
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[-1].lower()
        if suffix in loaders:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                loader = loaders[suffix](tmp_file.name)
                file_docs = loader.load()
                for doc in file_docs:
                    doc.metadata["source"] = uploaded_file.name
                documents.extend(file_docs)
    return documents

def create_vectorstore(docs, db_path=DB_FOLDER):
    """Tạo FAISS database"""
    embedding_model = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")
    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(db_path)
    return vectordb

def load_vectorstore(db_path=DB_FOLDER):
    """Load FAISS database"""
    embedding_model = HuggingFaceEmbeddings(model_name="Snowflake/snowflake-arctic-embed-l-v2.0")
    vectordb = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return vectordb

def create_gemini_llm():
    """Tạo Gemini model"""
    return GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def create_qa_chain(llm, retriever):
    """Tạo QA chain"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"  # Chỉ định rõ ràng 'output_key' là 'answer'
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

def get_fallback_answer(llm, query):
    """Trả lời fallback khi không có tài liệu liên quan"""
    return llm(query)

def delete_documents_from_vectorstore(doc_ids_to_delete, db_path=DB_FOLDER):
    """Xóa tài liệu khỏi FAISS database"""
    vectordb = load_vectorstore(db_path)
    vectordb.delete_documents(doc_ids_to_delete)
    vectordb.save_local(db_path)


if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    vectordb = load_vectorstore()
    llm = create_gemini_llm()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    st.session_state.qa_chain = create_qa_chain(llm, retriever)

# ------------------------ #
# Sidebar
# ------------------------ #
st.sidebar.header("Tùy chọn")
app_mode = st.sidebar.radio("Chọn chế độ:", ["📂 Upload Documents", "💬 Hỏi đáp", "📜 Lịch sử hỏi đáp", "🗑️ Xóa tài liệu"])

# ------------------------ #
# Main App
# ------------------------ #
if app_mode == "📂 Upload Documents":
    st.subheader("📂 Upload tài liệu mới vào hệ thống")

    uploaded_files = st.file_uploader(
        "Chọn nhiều file tài liệu",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True
    )

    if st.button("🔄 Xử lý và lưu vào DB"):
        if uploaded_files:
            with st.spinner("Đang xử lý tài liệu..."):
                docs = load_uploaded_documents(uploaded_files)
                splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                split_docs = splitter.split_documents(docs)
                create_vectorstore(split_docs, db_path=DB_FOLDER)
            st.success("✅ Tài liệu đã xử lý và lưu thành công!")
        else:
            st.warning("⚠️ Bạn chưa upload tài liệu!")

elif app_mode == "💬 Hỏi đáp":
    st.subheader("💬 Đặt câu hỏi")

    vectordb = load_vectorstore()

    # Lấy danh sách tất cả tài liệu
    all_docs = vectordb.similarity_search("dummy query", k=1000)
    unique_sources = sorted(list(set(doc.metadata.get("source", "") for doc in all_docs)))

    selected_sources = st.multiselect(
        "📚 Chọn tài liệu bạn muốn search (không chọn = tìm trong tất cả tài liệu):", 
        unique_sources
    )

    query = st.text_input("💬 Nhập câu hỏi tại đây:", key="query_input")

    col1, col2 = st.columns([2, 1])
    with col1:
        search_button = st.button("🚀 Tìm câu trả lời", use_container_width=True)
    with col2:
        reset_button = st.button("♻️ Reset hội thoại", use_container_width=True)

    if reset_button:
        st.session_state.qa_chain.memory.clear()
        st.session_state.chat_history.clear()
        st.success("✅ Đã reset hội thoại!")

    # Điều kiện thực hiện tìm kiếm chỉ khi nhấn nút hoặc Enter
    if search_button or st.session_state.get('search_triggered', False):
        if query.strip():
            with st.spinner("🤖 Đang truy vấn..."):
                if selected_sources:
                    retriever = vectordb.as_retriever(
                        search_type="similarity",
                        search_kwargs={
                            "k": 3,
                            "filter": lambda d: d.metadata.get("source", "") in selected_sources
                        }
                    )
                else:
                    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

                # Tạo chain mới tạm thời cho câu hỏi này
                qa_chain = create_qa_chain(create_gemini_llm(), retriever)
                result = qa_chain.invoke({"question": query})

                if result.get("answer"):
                    # Nếu có câu trả lời từ tài liệu
                    st.markdown("### 💬 Câu trả lời:")
                    st.write(result["answer"])

                    # Lưu lịch sử
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": result["answer"],
                        "sources": result["source_documents"]
                    })

                    st.markdown("### 📚 Tài liệu nguồn:")
                    for idx, doc in enumerate(result["source_documents"], 1):
                        st.write(f"**{idx}. Từ tài liệu:** `{doc.metadata.get('source', 'Không rõ tài liệu')}`")
                        with st.expander("Xem nội dung đoạn text"):
                            st.write(doc.page_content)
                else:
                    # Nếu không có câu trả lời từ tài liệu, trả lời từ mô hình
                    st.markdown("### 💬 Câu trả lời từ Gemini:")
                    answer = get_fallback_answer(create_gemini_llm(), query)
                    st.write(answer)
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer,
                        "sources": []  # Không có nguồn tài liệu
                    })

        else:
            st.warning("⚠️ Bạn chưa nhập câu hỏi!")

elif app_mode == "📜 Lịch sử hỏi đáp":
    st.subheader("📜 Xem lại lịch sử hỏi đáp")

    if st.session_state.chat_history:  
        for idx, chat in enumerate(st.session_state.chat_history[::-1], 1):
            st.markdown(f"### {idx}. **Câu hỏi:** {chat['question']}")
            st.markdown(f"**Trả lời:** {chat['answer']}")
            st.markdown("**Nguồn tài liệu:**")
            for doc in chat['sources']:
                st.write(f"- {doc.metadata.get('source', 'Unknown')}")
            st.divider()
    else:
        st.info("⚡️ Chưa có lịch sử hỏi đáp nào.")

elif app_mode == "🗑️ Xóa tài liệu":
    st.subheader("🗑️ Xóa tài liệu khỏi hệ thống")

    # Load vectorstore và lấy danh sách tài liệu
    vectordb = load_vectorstore()

    # Lấy danh sách tất cả tài liệu
    all_docs = vectordb.similarity_search("dummy query", k=1000)
    document_sources = sorted(list(set(doc.metadata.get("source", "") for doc in all_docs)))

    # Cho phép người dùng chọn tài liệu để xóa
    selected_documents = st.multiselect(
        "📚 Chọn tài liệu bạn muốn xóa:", 
        document_sources
    )

    if st.button("❌ Xóa tài liệu"):
        if selected_documents:
            with st.spinner("Đang xóa tài liệu..."):
                # Tạo danh sách các tài liệu cần xóa
                doc_ids_to_delete = [doc.metadata["source"] for doc in all_docs if doc.metadata.get("source") in selected_documents]
                delete_documents_from_vectorstore(doc_ids_to_delete)
                st.success("✅ Tài liệu đã xóa khỏi hệ thống!")
        else:
            st.warning("⚠️ Bạn chưa chọn tài liệu nào để xóa!")
