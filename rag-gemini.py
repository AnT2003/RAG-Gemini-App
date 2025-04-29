import os
import json
import tempfile
import shutil
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
GEMINI_API_KEY = "AIzaSyBUAYGkxXkMxQ_bQ6mqgwqCmEwieBRtD8c"
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

DB_FOLDER = "faiss_index"

st.set_page_config(page_title="RAG FAISS + Gemini", page_icon="🔍", layout="wide")
st.title("🔍 RAG App with Gemini")

# ------------------------ #
# Functions
# ------------------------ #

def load_uploaded_documents(uploaded_files):
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
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embedding_model)
    vectordb.save_local(db_path)
    return vectordb

def load_vectorstore(db_path=DB_FOLDER):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_index_path = os.path.join(db_path, "index.faiss")
    if not os.path.exists(faiss_index_path):
        st.warning(f"Không tìm thấy chỉ mục FAISS tại {faiss_index_path}. Tạo lại chỉ mục từ tài liệu...")
        raise FileNotFoundError(f"FAISS index file '{faiss_index_path}' not found.")
    vectordb = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)
    return vectordb

def create_gemini_llm():
    return GoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)

def create_qa_chain(llm, retriever):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
    )

def get_fallback_answer(llm, query):
    return llm(query)

def delete_documents_from_vectorstore(sources_to_delete, db_path=DB_FOLDER):
    vectordb = load_vectorstore(db_path)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    all_docs = vectordb.similarity_search("dummy query", k=1000)
    remaining_docs = [doc for doc in all_docs if doc.metadata.get("source") not in sources_to_delete]

    if not remaining_docs:
        shutil.rmtree(db_path)
        os.makedirs(db_path)
        return

    new_vectordb = FAISS.from_documents(remaining_docs, embedding_model)
    new_vectordb.save_local(db_path)

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    try:
        vectordb = load_vectorstore()
        llm = create_gemini_llm()
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        st.session_state.qa_chain = create_qa_chain(llm, retriever)
    except FileNotFoundError:
        st.warning("Chỉ mục FAISS chưa được tạo. Vui lòng tải tài liệu và tạo chỉ mục trước!")

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
    uploaded_files = st.file_uploader("Chọn nhiều file tài liệu", type=["pdf", "txt", "docx", "md"], accept_multiple_files=True)

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

    try:
        vectordb = load_vectorstore()
        all_docs = vectordb.similarity_search("dummy query", k=1000)
        unique_sources = sorted(list(set(doc.metadata.get("source", "") for doc in all_docs)))

        selected_sources = st.multiselect("📚 Chọn tài liệu bạn muốn search (không chọn = tất cả):", unique_sources)
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

                    qa_chain = create_qa_chain(create_gemini_llm(), retriever)
                    result = qa_chain.invoke({"question": query})

                    if result.get("answer"):
                        st.markdown("### 💬 Câu trả lời:")
                        st.write(result["answer"])

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
                        st.markdown("### 💬 Câu trả lời từ Gemini:")
                        answer = get_fallback_answer(create_gemini_llm(), query)
                        st.write(answer)
                        st.session_state.chat_history.append({
                            "question": query,
                            "answer": answer,
                            "sources": []
                        })
            else:
                st.warning("⚠️ Bạn chưa nhập câu hỏi!")

    except FileNotFoundError as e:
        st.warning(f"Lỗi: {e}")

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
    vectordb = load_vectorstore()
    all_docs = vectordb.similarity_search("dummy query", k=1000)
    document_sources = sorted(list(set(doc.metadata.get("source", "") for doc in all_docs)))

    selected_documents = st.multiselect("📚 Chọn tài liệu bạn muốn xóa:", document_sources)

    if st.button("❌ Xóa tài liệu"):
        if selected_documents:
            with st.spinner("Đang xóa tài liệu..."):
                delete_documents_from_vectorstore(selected_documents)
                st.success("✅ Tài liệu đã xóa khỏi hệ thống!")
        else:
            st.warning("⚠️ Bạn chưa chọn tài liệu nào để xóa!")
