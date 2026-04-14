"""RAG pipeline using LangChain, Weaviate, and Azure Files.

Ingests documents from an Azure file share, converts Azure Files documents
to LangChain Documents, indexes into Weaviate, and provides an interactive
Q&A loop.
"""

import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

import weaviate
from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate.vectorstores import WeaviateVectorStore
from weaviate.classes.init import Auth

from azure_files import DownloadedFile, connect_to_share, download_files, list_share_files
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CREDENTIAL,
    EMBEDDING_DIMENSIONS,
    OPENAI_CHAT_DEPLOYMENT,
    OPENAI_EMBEDDING_DEPLOYMENT,
    OPENAI_ENDPOINT,
    SHARE_NAME,
    STORAGE_ACCOUNT_NAME,
    TOKEN_PROVIDER,
)

# REST endpoint from the Weaviate Cloud console (not the gRPC endpoint).
# Example: https://abc123.c0.us-east1.gcp.weaviate.cloud
WEAVIATE_URL = os.environ["WEAVIATE_URL"]
WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
WEAVIATE_COLLECTION_NAME = os.getenv("WEAVIATE_COLLECTION_NAME", "AzureFilesRAG")

# Mapping of file extensions to LangChain document loaders and their kwargs
LOADER_MAP: dict[str, tuple] = {
    ".pdf": (PyPDFLoader, {}),
    ".docx": (Docx2txtLoader, {}),
    ".csv": (CSVLoader, {}),
    ".tsv": (CSVLoader, {"csv_args": {"delimiter": "\t"}}),
}

DEFAULT_LOADER = (TextLoader, {"encoding": "utf-8"})


def parse_downloaded_files(
    downloaded_files: list[DownloadedFile],
) -> list[Document]:
    """Parse downloaded files from an Azure file share into LangChain Documents.

    Args:
        downloaded_files: A list of DownloadedFile objects, each representing a
            file in an Azure file share, containing the path and access control
            metadata for a file.

    Returns: A list of LangChain Documents.
    """
    documents = []

    for info in downloaded_files:
        file_ext = os.path.splitext(info.file_name.lower())[1]
        loader_cls, kwargs = LOADER_MAP.get(file_ext, DEFAULT_LOADER)

        try:
            docs = loader_cls(info.local_path, **kwargs).load()
        except Exception:
            print(f"Failed to parse {info.relative_path}, skipping...")
            continue

        for doc in docs:
            doc.metadata.update({
                "azure_file_path": info.relative_path,
                "file_name": info.file_name,
            })
        documents.extend(docs)

    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks for embedding.

    Args:
        documents: A list of LangChain Documents to split.

    Returns: A list of smaller Document chunks with preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


def embed_and_index(
    chunks: list[Document],
) -> tuple[weaviate.WeaviateClient, WeaviateVectorStore]:
    """Embed document chunks via Azure OpenAI and upsert into Weaviate.

    Args:
        chunks: A list of chunked LangChain Documents to embed and index.

    Returns:
        A tuple of (Weaviate client, WeaviateVectorStore). The caller is
        responsible for closing the Weaviate client.
    """
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

    # Reset collection if specified via environment variable. This is useful
    # for development to ensure a clean slate on each run. In production, you
    # would typically not reset the collection.
    if os.getenv("RESET_INDEX") == "true":
        if client.collections.exists(WEAVIATE_COLLECTION_NAME):
            client.collections.delete(WEAVIATE_COLLECTION_NAME)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_EMBEDDING_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        dimensions=EMBEDDING_DIMENSIONS,
    )

    store = WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=WEAVIATE_COLLECTION_NAME,
    )

    return client, store


def build_qa_chain(vector_store: WeaviateVectorStore):
    """Build a retrieval question-answering (Q&A) chain.

    Args:
        vector_store: WeaviateVectorStore to retrieve from.
    """
    llm = AzureChatOpenAI(
        azure_endpoint=OPENAI_ENDPOINT,
        azure_deployment=OPENAI_CHAT_DEPLOYMENT,
        azure_ad_token_provider=TOKEN_PROVIDER,
        api_version="2024-12-01-preview",
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    prompt = PromptTemplate.from_template(
        "Answer the question based on the context below. "
        "Be specific and cite the source file name in brackets for each fact.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\nAnswer:"
    )

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"[{d.metadata.get('azure_file_path', '')}]\n{d.page_content}"
            for d in docs
        )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    """Main execution flow."""
    share = connect_to_share(STORAGE_ACCOUNT_NAME, SHARE_NAME, CREDENTIAL)

    # 1. List files from the share
    print("Scanning file share...")
    file_references = list_share_files(share)
    if not file_references:
        print("No files found.")
        return
    print(f"Found {len(file_references)} files.\n")

    # 2. Download files (shared Azure Files logic)
    print("Downloading files onto temporary local directory...")
    with tempfile.TemporaryDirectory() as temp_directory:
        downloaded = download_files(file_references, temp_directory)
        if not downloaded:
            print("No files downloaded.")
            return
        print()

        # 3. Parse into LangChain Documents
        print("Parsing files...")
        documents = parse_downloaded_files(downloaded)

    if not documents:
        print("No documents parsed.")
        return
    print(f"{len(documents)} documents.\n")

    # 4. Chunk
    print("Splitting into chunks...")
    chunks = chunk_documents(documents)
    print(f"{len(documents)} docs → {len(chunks)} chunks.\n")

    # 5. Embed and index
    print("Indexing into Weaviate...")
    weaviate_client, store = embed_and_index(chunks)
    print(f"{len(chunks)} chunks indexed.\n")

    qa_chain = build_qa_chain(store)
    print("Ready. Type 'quit' to exit.\n")

    try:
        while True:
            question = input("You: ").strip()
            if question.lower() in ("quit", "exit", "q"):
                break
            if not question:
                continue
            print(f"\nAnswer: {qa_chain.invoke(question)}\n")
    except KeyboardInterrupt:
        pass
    finally:
        weaviate_client.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
