# Azure Files RAG Pipeline — LangChain + Weaviate

This sample implements a Retrieval-Augmented Generation (RAG) pipeline that ingests documents from an [Azure file share](https://learn.microsoft.com/azure/storage/files/storage-files-introduction), indexes them into [Weaviate](https://weaviate.io/), and provides an interactive Q&A session powered by [Azure OpenAI](https://learn.microsoft.com/azure/ai-services/openai/overview) and [LangChain](https://www.langchain.com/).

## Prerequisites

- **Python 3.12+** — Windows users must use the **x64** version of Python (not x86/32-bit). You can verify with `python -c "import struct; print(struct.calcsize('P') * 8)"` which should print `64`.
- An **Azure subscription** with:
  - An [Azure Storage account](https://learn.microsoft.com/azure/storage/common/storage-account-create) with an Azure file share containing documents to index.
  - An [Azure OpenAI resource](https://learn.microsoft.com/azure/ai-services/openai/how-to/create-resource) with an embedding model (e.g., `text-embedding-3-small`) and a chat model (e.g., `gpt-4o-mini`) deployed.
- A [Weaviate Cloud](https://console.weaviate.cloud/) cluster and API key.
- [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) installed and signed in (`az login`).

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/ftrichardson1/azure-files-langchain-weaviate.git
cd azure-files-langchain-weaviate
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```powershell
py -3.12 -c "import struct; print(struct.calcsize('P') * 8)"   # Confirm "64"
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the sample environment file and open it in your editor:

```bash
cp .env.sample .env
code .env
```

| Variable | Description |
|---|---|
| `AZURE_STORAGE_ACCOUNT_NAME` | Name of your Azure Storage account |
| `AZURE_STORAGE_SHARE_NAME` | Name of the Azure file share with your documents |
| `AZURE_OPENAI_ENDPOINT` | Endpoint URL of your Azure OpenAI resource |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | Deployment name for your embedding model |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | Deployment name for your chat model |
| `WEAVIATE_URL` | REST endpoint URL from your Weaviate Cloud console |
| `WEAVIATE_API_KEY` | Your Weaviate API key |
| `WEAVIATE_COLLECTION_NAME` | Name of the Weaviate collection to create/use (default: `AzureFilesRAG`) |

### 5. Run the pipeline

```bash
python langchain-weaviate.py
```

The script:

1. Connects to your Azure file share using `DefaultAzureCredential`.
2. Downloads and parses all documents (PDF, DOCX, CSV, TXT, and more).
3. Splits documents into chunks for embedding.
4. Embeds chunks with Azure OpenAI and indexes them into Weaviate.
5. Starts an interactive Q&A session — ask questions about your documents.

Type `quit` to exit the Q&A session.

## Project structure

| File | Description |
|---|---|
| `langchain-weaviate.py` | Main RAG pipeline script |
| `azure_files.py` | Azure Files helper — connects, lists, and downloads files from a share |
| `config.py` | Loads environment variables and sets up Azure credentials |
| `requirements.txt` | Python dependencies |
| `.env.sample` | Template for required environment variables |

## Related resources

- [Azure Files for AI — RAG tutorial (LangChain + Weaviate)](https://learn.microsoft.com/azure/storage/files/artificial-intelligence/retrieval-augmented-generation/open-source-frameworks/tutorials/langchain-weaviate/tutorial-langchain-weaviate)
- [Azure OpenAI documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [Weaviate documentation](https://weaviate.io/developers/weaviate)
- [LangChain documentation](https://python.langchain.com/)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
