import asyncio
import base64
import json
import os
import sys
import tempfile
import time
from asyncio import gather
from enum import Enum
from urllib.parse import urlparse

import chromadb.utils.embedding_functions as embedding_functions
import numpy as np
import pymupdf
import requests
from chromadb.config import Settings
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from pydantic import BaseModel

import chromadb

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Financial_Statements_2023.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Hada_Labo_Gokujun_Lotion_Overview.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Shibata_et_al_Research_Article.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/V_Rohto_Premium_Product_Information.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/617b10e9-a71b-4f2a-a9ee-ffe11d8d64ae/dataset/Well-Being_Report_2024.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================


class DocumentCategory(Enum):
    COMPANY_INFORMATION = "company_information"  # 会社情報
    PRODUCT_INFORMATION = "product_information"  # 製品情報
    RESEARCH_INFORMATION = "research_information"  # 研究情報


class ClassifyResult(BaseModel):
    category: DocumentCategory
    is_valid: bool


class DocumentInfo(BaseModel):
    path: str
    source: str
    collection_name: str
    is_valid: bool


class PageDocument(BaseModel):
    source: str
    page: int
    text: str
    distance: float


class QueryArgs(BaseModel):
    collection_name: str
    vector_query_text: str
    query_keyword: str = None
    not_contain_word: str = None
    source: str = None


IMAGE_DIR = "page_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

VAR_DIR = "global_variables"
os.makedirs(VAR_DIR, exist_ok=True)

CHROMA_DIR = "chromadb"

DEFAULT_SOURCE_LIST = {
    "COMPANY_INFORMATION": [],
    "PRODUCT_INFORMATION": [],
    "RESEARCH_INFORMATION": [],
}


def save_source_list_str(data: str):
    path = os.path.join(VAR_DIR, "source_list_str.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def load_source_list_str() -> str | None:
    path = os.path.join(VAR_DIR, "source_list_str.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_document_infos(data: list[DocumentInfo]):
    path = os.path.join(VAR_DIR, "document_infos.json")
    with open(path, "w") as f:
        json.dump([item.model_dump() for item in data], f, indent=4)


def load_document_infos() -> list[DocumentInfo] | None:
    path = os.path.join(VAR_DIR, "document_infos.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return [DocumentInfo(**item) for item in json.load(f)]


def reset_directory(directory: str):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        os.remove(file_path)


def set_global_variables() -> tuple[dict, str]:
    loaded_document_infos = load_document_infos()
    document_infos = loaded_document_infos if loaded_document_infos else []

    loaded_source_list_str = load_source_list_str()
    source_list_str = loaded_source_list_str if loaded_source_list_str else ""

    return document_infos, source_list_str


COLLECTION_MAP = {}
IMAGE_COLLECTION_MAP = {}

REPHRASE_PROMPT = """
    Compose Args for hybrid search, this function performs a hybrid search by combining vector similarity and keyword-based filtering
    to retrieve relevant documents from the specified collection.

    Args:
        collection_name (str):
            The name of the document collection to search within.
            Must be one of the categories defined in the enum: ["COMPANY_INFORMATION", "PRODUCT_INFORMATION", "RESEARCH_INFORMATION"],
        vector_query_text (str):
            The query text used for vector-based similarity search. almost same as original query.
        query_keyword (str, optional):
            A keyword to filter the search results. Only documents containing this keyword will be considered.　Like proper noun.
            Defaults to None.
        not_contain_word (str, optional):
            A word to exclude from the search results. Documents containing this word will be filtered out. Like proper noun.
            Defaults to None.
        source (str, optional):
            A specific source to filter the documents. Only documents from this source will be retrieved.
            Defaults to None.
        top_k (int, optional):
            The number of top results to retrieve based on the search criteria.
            Defaults to 10.

    Notes:
        `query_keyword` and `not_contain_word` can not be used together.
        `query_keyword` does not contain space, comma, or period.
        If `query_keyword` is provided, only results containing this word are returned.
        If there are no result from search with `query_keyword`, you should set `query_keyword` None to allow a broader search.
        If `not_contain_word` is provided, results containing this word are excluded.

    Example:
        ストックオプションの有無を教えてください。
          > hybrid_search("COMPANY_INFORMATION", "ストックオプションの有無", query_keyword="ストックオプション")
        どういう製品がありますか？
          > hybrid_search("PRODUCT_INFORMATION", "製品一覧", source="〇〇社製品パンフレット")

    {}
            """


def download_file(url: str):
    parsed_url = urlparse(url)
    file_name = os.path.basename(parsed_url.path)
    response = requests.get(url, stream=True, timeout=10)
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, file_name)
    with open(file_path, "wb") as f:
        f.write(response.content)
    return file_path


def extract_10page(pdf_path: str):
    doc = pymupdf.open(pdf_path)

    num_pages = len(doc) if len(doc) < 10 else 10
    text = ""
    for num_page in range(num_pages):
        text += doc[num_page].get_text()

    return text


async def classify_pdf(pdf_path: str) -> ClassifyResult:
    text = extract_10page(pdf_path)
    user_prompt = f"What is the document classification? Is this text extracted from PDF valid? If no text, False.\nsource: {pdf_path}\ncontent: {text}"
    llm = ChatOpenAI(model=model, temperature=0.3)
    structured_llm = llm.with_structured_output(ClassifyResult)
    return await structured_llm.ainvoke(user_prompt)


# Classify PDF
async def classify_and_list_pdf(path: str):
    classify_result = await classify_pdf(path)

    file_name = os.path.basename(path)
    source = file_name.split(".")[0]
    return DocumentInfo(
        path=path,
        source=source,
        collection_name=classify_result.category.name,
        is_valid=classify_result.is_valid,
    )


async def classify_pdfs(paths: list[str]):
    tasks = [classify_and_list_pdf(path) for path in paths]
    return await asyncio.gather(*tasks)


def is_valid_source_collection(sources: list[DocumentInfo]):
    return all(source.is_valid for source in sources)


def update_source_list(document_infos: list[DocumentInfo]):
    source_list = DEFAULT_SOURCE_LIST
    for info in document_infos:
        source_list[info.collection_name].append(info)
    return source_list


def update_source_list_str(source_list: dict) -> str:
    source_list_str = ""
    for collection_name in source_list.keys():
        sources = []
        for item in source_list[collection_name]:
            sources.append(item.source)
        source_list_str += f"""
        **Collection** : {collection_name}
        Sources : {", ".join(sources)}
        """
    return source_list_str


# Create Chroma Collection
# Create MultiModal Collection  # 律速
async def create_multimodal_collection(client, collection_name: str):
    collection = client.get_or_create_collection(
        name=collection_name + "_IMAGE",
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader(),
    )
    return collection


# Create Collection
def create_collection_map(client, source_list: dict):
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]
    )

    if source_list["COMPANY_INFORMATION"]:
        COLLECTION_MAP["COMPANY_INFORMATION"] = client.get_or_create_collection(
            name="COMPANY_INFORMATION", embedding_function=emb_fn
        )
    if source_list["PRODUCT_INFORMATION"]:
        COLLECTION_MAP["PRODUCT_INFORMATION"] = client.get_or_create_collection(
            name="PRODUCT_INFORMATION", embedding_function=emb_fn
        )
    if source_list["RESEARCH_INFORMATION"]:
        COLLECTION_MAP["RESEARCH_INFORMATION"] = client.get_or_create_collection(
            name="RESEARCH_INFORMATION", embedding_function=emb_fn
        )


async def create_image_collection_map(client, source_list: dict):
    if not is_valid_source_collection(source_list["COMPANY_INFORMATION"]):
        company_image_collection = await create_multimodal_collection(client, "COMPANY_INFORMATION")
        IMAGE_COLLECTION_MAP["COMPANY_INFORMATION"] = company_image_collection
    if not is_valid_source_collection(source_list["PRODUCT_INFORMATION"]):
        product_image_collection = await create_multimodal_collection(client, "PRODUCT_INFORMATION")
        IMAGE_COLLECTION_MAP["PRODUCT_INFORMATION"] = product_image_collection
    if not is_valid_source_collection(source_list["RESEARCH_INFORMATION"]):
        research_image_collection = await create_multimodal_collection(
            client, "RESEARCH_INFORMATION"
        )
        IMAGE_COLLECTION_MAP["RESEARCH_INFORMATION"] = research_image_collection


# Extract PDF
async def save_page_image(
    page: pymupdf.Page, num_page: int, source: str, dir_name: str
) -> tuple[str, np.ndarray]:
    zoom_factor = 2.0
    mat = pymupdf.Matrix(zoom_factor, zoom_factor)  # Set zoom matrix
    pix = page.get_pixmap(matrix=mat)  # Apply zoom

    image_id = f"{source}_{num_page + 1}_image"
    page_image_path = os.path.join(dir_name, f"{image_id}.png")
    await asyncio.to_thread(pix.save, page_image_path)

    image_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.h, pix.w, pix.n
    )  # np.array(img)と一致を確認済み
    return image_id, image_data


async def add_pdf_image_data(
    doc: pymupdf.Document, source: str, collection_name: str, batch_size: int = 10
):
    image_collection = IMAGE_COLLECTION_MAP[collection_name]

    async def process_batch(batch):
        ids = []
        images = []

        for num_page, page in batch:
            page_id, image_data = await save_page_image(page, num_page, source, IMAGE_DIR)
            ids.append(page_id)
            images.append(image_data)

        await asyncio.to_thread(image_collection.add, ids=ids, images=images)

    batch = []
    for num_page, page in enumerate(doc):
        batch.append((num_page, page))
        if len(batch) == batch_size:
            await process_batch(batch)
            batch = []

    if batch:
        await process_batch(batch)


async def add_pdf_text_data(
    doc: pymupdf.Document, source: str, collection_name: str, batch_size: int = 10
):
    collection = COLLECTION_MAP[collection_name]

    async def extract_page_text(num_page, page):
        text = page.get_text()
        if not text:
            text = "No text content detected in the PDF. The document might consist of images"
        return {
            "id": f"{source}_{num_page + 1}",
            "text": text,
            "metadata": {"source": source, "page": num_page + 1},
        }

    async def process_batch(batch):
        ids, texts, metadatas = [], [], []

        for num_page, page in batch:
            page_data = await extract_page_text(num_page, page)
            ids.append(page_data["id"])
            texts.append(page_data["text"])
            metadatas.append(page_data["metadata"])

        await asyncio.to_thread(collection.add, documents=texts, metadatas=metadatas, ids=ids)

    batch = []
    for num_page, page in enumerate(doc):
        batch.append((num_page, page))

        if len(batch) == batch_size:
            await process_batch(batch)
            batch = []

    if batch:
        await process_batch(batch)


async def load_pdf(info: DocumentInfo):
    pdf_path = info.path
    source = info.source
    collection_name = info.collection_name

    file_name = os.path.basename(pdf_path)
    source = file_name.split(".")[0]

    with pymupdf.open(pdf_path) as doc:
        if not info.is_valid:
            await add_pdf_image_data(doc, source, collection_name)
        await add_pdf_text_data(doc, source, collection_name)


async def build_collection():
    paths = [download_file(url) for url in pdf_file_urls]
    has_client_data = False

    # load stored settings
    document_infos, source_list_str = set_global_variables()
    if document_infos and source_list_str:
        source_list = update_source_list(document_infos)

        # PDFのソースが変更された場合、ChromaDBをリセット
        past_sources = [info.source for info in document_infos]
        current_sources = [os.path.basename(path).split(".")[0] for path in paths]

        if not sorted(past_sources) == sorted(current_sources):
            client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
            client.reset()
            reset_directory(IMAGE_DIR)
            reset_directory(VAR_DIR)
            source_list = DEFAULT_SOURCE_LIST
            source_list_str = ""
        else:
            has_client_data = True

    if has_client_data:
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
        create_collection_map(client, source_list)
        await create_image_collection_map(client, source_list)

    else:
        # classify pdf
        document_infos = await classify_pdfs(paths)
        source_list = update_source_list(document_infos)
        source_list_str = update_source_list_str(source_list)
        save_document_infos(document_infos)
        save_source_list_str(source_list_str)

        # create collection
        client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=True))
        create_collection_map(client, source_list)
        await create_image_collection_map(client, source_list)

        # add pdf data to chroma
        tasks = [load_pdf(info) for info in document_infos]
        await gather(*tasks)
    return source_list_str


# Query


def encode_image(image_path):
    dirname = IMAGE_DIR
    path = os.path.join(dirname, image_path)

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def chat_image(client, text: str, image_paths: list[str]) -> str:
    messages = [{"role": "user", "content": [{"type": "text", "text": text}]}]

    for image_path in image_paths:
        base64_image = encode_image(image_path)
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        messages[0]["content"].append(image_message)

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
        max_tokens=10000,
    )
    return response.choices[0].message.content


def query_image_collection(collection_name: str, query_text: str, top_k=4):
    image_collection = IMAGE_COLLECTION_MAP[collection_name]
    image_results = image_collection.query(query_texts=[query_text], n_results=top_k)
    image_paths = image_results.get("ids")[0]
    return [image_path + ".png" for image_path in image_paths]


def hybrid_search(
    collection_name: str,
    vector_query_text: str,
    query_keyword: str = None,
    not_contain_word: str = None,
    source: str = None,
    top_k=18,
) -> str:
    if query_keyword:
        search_dict = {"$contains": query_keyword}  # 一語のみ
    elif not_contain_word:
        search_dict = {"$not_contains": not_contain_word}
    else:
        search_dict = None

    metadata_filter = {"source": {"$eq": source}} if source else None

    collection = COLLECTION_MAP[collection_name]
    results = collection.query(
        query_texts=[vector_query_text],
        n_results=top_k,
        where=metadata_filter,  # sourceで絞込
        where_document=search_dict,
    )
    if not results:
        return []

    documents = [
        PageDocument(
            source=results["metadatas"][0][i]["source"],
            page=results["metadatas"][0][i]["page"],
            text=results["documents"][0][i],
            distance=results["distances"][0][i],
        )
        for i in range(len(results["documents"][0]))
    ]
    documents = list(filter(lambda x: x.distance < 1.48, documents))

    text = ""
    for page in documents:
        text += f"# {page.source} Page {page.page}: \n{page.text}\n\n"
    return text


def rephrase_query(query: str, source_list_str: str) -> QueryArgs:
    rephrase_prompt = REPHRASE_PROMPT.format(source_list_str)
    llm = ChatOpenAI(model=model, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rephrase_prompt),
            ("human", "{text}"),
        ]
    )

    structured_llm = prompt | llm.with_structured_output(QueryArgs)
    return structured_llm.invoke(query)


def chat(client, text: str, retrieval_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"あなたはロートの社員です。検索情報をもとに、質問に答えてください。{retrieval_text}",
        },
        {"role": "user", "content": text},
    ]
    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
        max_tokens=10000,
    )
    return response.choices[0].message.content


def rag_implementation(question: str) -> str:
    """
    ロート製薬の製品・企業情報に関する質問に対して回答を生成する関数
    この関数は与えられた質問に対してRAGパイプラインを用いて回答を生成します。

    Args:
        question (str): ロート製薬の製品・企業情報に関する質問文字列

    Returns:
        answer (str): 質問に対する回答

    Note:
        - デバッグ出力は標準出力に出力しないでください
        - model 変数 と pdf_file_urls 変数は編集しないでください
        - 回答は日本語で生成してください
    """
    # build collection
    source_list_str = asyncio.run(build_collection())

    # query
    client = OpenAI()
    query_args = rephrase_query(question, source_list_str)
    hybrid_search_result = hybrid_search(**query_args.model_dump())

    # 検索結果が得られなかった場合
    if not hybrid_search_result:

        # 画像コレクションがある場合、画像検索を実施
        image_paths = []
        if hasattr(IMAGE_COLLECTION_MAP, query_args.collection_name):
            image_paths = query_image_collection(
                query_args.collection_name, query_args.vector_query_text
            )

        if image_paths:
            response = chat_image(client, question, image_paths)

        # 画像コレクションがなく、検索結果が得られなかった場合、メタデータフィルターを排除して再検索
        else:
            hybrid_search_result = hybrid_search(
                **query_args.model_dump(include={"collection_name", "vector_query_text"})
            )
            response = chat(client, question, hybrid_search_result)

    else:
        response = chat(client, question, hybrid_search_result)

    # 戻り値として質問に対する回答を返却してください。
    answer = response

    return answer


# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)

        for attempt in range(2):  # 最大2回試行
            try:
                run_id = cb.traced_runs[0].id
                break
            except IndexError:
                if attempt == 0:  # 1回目の失敗時のみ
                    time.sleep(3)  # 3秒待機して再試行
                else:  # 2回目も失敗した場合
                    raise RuntimeError("Failed to get run_id after 2 attempts")

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
# ==============================================================================
