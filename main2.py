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
from chromadb.utils.data_loaders import ImageLoader
from dotenv import load_dotenv
from langchain import callbacks
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI
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


class PageDocument(BaseModel):
    source: str
    page: int
    text: str
    distance: float


class DocumentInfo(BaseModel):
    path: str
    source: str
    collection_name: str
    is_valid: bool


class Table(BaseModel):
    table_name: str
    caption: str | None
    table_schema: str | None
    source: str
    page: int


class QueryArg(BaseModel):
    collection_name: str
    vector_query_text: str
    query_keyword: str | None = None
    source: str = None


class QueryArgs(BaseModel):
    args: list[QueryArg]


class EvalResult(BaseModel):
    is_answerable: bool


IMAGE_DIR = "page_images"
os.makedirs(IMAGE_DIR, exist_ok=True)
TABLE_IMAGE_DIR = "table_images"
os.makedirs(TABLE_IMAGE_DIR, exist_ok=True)


SOURCE_LIST = {
    "COMPANY_INFORMATION": [],
    "PRODUCT_INFORMATION": [],
    "RESEARCH_INFORMATION": [],
}

COLLECTION_MAP = {}
TABLE_COLLECTION_MAP = {}

REPHRASE_PROMPT = """
    Compose Args for hybrid search, this function performs a hybrid search by combining vector similarity and keyword-based filtering
    to retrieve relevant documents from the specified collection. Output multiple combinations for query_keyword.

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
        85期のキャッシュフローは？
          > query_keyword="キャッシュフロー"
            query_keyword="85期"
        ストックオプションの有無を教えてください。
          > hybrid_search_tool("COMPANY_INFORMATION", "ストックオプションの有無", query_keyword="ストックオプション")
        どういう製品がありますか？
          > hybrid_search_tool("PRODUCT_INFORMATION", "製品一覧", source="〇〇社製品パンフレット")

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


def update_source_list_str():
    global SOURCE_LIST_STR
    SOURCE_LIST_STR = ""
    for collection_name in SOURCE_LIST.keys():
        sources = []
        for item in SOURCE_LIST[collection_name]:
            sources.append(item.source)
        SOURCE_LIST_STR += f"""
        **Collection** : {collection_name}
        Sources : {", ".join(sources)}
        """


# Create Chroma Collection
# Create MultiModal Collection  # 律速
async def create_multimodal_collection(client, collection_name: str):
    collection = client.create_collection(
        name=collection_name + "_IMAGE",
        embedding_function=OpenCLIPEmbeddingFunction(),
        data_loader=ImageLoader(),
    )
    return collection


# Create Collection
def create_collection_map(client):
    emb_fn = embedding_functions.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small", api_key=os.environ["OPENAI_API_KEY"]
    )

    if SOURCE_LIST["COMPANY_INFORMATION"]:
        COLLECTION_MAP["COMPANY_INFORMATION"] = client.create_collection(
            name="COMPANY_INFORMATION", embedding_function=emb_fn
        )
        TABLE_COLLECTION_MAP["COMPANY_INFORMATION"] = client.create_collection(
            name="COMPANY_INFORMATION_TABLE", embedding_function=emb_fn
        )
    if SOURCE_LIST["PRODUCT_INFORMATION"]:
        COLLECTION_MAP["PRODUCT_INFORMATION"] = client.create_collection(
            name="PRODUCT_INFORMATION", embedding_function=emb_fn
        )
        TABLE_COLLECTION_MAP["PRODUCT_INFORMATION"] = client.create_collection(
            name="PRODUCT_INFORMATION_TABLE", embedding_function=emb_fn
        )
    if SOURCE_LIST["RESEARCH_INFORMATION"]:
        COLLECTION_MAP["RESEARCH_INFORMATION"] = client.create_collection(
            name="RESEARCH_INFORMATION", embedding_function=emb_fn
        )
        TABLE_COLLECTION_MAP["RESEARCH_INFORMATION"] = client.create_collection(
            name="RESEARCH_INFORMATION_TABLE", embedding_function=emb_fn
        )


# Extract PDF
async def save_page_image(page: pymupdf.Page, num_page: int, source: str, dir_name: str) -> str:
    zoom_factor = 2.0
    mat = pymupdf.Matrix(zoom_factor, zoom_factor)  # Set zoom matrix

    bbox = page.rect
    header_bbox = (bbox.x0, bbox.y0, bbox.x1, bbox.y1 * 0.1)
    footer_bbox = (bbox.x0, bbox.y1 * 0.9, bbox.x1, bbox.y1)
    pix = page.get_pixmap(matrix=mat)  # Apply zoom
    header_pix = page.get_pixmap(matrix=mat, clip=header_bbox)
    footer_pix = page.get_pixmap(matrix=mat, clip=footer_bbox)

    image_id = f"{source}_{num_page + 1}_image.png"
    image_path = os.path.join(dir_name, image_id)
    await asyncio.to_thread(pix.save, image_path)
    await asyncio.to_thread(header_pix.save, image_path.replace(".png", "_header.png"))
    await asyncio.to_thread(footer_pix.save, image_path.replace(".png", "_footer.png"))
    return image_id


async def save_table_image(
    page: pymupdf.Page, table_id: str, dir_name: str, bbox: tuple
) -> tuple[str, np.ndarray]:
    try:
        clip = pymupdf.Rect(*bbox) & page.rect
        if clip.width <= 0 or clip.height <= 0:
            raise ValueError("Invalid clip dimensions: width or height is zero or negative.")

        zoom_factor = 2.0
        mat = pymupdf.Matrix(zoom_factor, zoom_factor)

        pix = page.get_pixmap(matrix=mat, clip=clip)
        table_image_path = os.path.join(dir_name, f"{table_id}.png")
        await asyncio.to_thread(pix.save, table_image_path)

    except Exception:
        pass


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


async def add_pdf_text_from_image_data(
    doc: pymupdf.Document, source: str, collection_name: str, batch_size: int = 10
):
    collection = COLLECTION_MAP[collection_name]
    client = AsyncOpenAI()

    async def extract_page_text_from_image(num_page, page):
        image_path = await save_page_image(page, num_page, source, IMAGE_DIR)
        text = await chat_image_retry(
            client,
            text="画像には何が書かれていますか？ヘッダー、本文、フッター画像に含まれるテキストをすべて書き出してください。\n\n画像に含まれるテキストは以下の通りです。\n",
            image_paths=[image_path, image_path.replace(".png", "_header.png"), image_path.replace(".png", "_footer.png")],
            dir_name=IMAGE_DIR,
            system_prompt="あなたはOCRです。画像に含まれるテキストを、ヘッダー・フッター含めて漏れなくすべて書き出してください。",
        )
        return {
            "id": f"{source}_{num_page + 1}",
            "text": text,
            "metadata": {"source": source, "page": num_page + 1},
        }

    async def process_batch(batch):
        ids, texts, metadatas = [], [], []

        tasks = [extract_page_text_from_image(num_page, page) for num_page, page in batch]
        for page_data in await asyncio.gather(*tasks):
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


async def add_pdf_table_data(
    doc: pymupdf.Document, source: str, collection_name: DocumentCategory, batch_size: int = 10
):
    collection = TABLE_COLLECTION_MAP[collection_name]

    async def extract_page_tables(num_page, page):
        tables = await save_table(page, source, num_page)

        return [
            {
                "id": table.table_name,
                "text": f"table caption: {table.caption}\n\nschema: {table.table_schema}",
                "metadata": {"source": table.source, "page": num_page + 1},
            }
            for table in tables
        ]

    async def process_batch(batch):
        ids, texts, metadatas = [], [], []

        for num_page, page in batch:
            page_datas = await extract_page_tables(num_page, page)
            for page_data in page_datas:
                ids.append(page_data["id"])
                texts.append(page_data["text"])
                metadatas.append(page_data["metadata"])

        if ids and texts and metadatas:
            await asyncio.to_thread(collection.add, documents=texts, metadatas=metadatas, ids=ids)

    batch = []
    for num_page, page in enumerate(doc):
        batch.append((num_page, page))

        if len(batch) == batch_size:
            await process_batch(batch)
            batch = []

    if batch:
        await process_batch(batch)


def get_caption_text(page: pymupdf.Page, table_bbox, margin=20):
    """
    テーブル上下のキャプションを取得する
    table_bbox: (x0, y0, x1, y1) テーブル全体の座標 左上が原点（x0,y0）
    """
    x0, y0, x1, y1 = table_bbox
    top_box = (x0, y0 - margin, x1, y0)
    top_text = page.get_textbox(top_box)
    bottom_box = (x0, y1, x1, y1 + margin)
    bottom_text = page.get_textbox(bottom_box)
    return top_text.strip(), bottom_text.strip()


async def save_table(
    page: pymupdf.Page, source: str, num_page: int
) -> tuple[list[Table], list[str]]:
    tables = page.find_tables(strategy="lines_strict").tables

    table_datas = []
    if tables:
        for index, table in enumerate(tables):
            # table caption
            bbox = table.bbox
            caption = None
            if bbox:
                top_caption, bottom_caption = get_caption_text(page, bbox, margin=20)
                caption = f"{top_caption}\n\n{bottom_caption}"

            # table dataframe
            df = table.to_pandas()
            if table.header.external:  # ヘッダーが表の外部の場合、一行目をヘッダーに置き換える
                new_header = df.iloc[0]
                df = df[1:]
                df.columns = new_header
                df.reset_index(drop=True, inplace=True)

            if all(
                col is not None and col.startswith("Col") for col in df.columns
            ):  # ヘッダーがすべてデフォルト列名の場合、保存しない
                continue

            # テーブルはsqliteに保存せず、スクショする
            table_id = f"{source}_{num_page}_{index}"
            await save_table_image(page, table_id, TABLE_IMAGE_DIR, bbox)

            # table schema
            table_schema = {"columns": df.columns.tolist(), "index": df.iloc[:, 0].tolist()}

            table_data = Table(
                table_name=table_id,
                caption=caption,
                table_schema=json.dumps(table_schema, ensure_ascii=False, indent=2),
                source=source,
                page=num_page,
            )
            table_datas.append(table_data)

    return table_datas


async def load_pdf(info: DocumentInfo):
    pdf_path = info.path
    source = info.source
    collection_name = info.collection_name

    file_name = os.path.basename(pdf_path)
    source = file_name.split(".")[0]

    with pymupdf.open(pdf_path) as doc:
        if info.is_valid:
            await add_pdf_text_data(doc, source, collection_name)
            await add_pdf_table_data(doc, source, collection_name)
        else:
            await add_pdf_text_from_image_data(doc, source, collection_name)
            await add_pdf_table_data(doc, source, collection_name)


async def build_collection():
    paths = [download_file(url) for url in pdf_file_urls]
    # classify pdf
    document_infos = await classify_pdfs(paths)
    for info in document_infos:
        if info.collection_name in SOURCE_LIST:
            SOURCE_LIST[info.collection_name].append(info)
    update_source_list_str()

    # create collection
    client = chromadb.Client()
    create_collection_map(client)

    # add pdf data to chroma
    tasks = [load_pdf(info) for info in document_infos]
    await gather(*tasks)


# Query


def encode_image(image_path: str, dir_name: str):
    path = os.path.join(dir_name, image_path)

    with open(path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def chat_image(
    client: AsyncOpenAI, text: str, image_paths: list[str], dir_name: str, system_prompt: str = None
) -> str:

    messages = [
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]

    for image_path in image_paths:
        base64_image = encode_image(image_path, dir_name)
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        }
        messages[0]["content"].append(image_message)

    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})

    response = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
        max_tokens=10000,
    )
    return response.choices[0].message.content


async def chat_image_retry(
    client: AsyncOpenAI, text: str, image_paths: list[str], dir_name: str, system_prompt: str = None
) -> str:
    count = 0
    max_retry = 2
    while count < max_retry:
        response = await chat_image(client, text, image_paths, dir_name, system_prompt)
        if len(response) > 100:
            return response
        count += 1
    return response


def hybrid_search_base(
    collection,
    vector_query_text: str,
    query_keyword: str = None,
    source: str = None,
    top_k: int = 10,
):
    search_dict = {"$contains": query_keyword} if query_keyword else None
    metadata_filter = {"source": {"$eq": source}} if source else None

    return collection.query(
        query_texts=[vector_query_text],
        n_results=top_k,
        where=metadata_filter,  # ソースで絞り込み
        where_document=search_dict,  # キーワードで絞り込み
    )


def hybrid_search(
    collection_name: str,
    vector_query_text: str,
    query_keyword: str = None,
    source: str = None,
    top_k: int = 10,
) -> list[PageDocument]:
    collection = COLLECTION_MAP[collection_name]
    results = hybrid_search_base(collection, vector_query_text, query_keyword, source, top_k)

    if not results:
        return ""

    documents = [
        PageDocument(
            source=results["metadatas"][0][i]["source"],
            page=results["metadatas"][0][i]["page"],
            text=results["documents"][0][i],
            distance=results["distances"][0][i],
        )
        for i in range(len(results["documents"][0]))
    ]
    documents = [doc for doc in documents if doc.distance < 1.48]  # 類似度の閾値

    return documents


def hybrid_search_table(
    collection_name: str,
    vector_query_text: str,
    query_keyword: str = None,
    source: str = None,
    top_k: int = 5,
) -> list[str]:
    collection = TABLE_COLLECTION_MAP[collection_name]
    results = hybrid_search_base(collection, vector_query_text, query_keyword, source, top_k)
    if not results:
        return []
    return results.get("ids", [])[0]


def rephrase_query(query: str) -> QueryArgs:
    rephrase_prompt = REPHRASE_PROMPT.format(SOURCE_LIST_STR)
    llm = ChatOpenAI(model=model, temperature=0.3)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", rephrase_prompt),
            ("human", "{text}"),
        ]
    )

    structured_llm = prompt | llm.with_structured_output(QueryArgs)
    return structured_llm.invoke(query)


async def chat(client: AsyncOpenAI, text: str, retrieval_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": f"あなたはロートの社員です。検索情報をもとに、質問に答えてください。{retrieval_text}",
        },
        {"role": "user", "content": text},
    ]
    response = await client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=messages,
        max_tokens=10000,
    )
    return response.choices[0].message.content


async def eval(client: AsyncOpenAI, query: str, answer: str) -> bool:
    messages = [
        {
            "role": "system",
            "content": """与えられた回答が、質問に回答していない場合は、falseを、質問に回答している場合は、trueを返してください。回答はJSONです。
                        例
                        質問: 在外子会社の従業員数は？\n\n回答: 在外子会社の従業員数に関する資料は見つかりませんでした。  ＞　false
                        質問: 在外子会社の従業員数は？\n\n回答: 在外子会社の従業員数については分かりません。  ＞　false
                        質問: 在外子会社の従業員数は？\n\n回答: 在外子会社の従業員数については答えられません。  ＞　false
                        """,
        },
        {"role": "assistant", "content": f"質問: {query}\n\n回答: {answer}"},
    ]
    response = await client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=messages,
        max_tokens=100,
        response_format=EvalResult,
    )
    result = EvalResult.model_validate_json(response.choices[0].message.content)
    return result.is_answerable


def deduplicate_documents(retrieval_texts_list: list[list[PageDocument]]) -> str:
    seen = set()
    combined_text = ""

    for document_list in retrieval_texts_list:
        for page in document_list:
            key = (page.source, page.page)
            if key not in seen:
                seen.add(key)
                combined_text += f"# {page.source} Page {page.page}:\n{page.text}\n\n"
    return combined_text


async def query(client: AsyncOpenAI, question: str, query_args: list[QueryArg]):
    retrieval_texts_list = []
    retrieval_tables_list = []
    for query_arg in query_args:
        retrieval_texts_list.append(hybrid_search(**query_arg.model_dump()))
        retrieval_tables_list.append(hybrid_search_table(**query_arg.model_dump()))

    retrieval_texts = deduplicate_documents(retrieval_texts_list)
    retrieval_tables = []
    for table in retrieval_tables_list:
        retrieval_tables.extend(table)
    table_names = list(set(retrieval_tables))
    table_image_paths = [f"{name}.png" for name in table_names]

    # 検索結果が得られなかった場合、メタデータフィルターを排除して再検索
    if not retrieval_texts:
        retrieval_texts = hybrid_search(
            query_args[0].collection_name, query_args[0].vector_query_text
        )
        retrieval_texts = deduplicate_documents(retrieval_texts_list)
    text_response = await chat(client, question, retrieval_texts)
    is_answerable = await eval(client, question, text_response)
    if not is_answerable:
        return await chat_image(client, question, table_image_paths, TABLE_IMAGE_DIR)
    else:
        return text_response


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
    asyncio.run(build_collection())

    # query
    client = AsyncOpenAI()
    response = rephrase_query(question)
    query_args = response.args
    response = asyncio.run(query(client, question, query_args))

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
