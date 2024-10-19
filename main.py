import asyncio
import datetime
import json
import re
import sys
import uuid

from dotenv import load_dotenv
from langchain import callbacks
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    # "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
]
# ==============================================================================


# ==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
# ==============================================================================
KEYWORDS = []  # from Sudachi, to hybrid search(vetor and BM25) in chroma

# リクエスト5回制限のため、Batch API を使用する。
# ①　テキスト変換（代名詞、年月日） : 時間かかるなら没か。
# ②　Embedding
# ③　クエリ変換
# ④　ReRank
# ⑤　結果出力

Queries = [
    "ソフトウェア開発業務委託契約について、委託料の金額はいくらですか？",
    "グラフィックデザイン制作業務委託契約について、受託者はいつまでに仕様書を作成して委託者の承諾を得る必要がありますか？",
    "コールセンター業務委託契約における請求書の発行プロセスについて、締め日と発行期限を具体的に説明してください。",
    "フューチャービルディング株式会社との契約書において、第3条第1項にはどのような内容が記載されていますか？",
]


# Schema
class Article(BaseModel):  # collection1
    article_number: int | None
    heading: str
    text: str
    related_articles: list[int]  # list of article_id


class Party(BaseModel):
    name: str
    address: str
    contact: str
    delegate: str


class ContractAgreement(BaseModel):  # collection2
    title: str
    PartyA: Party
    PartyB: Party
    created_at: str = Field(description="ISO 8601 format datetime")


# 1st LLM (analyse query [contract(type, date), party, , etc.])
# ①　近い文書をサーチ
# ②　サーチ文書をもとに、以下の情報を確定させる。

# Q. 〇〇会社の2024年の代理契約書において、第3条第1項にはどのような内容が記載されていますか？
# A. party: 〇〇会社, title: 代理契約書,  created_at: 2024 > party, titleでfilter
# A. Heading: , article_number: 3,
# Article検索で完全一致させるのは怖いので、filterにtitleの候補を複数入れる形で
Prompt = """

{"contract_agreement":
 "party": None,
 "article_number": None,   options = ["定義", "]
 }
"""


async def main_processes() -> bool:
    tasks = []
    while pdf_file_urls:
        file_path = pdf_file_urls.pop(0)
        tasks.append(main_process(file_path))
    results = await asyncio.gather(*tasks)
    return all(results)


async def main_process(file_path: str) -> bool:
    """5sec / file"""
    start_time = datetime.datetime.now()
    # Load PDF
    document_id = uuid.uuid4().hex
    documents = DocumentLoader.load_pdf(file_path)

    # Extract Contract Agreement
    result = await extract_contract_agreement(documents)
    data = json.loads(result)
    for document in documents:  # filterに使用する値を追加
        document.metadata["source"] = document_id
        document.metadata["Title"] = data.get("title", "")
        document.metadata["PartyA"] = data.get("PartyA", {}).get("name", "")
        document.metadata["PartyB"] = data.get("PartyB", {}).get("name", "")

    # Add Chroma
    ChromaManager("article").add_documents(documents)

    document = Document(page_content=result, id=document_id)
    ChromaManager("contract_agreement").add_documents([document])
    end_time = datetime.datetime.now()
    print(f"Time: {end_time - start_time}")
    return True


# Step1. Load PDF
class DocumentLoader:
    ARTICLE_SEPARATOR = "第\s*[０-９0-9]+\s*条"
    ATTACHMENT_SEPARATOR = "[（【]\s*別紙\s*"

    @classmethod
    def load_pdf(cls, path: str) -> list[Document]:
        separators = [f"\n+{cls.ARTICLE_SEPARATOR}", f"\n+{cls.ATTACHMENT_SEPARATOR}\n+"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
            separators=separators,
            is_separator_regex=True,
        )

        # pdf_loader = PyPDFLoader(file_path=path)
        pdf_lumber_loader = PDFPlumberLoader(file_path=path)  # TODO Check
        documents = pdf_lumber_loader.load_and_split(text_splitter=text_splitter)

        documents = DocumentLoader._preprocess_documents(documents)
        return documents

    @classmethod
    def _preprocess_documents(cls, documents: list[Document]) -> list[Document]:
        """
        本文（序文、条文（複数）、締結文)、別紙の構成単位で結合する。
        本文メタデータにタイトル、条番号を付与する。
        """
        # 別紙を判定する。
        attachment_indices = []
        for index, document in enumerate(documents):
            document.page_content = document.page_content.strip()
            del document.metadata["file_path"]
            del document.metadata["Producer"]
            del document.metadata["CreationDate"]
            del document.metadata["ModDate"]

            if re.match(cls.ATTACHMENT_SEPARATOR, document.page_content):
                attachment_indices.append(index)

        # 別紙を結合する。
        if attachment_indices:
            DocumentLoader._combine_attachments(documents, attachment_indices)

        # 本文から、各条文のタイトル、条番号を抜き出す。
        main_body_range = (0, attachment_indices[0] - 1 if attachment_indices else len(documents))
        for document in documents[: main_body_range[1]]:

            document.metadata["Heading"] = DocumentLoader._extract_article_title(
                document.page_content
            )
            article_numbers = DocumentLoader._extract_article_numbers(document.page_content)
            if article_numbers:
                document.metadata["article_number"] = article_numbers[0]
                if document.metadata["Heading"] != "存続条項":
                    # 存続条項は関連条文を呼び出すメリットが低いため、除外
                    document.metadata["related_article_number"] = ",".join(article_numbers[1:])

        # Headingが取得できていない場合、一つ前の条文に結合する。
        for index in range(main_body_range[1] - 1, main_body_range[0], -1):
            document = documents[index]
            if not document.metadata.get("Heading"):
                documents[index - 1].page_content += document.page_content
                documents.pop(index)

        # article_numberのない本文を、序文及び締結文に分類。
        start_article = False
        start_attachment = False
        for document in documents[main_body_range[0] : main_body_range[1]]:
            if document.metadata.get("article_number"):
                start_article = True
            elif document.metadata.get("Heading") == "attachment":
                start_attachment = True
            else:
                if start_article and not start_attachment:
                    document.metadata["Heading"] = "signature"
                else:
                    document.metadata["Heading"] = "premable"

        # TODO 序文、締結文を各々結合

        return documents

    # FIXME
    @staticmethod
    def _combine_attachments(documents: list[Document], attachment_indices: list[int]):
        attachments = []
        for i in range(len(attachment_indices)):
            # 別紙範囲を指定
            start = attachment_indices[i]
            if i + 1 < len(attachment_indices):
                end = attachment_indices[i + 1] - 1
            else:
                end = len(documents) - 1

            combined_content = "".join([doc.page_content for doc in documents[start : end + 1]])
            attachment = documents[start]
            attachment.page_content = combined_content
            attachment.metadata["Heading"] = "attachment"
            attachments.append(attachment)

        # 別紙を結合したものを追加（結合前を削除）
        documents = documents[: attachment_indices[0]]
        documents.extend(attachments)
        return documents

    @classmethod
    def _extract_article_title(cls, text: str) -> str:
        pattern = rf"{cls.ARTICLE_SEPARATOR}\s*（([^）]+)）"
        match_text = re.match(pattern, text)
        return match_text.group(1) if match_text else ""

    @staticmethod
    def _extract_article_numbers(text: str) -> list[str]:
        """
        テキストの条番号及び関連する条番号を抜き出す。(項については不要)(TODO 漢数字は余裕があれば対応）
        article_number = match_nums[0]
        related_article_number = match_nums[1:]

        例: 第10条 第4条第1項から第2条第3項までと前条
        article_number = 10
        related_article_number = [4, 2, 9]
        """

        def fullwidth_to_halfwidth(zenkaku_num: str):
            return zenkaku_num.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

        match_nums = []  # []の場合は、第十条等で、article_numberが取得できない場合
        match_texts = re.findall(r"第\s*([０-９0-9]+)\s*条", text)
        if match_texts:
            match_nums = [int(fullwidth_to_halfwidth(num)) for num in match_texts]

            match_text = re.search("前条", text)
            if match_text:
                match_nums.append(match_nums[0] - 1)
        return list(map(str, match_nums))


# Step2. Extract Contarct Agreement FIXME
async def extract_contract_agreement(documents: list[str]) -> dict:
    contract_text = ""
    for document in documents:
        if document.metadata["Heading"] in ["premable", "signature"]:
            contract_text += document.page_content + "\n\n"

    chain = extract_contract_chain()
    result = await chain.ainvoke(contract_text)
    return result.json()


def extract_contract_chain():
    prompt = ChatPromptTemplate.from_messages(
        [("system", "Extract the contract agreement information."), ("human", "{text}")]
    )
    model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0.3)
    structured_llm = model.with_structured_output(ContractAgreement)
    chain = prompt | structured_llm
    return chain


# Chroma
class ChromaManager:
    def __init__(self, collection_name: str):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = self.vector_store.add_documents(documents)
        print(f"documents added to chroma {len(ids)} / {len(documents)}.")
        return ids

    def query(self, query: str, filter: dict, k: int = 4) -> list[tuple[Document, float]]:
        results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            embedding=self.embeddings.embed_query(query),
            k=k,
            filter=filter,
            # where_document={"$contains": {"text": "hello"}},
        )
        # results = self.vector_store.max_marginal_relevance_search_by_vector
        print(results)
        return results


def rag_implementation(question: str) -> str:
    return asyncio.run(main_processes())


rag_implementation("")

# ==============================================================================


# ==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
# ==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


# if __name__ == "__main__":
#     load_dotenv()

#     if len(sys.argv) > 1:
#         question = sys.argv[1]
#         main(question)
#     else:
#         print("Please provide a question as a command-line argument.")
#         sys.exit(1)
# ==============================================================================
