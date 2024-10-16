import datetime
import json
import os
import sys

from dotenv import load_dotenv
from langchain import callbacks
from langchain_community.document_loaders import PDFPlumberLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_openai.embeddings.base import OpenAIEmbeddings  # batch使っていない
from langchain_text_splitters import (
    CharacterTextSplitter,
    NLTKTextSplitter,
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
)
from pydantic import BaseModel

# ==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
# ==============================================================================
model = "gpt-4o-mini"
pdf_file_urls = [
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Architectural_Design_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Call_Center_Operation_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Consulting_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Content_Production_Service_Contract_(Request_Form).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Customer_Referral_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Draft_Editing_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Graphic_Design_Production_Service_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Advisory_Service_Contract_(Preparatory_Committee).pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/M&A_Intermediary_Service_Contract_SME_M&A_[Small_and_Medium_Enterprises].pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Manufacturing_Sales_Post-Safety_Management_Contract.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/software_development_outsourcing_contracts.pdf",
    "https://storage.googleapis.com/gg-raggle-public/competitions/29676d73-5675-4278-b1a6-d4a9fdd0a0ba/dataset/Technical_Verification_(PoC)_Contract.pdf",
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


class Article(BaseModel):
    article_id: int
    title: str
    text: str
    # embedding: list[float]
    related_articles: list[int]  # list of article_id


class Party(BaseModel):  # colletion2
    name: str
    address: str
    contact: str
    delegate: str
    # embedding: list[float]


class ContractAgreement(BaseModel):  # collection1
    title: str
    text: str
    PartyA: Party
    PartyB: Party
    created_at: str  # 可能ならdatetime
    # embedding: list[float]
    articles: list[Article]


# 1st LLM (analyse query [contract(type, date), party, , etc.])
Prompt = """

{"contract_agreement":
 "party": None,
 "article_title": None,   options = ["定義", "]
 "Paragraph": None,
 }
"""


def load_pdf_files(dir_path: str = "dataset"):
    pdf_files = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".pdf"):
            pdf_files.append(os.path.join(dir_path, file_name))
    return pdf_files


def load_pdf(path: str) -> list[Document]:
    separators = ["\n+第\s*[０-９0-9一二三四五六七八九十百]+\s*条", "\n+[（【]\s*別紙\s*[）】]\n+"]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
        separators=separators,
        is_separator_regex=True,
    )

    # pdf_loader = PyPDFLoader(file_path=path)
    pdf_lumber_loader = PDFPlumberLoader(file_path=path)  # TODO Check
    documents = pdf_lumber_loader.load_and_split(text_splitter=text_splitter)

    # document.page_contentがseparatorsにマッチしていない場合、一つ前のdocument.page_contentに結合する。
    # [０-９0-9一二三四五六七八九十百]を抜き出し、intに変換し、article_idに渡す。
    for document in documents:
        print(document.id)
        print(document.metadata)
        print(document.page_content)

    # 条のタイトルを抜き出し、article_titleに渡す。

    # 関連する条項（前項は問題なし。第2条第1項等を抜き出し、related_articlesに渡す。前条の場合も同様。）
    # （存続条項は除外する）

    return documents


load_pdf(pdf_file_urls[0])


def rag_implementation(question: str) -> str:
    # once load pdf_files
    pass


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
