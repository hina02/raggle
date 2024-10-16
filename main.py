import json
import sys
import os
import datetime

from dotenv import load_dotenv
from langchain import callbacks
from pydantic import BaseModel
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter, SpacyTextSplitter
from langchain_openai.embeddings.base import OpenAIEmbeddings   # batch使っていない


#==============================================================================
# !!! 警告 !!!: 以下の変数を変更しないでください。
#==============================================================================
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
#==============================================================================


#==============================================================================
# この関数を編集して、あなたの RAG パイプラインを実装してください。
# !!! 注意 !!!: デバッグ過程は標準出力に出力しないでください。
#==============================================================================
KEYWORDS = []   # from Sudachi, to hybrid search(vetor and BM25) in chroma

# リクエスト5回制限のため、Batch API を使用する。
# ①　テキスト変換（代名詞、年月日）
# ②　Embedding　
# ③　クエリ変換
# ④　ReRank
# ⑤　結果出力

class Paragraph(BaseModel):
    paragraph_id: int
    text: str
    embedding_content: str  # 代名詞を明確な名詞に変換
    # embedding: list[float]

class Article(BaseModel):
    article_id: int
    text: str
    # embedding: list[float]
    paragraphs: list[Paragraph]

class Party(BaseModel): # colletion2
    name: str
    address: str
    contact: str
    delegate: str
    # embedding: list[float]

class ContractAgreement(BaseModel): #collection1
    title: str
    text: str
    PartyA: Party
    PartyB: Party
    created_at: str # 可能ならdatetime
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
        if file_name.endswith('.pdf'):
            pdf_files.append(os.path.join(dir_path, file_name))
    return pdf_files

def load_pdf(path: str) -> list[Document]:
    pdf_loader = PyPDFLoader(file_path=path)
    separators = []
    article_pattern = r"第[ 　]*[０-９0-9一二三四五六七八九十百]+[ 　]*条"
    section_pattern = r"[ 　]*[０-９0-9]+[ 　]*\.[ 　]*"

    text_splitter = RecursiveCharacterTextSplitter(separators = separators, is_separator_regex = True)


    pdf_table_loader = PDFPlumberLoader(file_path=path) # TODO Check
    pages = pdf_loader.load_and_split(text_splitter=text_splitter)


    pdf_files = pdf_loader.load_files(pdf_file_urls)
    return pdf_files


def rag_implementation(question: str) -> str:
    # once load pdf_files

    
#==============================================================================


#==============================================================================
# !!! 警告 !!!: 以下の関数を編集しないでください。
#==============================================================================
def main(question: str):
    with callbacks.collect_runs() as cb:
        result = rag_implementation(question)
        run_id = cb.traced_runs[0].id

    output = {"result": result, "run_id": str(run_id)}
    print(json.dumps(output))


if __name__ == "__main__":
    load_dotenv()

    if len(sys.argv) > 1:
        question = sys.argv[1]
        main(question)
    else:
        print("Please provide a question as a command-line argument.")
        sys.exit(1)
#==============================================================================
