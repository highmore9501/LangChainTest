from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
import os


# 用文件夹里的txt文件生成长期记忆库
def generate_character_feature_chats(character_name: str):
    loader = DirectoryLoader(f"src/settings/featured_chats/{character_name}", glob="*.txt", show_progress=True,
                             use_multithreading=True, loader_cls=TextLoader, loader_kwargs={'autodetect_encoding': True})
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(docs)} chunks.")
    for i in range(len(docs)):
        print(docs[i].page_content)

    # 保存embedding到db文件夹下
    persist_directory = f'vectorDB/{character_name}'

    # 如果没有db文件夹，就创建一个
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embedding = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding, persist_directory=persist_directory)
    # 使用vectordb持久化，也就是保存到db文件夹下
    vectordb.persist()
    vectordb = None


if __name__ == "__main__":
    generate_character_feature_chats("韦小宝")
