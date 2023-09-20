# 导入各种模块
from promptTemplates.chatPromptTemplate import ChatPromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import VectorDBQA
from selectCharacter import selectCharacter
import yaml
from modelLoaders.huggingFaceLoader import HuggingFaceLaoder
from transformers import pipeline
from langchain import HuggingFacePipeline

# 加载模型
modelLoader = HuggingFaceLaoder()
model = modelLoader.model
tokenizer = modelLoader.tokenizer
generate_params = modelLoader.getGenerateParams()

pipe = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, **generate_params
)

llm = HuggingFacePipeline(pipeline=pipe)

# 设定长短期记忆的大小限制
featured_chats_max_tokens = 4096
recent_chat_contents_max_tokens = 4096

# 选择聊天对象
character_settings = selectCharacter()
# 加载角色的设定
character_name = character_settings['character_name']
character_persona = character_settings['persona']
user_name = character_settings['user_name']
user_persona = character_settings['user_persona']
location = character_settings['location']
environment = character_settings['environment']
identity = character_settings['identity']
relationship_description = character_settings['relationship_description']


# 初始化聊天短期记忆库,库里只保留最近19次对话内容
memory = ConversationBufferWindowMemory(
    human_prefix=user_name, ai_prefix=character_name, k=19)

# 加载角色的长期记忆
# 读取长期记忆库
persist_directory = f"../vectorDB/{character_name}"
embedding = HuggingFaceEmbeddings()
vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

# 生成聊天模板
promptTemplate = ChatPromptTemplate(memory=memory, db=vectordb, featured_chats_max_tokens=featured_chats_max_tokens,
                                    recent_chat_contents_max_tokens=recent_chat_contents_max_tokens)

# 开始聊天循环
while True:
    inputText = input("请输入你想说的话: ")

    promptFormatArgs = {
        "character_name": character_name,
        "user_name": user_name,
        "character_persona": character_persona,
        "user_persona": user_persona,
        "time": "不详",
        "location": location if location is not None else "",
        "environment": environment if environment is not None else "",
        "identity": identity if identity is not None else "",
        "relationship_description": relationship_description if relationship_description is not None else "",
        "nearest_user_chat": inputText
    }

    # 拼接聊天模板，生成当前的聊天内容
    currentPrompt = promptTemplate.format(**promptFormatArgs)

    answer = llm(currentPrompt)
    print(user_name + ": " + inputText)
    print(character_name + ": " + answer)

    # 将本次完成的聊天内容存入短期数据库
    memory.save_context(
        {"input": inputText},
        {"output": answer}
    )
