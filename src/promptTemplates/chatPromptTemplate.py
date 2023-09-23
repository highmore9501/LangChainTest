from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.vectorstores import Chroma
import yaml


class ChatPromptTemplate():
    memory: ConversationBufferWindowMemory
    db: Chroma
    featured_chats_max_tokens: int
    recent_chat_contents_max_tokens: int
    prompt: PromptTemplate

    def __init__(self, chatType: str, memory: ConversationBufferWindowMemory, db: Chroma, featured_chats_max_tokens: int = 1024, recent_chat_contents_max_tokens: int = 1024):
        self.memory = memory
        self.db = db
        self.featured_chats_max_tokens = featured_chats_max_tokens
        self.recent_chat_contents_max_tokens = recent_chat_contents_max_tokens
        # 加载基础的聊天模板
        commonChatPromptTemplatePath = f"promptTemplates\{chatType}.yaml"
        commonChatPromptTemplate = yaml.load(open(
            commonChatPromptTemplatePath, encoding="utf-8"), Loader=yaml.FullLoader)["prompt"]
        print(commonChatPromptTemplate)
        self.prompt = PromptTemplate.from_template(commonChatPromptTemplate)

    def format(self, **kwargs) -> str:
        character_name = kwargs["character_name"]
        user_name = kwargs["user_name"]
        inputText = kwargs["nearest_user_chat"]

        featured_chats = self.query_featured_chats(inputText)
        kwargs["featured_chats"] = featured_chats
        # 查询最近的聊天内容
        chat_historys = self.memory.load_memory_variables({})['history']

        recent_chat_contents = self.format_recent_chat_contents(
            chat_historys, character_name, user_name)
        kwargs["recent_chat_contents"] = recent_chat_contents
        result = self.format_prompt(**kwargs)
        # 因为原模板中不能有{...:....}这样的格式，上面的代码会报错，但又需要这样的格式，所以模板用<...:...>代替{...:...}
        # 然后再用下面的代码将<...:...>替换成{...:...}
        result = result.replace("<", "{")
        result = result.replace(">", "}")
        return result

    def query_featured_chats(self, inputText):
        # 查询长期记忆中与输入文本相似的内容，大小不要超过featured_chats_max_tokens
        max_tokens = self.featured_chats_max_tokens
        historys = self.db.search(inputText, search_type="similarity", k=19)
        page_contents = []
        for histroy in historys:
            page_contents.append(histroy.page_content)
        # 去掉重复的内容
        page_contents = list(set(page_contents))
        # 添加内容，限制在max_tokens以内
        featured_chats = []
        for content in page_contents:
            max_tokens -= len(content)
            if max_tokens <= 0:
                break
            else:
                featured_chats.append(content)
        # 将featured_chats的元素用\n连接起来
        result = "\n".join(featured_chats)
        return result

    def format_recent_chat_contents(self, chat_historys, user_name, character_name):
        # 下面这个format_recent_chat_contents方法要根据前面已经设定好的token上限来提取聊天内容，超出上限的古早对话会被丢弃
        # 然后将提取内容中的Human:和AI:替换成用户和ai人物的名字
        if len(chat_historys) < 2:
            return ""

        recent_chat_contents = ""
        max_tokens = self.recent_chat_contents_max_tokens
        # 把chat_historys以"\n"为分隔符分割成一个列表
        chat_historys_list = chat_historys.split("\n")
        # 计算chat_historys_list最后两个元素的字符数
        last_two_elements_length = len(
            chat_historys_list[-1]) + len(chat_historys_list[-2])

        # 如果last_two_elements_length < max_tokens，那么就把chat_historys_list最后两个元素加入到recent_chat_contents中
        while last_two_elements_length < max_tokens and len(chat_historys_list) > 1:
            recent_chat_contents = recent_chat_contents + "\n".join(
                [chat_historys_list[-1], chat_historys_list[-2]])
            max_tokens -= last_two_elements_length
            chat_historys_list = chat_historys_list[:-2]
            if len(chat_historys_list) > 1:
                last_two_elements_length = len(
                    chat_historys_list[-1]) + len(chat_historys_list[-2])

        return recent_chat_contents
