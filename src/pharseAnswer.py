import json


def pharseAnswer(answer):
    # 尝试解析answer，如果解析失败，那么就返回["regenerate_chat"]，以便于重新生成聊天内容
    try:
        # 如果anwer的结尾不是}，那么就在结尾加上}
        if answer[-1] != "}":
            answer = answer + "}"
        # 提取{}中的内容，包括{}
        answer = answer[answer.find("{"):answer.find("}") + 1]
        dictionary = json.loads(answer)
        print(dictionary)

        answerType = dictionary["type"]

        if answerType == "reply_chat":
            content = dictionary["content"]
            return [answerType, content]
        else:
            answerType == "send_image"
            return [answerType]
    except:
        return ["regenerate_chat"]
