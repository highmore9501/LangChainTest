import json


def pharseAnswer(answer):
    if answer == "":
        return ["regenerate_chat"]
    # 去掉answer中的换行符
    answer = answer.replace("\n", "")
    # 去掉answer中的空格
    answer = answer.replace(" ", "")
    # 去掉answer中的\
    answer = answer.replace("\\", "")
    # 如果answer不包含{，那么可以假定answer是聊天内容，直接返回
    if "{" not in answer:
        return ["reply_chat", answer]
    # 尝试解析answer，如果解析失败，那么就返回["regenerate_chat"]，以便于重新生成聊天内容
    try:
        # 提取"type":后面的，第一个位于""中的内容，这个内容表示回复的类型
        answerType = answer.split('"type":')[1].split('"')[1]

        if answerType == "reply_chat":
            # 提取"content":后面的，第一个位于""中的内容，这个内容表示回复的内容
            content = answer.split('"content":')[1].split('"')[1]
            return [answerType, content]
        else:
            answerType == "send_image"
            return [answerType]
    except:
        return ["regenerate_chat"]


if __name__ == "__main__":
    answer = '''
{
    "type": "reply_chat",
    "content": "当然可以。我会非常认真地听取您的意见并尽力吸收它们来提高我的技巧水平。非常感谢您的关心和支持！"
}'''
    print(pharseAnswer(answer))
