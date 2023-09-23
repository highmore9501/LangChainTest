import json


def pharseAnswer(answer):
    """Pharse the answer from the user.

        Args:
            answer (str): The answer from the user.

        Returns:
            str: The pharsed answer.
        """
    # 提取{}中的内容，包括{}
    answer = answer[answer.find("{"):answer.find("}") + 1]

    dictionary = json.loads(answer)
    print(dictionary)

    answerType = dictionary["type"]

    if answerType == "reply_chat":
        content = dictionary["content"]
        return [answerType, content]
    else:
        answerType == "reply_image"
        return answerType
