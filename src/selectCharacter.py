import yaml

character_List = ['神里绫华']


def selectCharacter():
    character_name = None

    while character_name is None:
        character_index = input(f"请选择你想聊天的角色名\n1. 神里绫华\n")
        # 如果输入的是整数，而且在角色列表中
        if character_index.isdigit() and int(character_index) <= len(character_List):
            character_name = character_List[int(character_index) - 1]
        else:
            print("输入错误，请重新输入")

    character_file_path = f"settings/characters/{character_name}.yaml"
    print(character_file_path)
    character_settings = yaml.load(
        open(character_file_path, encoding="utf-8"), Loader=yaml.FullLoader)

    return character_settings
