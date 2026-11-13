import os

file_path = os.path.join("C:/Users/lsasi/OneDrive/Desktop/probnikraga/data/lists.txt")
target_char = "#"


if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
 
    updated_content = content.replace(target_char, f"\n{target_char}")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(updated_content)

    print("Готово! Перенос строки успешно добавлен.")
else:
    print(f"Ошибка: файл по пути {file_path} не найден.")
