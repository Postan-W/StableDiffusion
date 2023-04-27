import os

#往文件夹下添加下一个不重复名的文件
def get_next(target_dir,suffix):
    files = os.listdir(target_dir)
    files = [file.split(".")[0] for file in files]
    numbers = []
    for name in files:
        try:
            number = int(name)
            numbers.append(number)
        except:
            pass
    max_number = max(numbers)
    return str(max_number+1) + suffix

# print(get_next("./images",".jpg"))