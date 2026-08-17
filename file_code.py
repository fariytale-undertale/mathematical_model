from pathlib import Path#类似c语言的头文件
#1读取文件
#使用path.read_text(encoding="utf-8")读取中文文件
path = Path("file_trial.txt")
contents = path.read_text()#读取文件全部内容， path.read_text()在读取完后会返回一个空字符串，会多一个空行
contents=contents.rstrip()#删除空行，也可 path.read_text().rstrip()
print(contents)

###相对文件路径，文件夹/文件名；绝对文件路径

#2访问文件中的各行
lines = contents.splitlines()#分成一系列行
for line in lines:
    print(line)

#2.2访问文件中各字符串
words =contents.split()

#3使用文件内容(创建一个字符串储存所有内容)
string = ""
for line in lines :#注：所有文本都是字符串，想用数字要用int(),float()转换
    string += line 
print(string)

#4写入文件
##1写入一行
path.write_text("I love you")#只能传入字符串形式，若要传入数字，需用str()字符串化
##2写入多行
c="a.\n"
c+="b.\n"
path.write_text(c)#与c语言一样，该操作会先清空文件再写入

#5访问多个文件
filenames=["1.txt","2.txt"]
for file in filenames:
    path=Path(file)
    try:#此处还可以用if path.exist():
        contents =path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"Sorry,the file {path} does not exist")
    else:
        pass#什么都不做，静默

#6存储数据(用模块json存储)
import json

numbers = [2, 3, 5, 7, 11, 13]

path = Path("number.json")#通常使用文件拓展名.json
contents =json.dumps(numbers)#存储
path.write_text(contents)

contents = path.read_text()
numbers = json.loads(contents)#读取
print(numbers)


