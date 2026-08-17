from pathlib import Path
path = Path("PEP.txt")
contents = path.read_text(encoding="utf-8")#读取文件全部内容， path.read_text()在读取完后会返回一个空字符串，会多一个空行
contents=contents.rstrip()#删除空行，也可 path.read_text().rstrip()
print(contents)