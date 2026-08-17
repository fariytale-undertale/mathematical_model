#python中的异常名为exception
#异常一般会出现traceback并附有一些信息，为了避免给用户输出traceback
#1异常ZeroDivisionError
try:
    answer=5/0
except ZeroDivisionError:
    print("You can't divide by zero!")
else :
    print(answer)

#2异常FileNotFoundError
from pathlib import Path

path=Path("alice.txt")
try:
    contents =path.read_text(encoding="utf-8")
except FileNotFoundError:
    print(f"Sorry,the file {path} does not exist")