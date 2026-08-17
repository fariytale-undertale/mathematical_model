age = input("Tell me somethinyour age\n")#内部为提示语句
print(age)
#if age >12 :  #不可比较，因为输入为字符串

prompt ="name\t"
prompt += "What your first name\n"#输出多行文本
name = input(prompt)
print(name)

#用int将字符串转化为数字
age = int(age)
if age >12 :
    print("true")
