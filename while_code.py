#格式
number = 1
while number <= 5 :
    print(number)
    number += 1

#使用标志
x=True
while x :
    message = input()
    if message == "quit" :
        x = False
    else :
        print(message)

#break退出循环，countinue结束当前循环进入下一个循环
#可用crtl+c结束无限循环

#在列表中移动元素
uc_users = ["alice","bob","candace"]
c_users = []
while uc_users : # 列表非空时为 True，空列表时为 False
    user =uc_users.pop()
    c_users.append(user)
print(c_users)
#删除特定值的所有列表元素
pets = ["dog","cat","rabbit","cat","cat","dog"]
while "cat" in pets :
    pets.remove("cat")
print(pets)
#使用用户输入填充字典