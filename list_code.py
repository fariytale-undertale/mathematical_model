bicycles=["trek","cannondale","redline","specialized"]#可定义为空，即[]
print(bicycles)#可一次性全部打印
#同数组

print(bicycles[0])
print(bicycles[-1])#倒数第一个
print(bicycles[-4])#倒数第四个

#1在列表末尾添加元素
bicycles.append("ducati")
#2在列表中插入元素
bicycles.insert(0,"ducati")

#3在列表中删除元素
del bicycles[0]#用del删除具体位置的元素
popped_bicycles=bicycles.pop()#用pop（）删除:删除末尾元素并将其返回(等价于栈的弹出)
#.pop（i）等价于弹出bicycles[i]并将其从中删除
bicycles.remove("redline")#remove根据值删除(只删除第一个出现的值，注意循环)

#4列表排序
bicycles.sort()#按首字母顺序排序，改变列表【根据ASCII码排序，大写字母比小写字母小】
bicycles.sort(key=str.lower)#不区分大小写顺序排序
bicycles.sort(reverse=True)#按首字母反向排序，改变列表
sorted(bicycles)#不改变列表，返回排序后的列表
sorted(bicycles,reverse=True)

#5反转列表
bicycles.reverse()

#6确定列表长度
len(bicycles)#返回一个整数

#7遍历
for bicycle in bicycles:
    print(bicycle)

#8数值列表

##1创建数值列表
for value in range(1,5):#1-4
    print(value)
numbers=list(range(1,6))#创建
range(1,12,2)#最后一个数字表示间隔，这里表示奇数
min(numbers)
max(numbers)
sum(numbers)
##2列表推导式
squares=[value for value in range(1,11)]
squares=[value**2 for value in range(1,11)]

#9切片（取列表的一部分）
players=["charles","martina","michael","florence","eli"]
print(players[0:3])#0-2,同range
print(players[:4])#默认从开头开始
print(players[2:])#默认到结尾
print(players[0:4:2])#第三个表示间隔
for player in players[0:3]:#同上
    break

#10复制列表
foods=["pizza","frenchfries","apple"]
new_foods=foods[:]#这会创建一个新列表，new_foods=foods指向的是同一个列表

#11元组（不可变列表）
dimensions=(200,50)#用圆括号标识(严格的说是用逗号来标识的，如dimensions=（1，）)
#dimensions[0]=1会报错，但可以给整个元组赋值，如dimensions=(200,100)
