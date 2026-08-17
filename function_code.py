#1函数定义def
def greet():
    """显示简单的问候语"""#文档字符串的注释，指明函数作用，由三个双引号引起
    print("Hello!")


#2向函数传递信息
def greet(name):
    """显示简单的问候语"""#文档字符串的注释，指明函数作用，由三个双引号引起
    print(f"Hello!{name.title()}")
greet("alan")

#3关键字实参（可调换参数顺序）
greet(name="alex")

#4默认值(不给设有默认值的形参传值时使用默认值)
def pet(pet_name,pet_type="dog"):#设有默认值时多使用关键字实参
    print(f"{pet_type}:{pet_name}")
#特殊用途，可将不一定会用到的参数的默认值设为空字符''或None

#5返回值（同c）可返回字典

#6传递列表(在函数中修改列表是永久的)(等价于c传递数组地址）
#传递不可修改的列表：list[:]

#7传递任意数量实参(*args：收集任意数量的位置实参；**kwargs:收集任意数量的关键字实参)
def make_pizza(size,*toppings) :#很像c中的传地址,但实际上是创建一个名为toppings的元组
    """打印顾客点的所有配料"""
    print(toppings)
make_pizza(12,"mushrooms","cheese")

def profile(first,last,**users_info):#**可创建一个字典
    users_info["first_name"]=first
    users_info["last_name"]=last
    return users_info
p=profile("a","b",location="princeton",field="physics")
print(p)

#8将函数存储在模块中(导入实质上是复制)
#如文件pizza.py中含有def make_pizza().

#导入模块(文件)：import pizza
#使用函数：pizza.make_pizza()

#导入特定函数：from pizza import f1,f2,f3
#导入特定函数使用函数：f1()

#使用as给函数指定别名：from pizza import make_pizza as mp
#使用：mp()

#使用as给模块指定别名：import pizza as p
#使用:p.make_pizza()

#用*导入所有函数（不建议）：from pizza import *
#使用：make_pizza()