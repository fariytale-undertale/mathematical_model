alien_0 = {"color": "green","points":5}#初始化，每个部分被称为键值对
print(alien_0["color"])#取用
#1添加
alien_0["x_position"]=0
alien_0["y_position"]=25
print(alien_0)

#2修改
alien_0["color"]="yellow"

#3删除
del alien_0["y_position"]

#4定义
favourite_languages={
    "jen":"python",
    "sarah":"c",
    "phil":"java",
    "edward":"rust",#此逗号可加可不加
}

#5使用get来访问值（可处理不存在的键）
alien_0.get("points","No point value assigned")#若第二个参数没写，不存在时则会返回None

#6遍历
##1遍历所有键值对
for key,value in favourite_languages.items():#用.items取键值对，key和value可任意命名
    print(f"Key:{key}")
    print(f"Value:{value}")
##2遍历所有键
for name in favourite_languages.keys():#.keys可省略，一般保留可读性
    print(name.title())
#3按特定顺序遍历所有的键
for name in sorted(favourite_languages.keys()):
    print(f"{name.title()}")
##4遍历所有值
for language in favourite_languages.values():
    print(f"{language.title()}")
##5遍历所有值且不输出重复元素
for language in set(favourite_languages.values()):#实际上是创造一个集合set，集合只有值没有键，如la={"c","python","java"}为一个集合，集合中的重复元素会被省略
    print(f"{language.title()}")

#7字典列表
alien_0 = {"color": "green","points":5}
alien_1 = {"color": "yellow","points":10}
alien_2 = {"color": "blue","points":15}
alien = [alien_0,alien_1,alien_2]

alien=[]#创建一个含30个元素的列表
for alien_number in range(30):#range(x)表示从0到29，等价于循环30次
    new_alien = {"color": "green","points":5}
    alien.append(new_alien)

#8在字典中储存列表
pizza ={
    "crust":"thick",
    "toppings":["mushrooms","extra cheese"],
}
print(pizza["toppings"])

#9在字典中储存字典
users = {
    "aeinstein" : {
        "first":"albert",
        "last" :"einstein",
        "location" : "princeton",
    },

    "mcurie": {
        "first":"marine",
        "last" :"curie",
        "location" : "paris",
    },
}

