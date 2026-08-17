cars=["audi","bmw","subaru","toyota"]

for car in cars:
    if car=="bmw":
        print(car.upper())
    elif car=="audi":#等价于else if
        print(car.lower())
    else:
        print(car.title())

car.lower()=="bmw"#不区分大小写


#检查多个条件
age=19
if age>=18 and age<=20:
    print("true")
if age>=18 or age<=20:
    print("true")

#检查特定值是否在内
if "audi" in cars:
    print("true")
if "auti" not in cars:
    print("false")

#检验列表非空
if cars:
    print(cars)
else:
    print("error")
