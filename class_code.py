#一次尝试
class Dog:#首字母大写的名称称为类
    """一次模拟小狗的简单尝试"""

    def __init__(self,name,age):#当根据该类创建新的实例时，__init__函数会自动运行,前后各两个下划线
        """初始化"""
        self.name=name
        self.age=age#self起到指向实例本体的作用，通过实例访问的变量称为属性

    def sit(self):
        print(f"{self.name} is now sitting.")

    def roll_over(self):
        print(f"{self.name} rolled over!")


class Car:
    def __init__(self,make,model,year):
        self.make=make
        self.model=model
        self.yesr=year
        self.odometer_reading=0#给属性设定默认值

    def update_odometer(self,mileage):
        self.odometer_reading = mileage
    
    def increment_odometer(self,miles):
        self.odometer_reading+=miles

my_dog=Dog("Willie",6)
#1访问属性
my_dog.name
my_dog.age
#2调用方法
my_dog.sit()
my_dog.roll_over()

my_car=Car("audi","a4",2024)
#3修改属性的值
my_car.odometer_reading = 23
my_car.update_odometer(23)
my_car.increment_odometer(10)

#4继承（编写一个类的特殊版本，即子类）
class ElectricCar(Car):
    def __init__(self, make, model, year):
        super().__init__(make,model,year)#super()使可调用父类的所有方法
        self.bettery_size = 40#加特有属性

    #重写父类的方法def update_odometer(self,miles):

my_leaf = ElectricCar("nissan","leaf",2024)

#5将实例用作属性(即将大类中的一个模块拆分出来)
class Battery:
    def __init__(self,battery_size =40):
        self.bettey_size = battery_size

#class ElectricCar(Car):
    #def __init__(self, make, model, year):
        #super().__init__(make,model,year)
        #self.battery=Battery()

#6导入类(同函数)
#car.py中有class Car
#from car import Car,ElectricCar

#7导入整个模块，再用点号访问类(同函数)
#car.Car()

#8导入模块中的所有类(不推荐，同函数)
#from Car import *

#9导入的类中被导入了类时，被导入的类应先于该类导入文件

#10使用别名(同函数用as)

