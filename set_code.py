# 使用 set() 函数创建集合
another_set = set([1, 2, 3, 4])
my_set = {1, 2, 3}

# 添加元素
my_set.add(4)

# 删除元素
my_set.remove(3)  # 如果元素不存在会抛出 KeyError
# my_set.discard(5)  # 如果元素不存在不会抛出异常

# 清空集合
my_set.clear()

# 判断元素是否在集合中
print(1 in my_set)  # 输出: False

set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
# 并集
print(set1 | set2)  # 输出: {1, 2, 3, 4, 5, 6}
print(set1.union(set2))  # 输出: {1, 2, 3, 4, 5, 6}

# 交集
print(set1 & set2)  # 输出: {3, 4}
print(set1.intersection(set2))  # 输出: {3, 4}

# 差集
print(set1 - set2)  # 输出: {1, 2}
print(set1.difference(set2))  # 输出: {1, 2}

# 对称差集
print(set1 ^ set2)  # 输出: {1, 2, 5, 6}
print(set1.symmetric_difference(set2))  # 输出: {1, 2, 5, 6}
my_set = {1, 2, 3, 4}

#循环
for item in my_set:
    print(item)
#长度
print(len(my_set))  # 输出: 4
#复制
copied_set = my_set.copy()
print(copied_set)  # 输出: {1, 2, 3, 4}

removed_item = my_set.pop()
print(removed_item)  # 输出: 随机的一个元素，例如 1
print(my_set)  # 输出: 剩下的元素，例如 {2, 3, 4}