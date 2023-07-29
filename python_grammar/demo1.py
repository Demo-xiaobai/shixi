import copy
'''
直接赋值：其实就是对象的引用（别名）。

浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象。

深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。
'''
a = {1:[1,2,3]}
b = a
c = a.copy()
d = copy.deepcopy(a)

a[1].append(256)

print(c)
print(b)



print(d)

