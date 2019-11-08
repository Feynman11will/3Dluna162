
from train.detector import trainner

'''
python的模块导入
1. 简单地三种情况
    1. 平级导入
    2. 主程序在主目录下，模块在子目录下：
     -- src
    |-- mod1.py
    |-- mod2
    |     |-- mod2.py
    |-- test1.py
    在mode2中生成__init__配置文件
    from mod2.mod2 import *  或者 import mod2.mod2.
    3. 
     -- src
    |-- mod1.py
    |-- mod2
    |     |-- mod2.py
    |-- sub
    |     |-- test2.py
    |-- test1.py
    若在程序test2.py中导入模块mod1和mod2。
    首先需要在mod2下建立__init__.py文件(同(2))，src下不必建立该文件。
    下面程序执行方式均在程序文件所在目录下执行，如test2.py是在
    # cd sub;之后执行
    # python test2.py
    在test2中
    import sys
    sys.path.append("..")
    import mod1
    import mod2.mod2
'''