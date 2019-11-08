# 路径解析方法
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
    3. 目录之间索引
       
            --src
            |-- mod1.py
            |-- mod2
            |     |-- mod2.py
            |-- sub
            |     |-- test2.py
            |-- test1.py
               
     
            若在程序test2.py中导入模块mod1和mod2。
    
            首先需要在mod2下建立__init__.py文件(同(2))，src下不必建立该文件。
    
            下面程序执行方式均在程序文件所在目录下执行，如test2.py是在
            
            1# cd sub;之后执行
            
            1# python test2.py
            
            在test2中
            
            import sys
            
            sys.path.append("..")
            
            import mod1
            
            import mod2.mod2
2. assert  方法

        1. raise :
            raise Exception("Invalid level!", level)
        2. assert 断言： 表达式为false为false时触发异常
            使用方法：
                assert(expression, [parameter])
                如：assert(a==b,'产生错误')
3. numpy学习
        
        1. np.where()返回符合少选条件的列表，元素为每一个维度对应位置的下标索引，
            形状为[n,]数值n为变量，不确定
        2. np.meshgrid(xx,yy,zz)xxyyzz为3个列表
            返回n维图像的点云坐标索引列表，长度为n
        3. np.load()输出的数据的形状[1,orig_shape]
        4. np 进行选择，a[list1,...listn] n为维度个数，listi为列表
            那么输出的数据的维度个数为n 
            a[0,0,0,0]这样输出的数据维度会下降
            
4. python shutil 函数
   
        1. python 操作文件例如复制粘贴等操作，使用shutil
5. python isinstance
    
        1. isinstance 
            isinstance() 函数来判断一个对象是否是一个已知的类型，类似 type()。