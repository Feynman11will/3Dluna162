# 1. 描述
- 1 该文件主要实现了数据集的处理和准备工作

# 2. use method 
- 1 configure 中存储了输入输出文件的路径
- 2 loader 实现了mhd文件加载、处理方法
- 3 prepare 原始数据转换提取肺部区域方法（luna16 官方已经将mask存储在segment中，此步骤可忽略）
- 4 step1 实现了prepare操作中的数据前期数据处理操作
- 5 extract 实现了肺部区域特征提取操作


