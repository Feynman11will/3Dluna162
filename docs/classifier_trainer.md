# 分类器训练思路

1. 总体流程
    
        1. 使用目标目标检测器输出的结果，以疑似含有目标的坐标为中心，crop出roi区域，crop_size =94
            data_classifer __getItem__函数逻辑
            1. conf_list pbb[:,0] 存在目标的置信度
            2. 计算要使用的id, id号码为conf的topk个pbb或者随机采样topk个pbb
            3. 根据该id，从有疑似目标的roi中提取roi检测区域 roi 的crop size 为96
            return 
                1. croplist topk个roi
                2. coordlist 1中对应的coord
                3. isnodlist 检测器中是否有目标 shape is topk 与 lbb.py 的iou大于阈值则为目标
                4. y ct图像对应的样本的分类
        2. 目标检测的流程
         
            input: b*topk 个roi区域 以及 coordlist  b x topk 坐标
                xlist: b x topk x 1x 96 x 96 x 96 # top5 cube proposal 的维度
                coordlist: b x topk x 3 x 24 x 24 x 24
            逻辑：
                使用目标检测网络对b*topk个目标区域检测
                noduleFeat 分类结果输出层的前一层，处理：对noduleFeat获取中心的centerfeat，维度[nxtopk, 64, 1, 1, 1 ]
                    经过两个全连接层,最终输出的维度为1 得到out
                    out shape = [b,topk]
                nodulePred 结节预测输出 形状为 [bxtopk,24,24,24,3,5] view as [b, topk, 24x24x24x3x5]
                casePred 为 对noduleFeat sigmod 偏移
                return : noduleFeat nodulePred out
            分类器的逻辑：
                1. 使用数据输入器根据目标检测器输出的topk个预测结果位置，在预测结果附近crop 出topk个目标作为疑似位置，cropsize=96
                2. 对该topk个roi进行分类
        3. 损失函数为交叉熵损失作为分类器的损失函数