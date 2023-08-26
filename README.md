# Infrared-and-Visible-Light-Fusion-Network-based-on-Multi-Scale-Autoencoder-Architecture
多尺度自编码网络的红外与可见光融合网络
分为两阶段：训练阶段：要训练一个自动编码器网络，其中编码器能够提取多尺度深度特征，解码器从这些特征重建输入图像。
融合阶段：融合网络包含三个主要部分：编码器、融合策略和解码器，这里编码器、解码器的参数由上述训练得到。
文件介绍：
dataset.py：编程并调试两个加载数据集的类。一个用于加载训练集和验证集，对coco数据集的单张自然图像进行预处理，返回tensor形式的自然图像。另一个用于加载测试集，对TNO数据集的红外与可见光图像进行预处理，返回tensor形式红外与可见光图像
Net.py:构建出此融合方法的网络。在初始化函数中定义各种神经网络操作。fusion仅在最后阶段使用，即final_net,训练只需训练encoder和decoder。
train.py:训练
test.py:测试
