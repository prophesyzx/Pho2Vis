# Pho2Vis
使用深度学习模型，基于全天空成像仪所拍摄的圆形天空图片进行大气能见度预测

## 目前进度
1. 实现了读取数据集的Dataset类
2. 微调预训练模型DenseNet121
3. 整体训练和评估模型框架已完成

## 目前问题
1. 由于没有测试集数据，不确定对训练集的读取是否能应用于测试集
2. 在数据处理上，目前是将不同曝光度的Image作为不同的图片进行训练，**但是**，还可以将两张图片进行拼接后进行训练，暂时没有对这个进行实现。
