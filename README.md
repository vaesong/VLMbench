# my_vlmbench

## train hiverformer

训练的代码在 `vlm/scripts/train_hiverformer.py` 下面。主要是把 `hiverformer` 的网络结构用到了 `VLMbench` 的数据集上。

使用了 `VLMbench` 的 `Dataset` 用来处理数据集，修改后的文件是 `VLDataloader_renjie.py`

## test 

测试文件在 `vlm/scripts/vlm_test.py`。