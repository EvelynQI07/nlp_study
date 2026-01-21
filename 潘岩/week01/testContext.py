# conda activate py312
import jieba
import sklearn
import torch
print("jieba版本：", jieba.__version__)
print("sklearn版本：", sklearn.__version__)
print("PyTorch版本：", torch.__version__)
print("CUDA是否可用：", torch.cuda.is_available())
