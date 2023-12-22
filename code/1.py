# 测试GPU版本 2023.12.13  cuda10.1版 torch1.7   但是ocr等需要>>1.8
#import torch
#print(torch.__version__)

import torch
print(torch.cuda.is_available())