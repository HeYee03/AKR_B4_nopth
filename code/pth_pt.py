import torch
import os
from net import vgg16
device = torch.device("cpu") #device=torch.device('cuda'if torch.cuda.is_available() else "cpu")
model=vgg16()
model.load_state_dict(torch.load("./AKR_B4_12_1915.pth"),device)
model.eval()
example=torch.ones(1,3,224,224)
master=torch.jit.trace(model,example)

master.save("ptAKR_B4_12_1915.pt")
