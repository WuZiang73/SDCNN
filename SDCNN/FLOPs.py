import torch
from thop import profile
from model.dsrnet import Net

if __name__ == "__main__":
    scale = 4
    model = Net(scale=scale)
    model = model.cuda()  
    model.eval()
    input_tensor = torch.randn(1, 3, 256, 256).cuda()  
    with torch.no_grad():
        flops, params = profile(model, inputs=(input_tensor, scale))
    print(f"FLOPs: {flops / 1e9:.2f} GFLOPs, Params: {params / 1e6:.4f} M")