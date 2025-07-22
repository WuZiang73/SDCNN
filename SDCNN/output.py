import torch
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from model.dsrnet import Net

scale = 4
print(f"Using scale: {scale}")

try:
    model = Net(scale=scale)
    print("Model instantiated.")
    model.load_state_dict(torch.load('/data1/wza/DSRNet-main/DSRNet/checkpoint/dsrnet_x4_880000.pth'), strict=False)
    print("Model weights loaded.")
    model.eval()
    model.cuda()
    print("Model set to eval and moved to CUDA.")

    img_path = '/data1/wza/DSRNet-main/DSRNet/CanPic/BloodImage_00000.jpg'
    img = Image.open(img_path).convert('RGB')
    print(f"Input image loaded: {img_path}, size: {img.size}, mode: {img.mode}")

    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0).cuda()
    print(f"Input tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}, min: {img_tensor.min().item()}, max: {img_tensor.max().item()}")

    with torch.no_grad():
        output = model(img_tensor, scale)
    print(f"Raw output tensor shape: {output.shape}, dtype: {output.dtype}")

    output = output.squeeze(0).cpu().clamp(0, 1)
    print(f"Output tensor after squeeze and clamp: shape {output.shape}, min {output.min().item()}, max {output.max().item()}")

    output_np = (output.numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
    print(f"Output numpy array shape: {output_np.shape}, dtype: {output_np.dtype}, min: {output_np.min()}, max: {output_np.max()}")

    # 确保通道顺序是 RGB
    # 如果模型输出是 BGR 格式，取消下一行的注释
    # output_np = output_np[:, :, ::-1]

    output_img = Image.fromarray(output_np)
    output_img.save('can_sr_result03.png')
    print("Super-resolution result saved")

except Exception as e:
    print(f"An error occurred: {e}")