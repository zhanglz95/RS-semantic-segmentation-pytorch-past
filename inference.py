import torch
from torch.nn import functional as NF
from torchvision.transforms import functional as F
import models as M
from PIL import Image
from pathlib import Path
import numpy as np

if __name__ == "__main__":
	model = getattr(M, "DinkNet34")()
	device = torch.device("cuda:0")

	dictPath = Path("./saved/visiontek_road_rgb_dinknet_res34/12-16-21:57/DinkNet34-best_iou.pth")
	model.load_state_dict(torch.load(dictPath))
	model.to(device)

	srcDir = Path("./demo/images")
	dstDir = Path("./demo/outputs")

	imgList = srcDir.glob("*.jpg")
	for imgpath in imgList:
		img = np.array(Image.open(imgpath))
		img = F.to_tensor(img)
		img = torch.unsqueeze(img, 0).to(device)
		output = model(img)
		output = NF.softmax(output, dim=1)
		output = torch.argmax(output, dim=1)
		output = output[0]
		output = output.cpu().numpy() * 255
		output = np.asarray(output, dtype=np.uint8)
		output = Image.fromarray(output)
		output.save(dstDir / imgpath.name)



