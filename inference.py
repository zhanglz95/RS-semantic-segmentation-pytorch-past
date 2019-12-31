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

	dictPath = Path("./saved/deepglobe_road_rgb_dinknet_res34/12-27-19:13/DinkNet34-best_loss.pth")
	model.load_state_dict(torch.load(dictPath))
	model.to(device)

	srcDir = Path("./data/deepglobe/road/valid/images")
	dstDir = Path("./demo/deepglobe_valid_base_dice_div_4_tta_vote")
	if not dstDir.exists():
		dstDir.mkdir()
	imgList = srcDir.glob("*.jpg")
	for imgpath in imgList:
		img = Image.open(imgpath)

		img_HorizontalFlip = np.array(img.transpose(Image.FLIP_TOP_BOTTOM))
		img_VerticalFlip = np.array(img.transpose(Image.FLIP_LEFT_RIGHT))
		img_90 = np.array(img.rotate(90))
		img_180 = np.array(img.rotate(180))
		img_270 = np.array(img.rotate(270))
		img = np.array(img)

		img_HorizontalFlip = F.to_tensor(img_HorizontalFlip).unsqueeze(0)
		img_VerticalFlip = F.to_tensor(img_VerticalFlip).unsqueeze(0)
		img_90 = F.to_tensor(img_90).unsqueeze(0)
		img_180 = F.to_tensor(img_180).unsqueeze(0)
		img_270 = F.to_tensor(img_270).unsqueeze(0)
		img = F.to_tensor(img).unsqueeze(0)

		img1 = torch.cat(
			(
				img, 
				img_HorizontalFlip, 
				img_VerticalFlip
			), 
			0).to(device)

		img2 = torch.cat(
			(
				img_90, 
				img_180, 
				img_270
			), 
			0).to(device)

		output1 = model(img1)
		output2 = model(img2)

		output1 = NF.softmax(output1, dim=1)
		output2 = NF.softmax(output2, dim=1)

		output1 = torch.argmax(output1, dim=1)
		output2 = torch.argmax(output2, dim=1)

		output = output1[0].cpu().numpy()
		output_HorizontalFlip = output1[1].cpu().numpy()
		output_VerticalFlip = output1[2].cpu().numpy()
		output_90 = output2[0].cpu().numpy()
		output_180 = output2[1].cpu().numpy()
		output_270 = output2[2].cpu().numpy()

		output = output[:, :, np.newaxis]
		output_HorizontalFlip = output_HorizontalFlip[:, :, np.newaxis]
		output_VerticalFlip = output_VerticalFlip[:, :, np.newaxis]
		output_90 = output_90[:, :, np.newaxis]
		output_180 = output_180[:, :, np.newaxis]
		output_270 = output_270[:, :, np.newaxis]

		output = np.concatenate((output, output, output), 2)
		output_HorizontalFlip = np.concatenate((output_HorizontalFlip, output_HorizontalFlip, output_HorizontalFlip), 2)
		output_VerticalFlip = np.concatenate((output_VerticalFlip, output_VerticalFlip, output_VerticalFlip), 2)
		output_90 = np.concatenate((output_90, output_90, output_90), 2)
		output_180 = np.concatenate((output_180, output_180, output_180), 2)
		output_270 = np.concatenate((output_270, output_270, output_270), 2)

		output = np.asarray(output)
		output_HorizontalFlip = np.asarray(output_HorizontalFlip)
		output_VerticalFlip = np.asarray(output_VerticalFlip)
		output_90 = np.asarray(output_90)
		output_180 = np.asarray(output_180)
		output_270 = np.asarray(output_270)

		output_HorizontalFlip = np.flip(output_HorizontalFlip, 0)
		output_VerticalFlip = np.flip(output_VerticalFlip, 1)
		output_90 = np.rot90(output_90, 3)
		output_180 = np.rot90(output_180, 2)
		output_270 = np.rot90(output_270, 1)

		output = np.array([
			output, 
			output_HorizontalFlip,
			output_VerticalFlip,
			output_90,
			output_180,
			output_270
			])

		# output = np.max(output, 0)
		# output = np.min(output, 0)
		output = np.sum(output, 0)
		fg = output >= 3
		output = np.array(fg, dtype=np.uint8) * 255

		output = Image.fromarray(output)
		output.save(dstDir / imgpath.name.replace("_sat.jpg", "_mask.png"))



