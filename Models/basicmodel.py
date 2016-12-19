import csv

class BasicModel:

	def getTargets():
		# Get the targets
		with open('../data/targets.csv', 'rb') as f:
	    reader = csv.reader(f)
	    targets = list(reader)
	    return targets

	def getImages(imgPath): 
		# Fetch all directory listings of set_train and sort them on the image number
		allImageSrc = sorted(glob.glob(imgPath), key=extractImgNumber)
		return allImageSrc

	def extractImgNumber(imgPath):
		imgName = imgPath.split(os.sep)[-1]
		imgNum = int(imgName.split('_')[-1][:-4])
		return imgNum



