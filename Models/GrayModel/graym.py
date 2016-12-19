

useSimpleData = True
plotImage = True
modelMaker = GrayModel()

grayModel = modelMaker.makeModel(useSimpleData, plotImage)




class GrayModel():

	# Initialized with the models we want to train with and the voting
	# classifier that will be used to combine those models
	def __init__():
		self.names = [
			"Linear SVM",
			"Poly SVM", 
			"Sigmoid SVM", 
			"Random Forest",
			"Gaussian Process",
			"AdaBoost"
		],
		self.models = [
			SVC(kernel="linear", C=1.0, probability=True, class_weight={0: 10}),
			SVC(kernel="poly", C=1.0, probability=True, class_weight={0: 10}),
			SVC(kernel="sigmoid", C=1.0, probability=True, class_weight={0: 10}),
			RandomForestClassifier(max_depth=5, max_features=1),
			GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
			AdaBoostClassifier(base_estimator=GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True)),
		],
		self.voter = VotingClassifier(self.models.items(), voting='soft') 
	
	# ======
	# MAIN FUNCTION:
	#	Extracts the features from the data
	#   Preprocesses the features
	#   Trains the models
	# 	Performs Cross Val scorings for all models
	#	Combines models and creates a submission file
	#
	# 		Optionally, trains model on 2D and plots an
	#		image for better visualisation

	def makeModel(simple=True, plotImage=True):
		if(simple):
			data = extractFeatures("../../data/set_train" ,True)
		else 
			data = extractFeatures("../../data/set_train", False)


		data = extractFeatures(BasicModel.getImages())
		targets = BasicModel.getTargets()
		preprocessedData = preprocessData(data)

		mdl = trainModel(data)
		return mdl

	def trainModel():

	def plotImages():

	def preprocessData(data):




	def crossvalModel(data):

	def extractFeatures(dataDir, simple=True):
		if(simple) data = extract2DSlices("../data/set_train", 450, 800, 8)
		else data = extract3DFull("../data/set_test", 450, 800, 8)
		return data
		# Maybe reduce resolution for images?



class ColorFeatureExtraction(BasicFeatureExtraction):
	def __init__():
		self.featuresDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

	# Simple version
	def extract2DSlices(imgDir, minColor, maxColor, nPartitions=1):
		# If we have extracted these exact features before and saved them,
		# we can return them right away
		preloadedData = getIfExists(imgDir, minColor, maxColor, nPartitions=1)
		if(preloadedData) return preloadedData

		# Otherwise start extracting:
		allColoredZones = []
		# Fetch all directory listings of set_train and sort them on the image number
		allImageSrc = BasicFeatureExtraction.getImages()
		n_samples = len(allImageSrc);
		print "Found "+str(n_samples)+" images!"
		print "Preparing the data"
		printProgress(0, n_samples)
		for i in range(0,n_samples):
			img = nib.load(allImageSrc[i])
			imgData = img.get_data();
			imgDataDisected = imgData[20:148, 35:163, 88, 0]

			colZones = np.asarray([[[0]*nPartitions]*nPartitions]*nPartitions)
			# Size should be same for all dimensions, imgData should
			# have same dimensions for x, y, z all such that they can be
			# divided by nPartitions
			for x in range(imgDataDisected.shape[0]):
				for y in range(imgDataDisected.shape[1]):
					#for z in range(imgDataDisected.shape[2]):
						val = imgDataDisected[x][y]#[z]
						partX = int((x*nPartitions)/imgDataDisected.shape[0])
						partY = int((y*nPartitions)/imgDataDisected.shape[1])
						#partZ = int((z*nPartitions)/imgDataDisected.shape[2])
						if val <= maxColor and val >= minColor:
							colZones[partX][partY] += 1

			allColoredZones.append(colZones.flatten().tolist())
			printProgress(i+1, n_samples)		
			

		print "\nStoring the features in "+outputFileName
		output = open(outputFileName,"wb")
		pickle.dump(allColoredZones,output)
		output.close()
		print "Done"
		return allColoredZones

	# Normal version
	def extract3DFull(imgDir, minColor, maxColor, nPartitions=1):
		return


	def getIfExists(imgDir, minColor, maxColor, nPartitions=1):
		# This is the cache for the feature, used to make sure we do the heavy computations more often than neccesary
		outputFileName = os.path.join(self.featuresDir,"coloredzones_"+str(nPartitions)+"_"+str(minColor)+"_"+str(maxColor)+"_"+imgDir.replace(os.sep,"-")+".feature")
		if os.path.isfile(outputFileName):
			save = open(outputFileName,'rb')
			loadedData = pickle.load(save)
			save.close()
			return loadedData

		return False

	