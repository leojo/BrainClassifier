from extract_features import *

extractHistograms("../data/set_train",maxValue = 4000, nBins = 45, nPartitions = 9)
extractHistograms("../data/set_test",maxValue = 4000, nBins = 45, nPartitions = 9)

