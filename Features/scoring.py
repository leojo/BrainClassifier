def hammingLoss(y_pred,y):
	'''
	Returns the average normalized hamming distance from the prediction
	to the actual value
	'''
	D = y.shape[0]
	L = y.shape[1]
	sum = 0.0
	for i in range(D):
		for j in range(L):
			diff = abs(y_pred[i][j]-y[i][j])
			sum += diff
	sum /= L*D
	return sum

def partialHammingLoss(y_pred,y):
	D = y.shape[0]
	sum = 0.0
	for i in range(D):
		diff = abs(y_pred[i]-y[i])
		sum += diff
	sum /= D
	return sum