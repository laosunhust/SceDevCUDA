#include "SceNodes.h"

__constant__ double sceInterPara[5];
__constant__ double sceIntraPara[4];
double sceInterParaCPU[5];
double sceIntraParaCPU[4];

// This template method expands an input sequence by
// replicating each element a variable number of times. For example,
//
//   expand([2,2,2],[A,B,C]) -> [A,A,B,B,C,C]
//   expand([3,0,1],[A,B,C]) -> [A,A,A,C]
//   expand([1,3,2],[A,B,C]) -> [A,B,B,B,C,C]
//
// The element counts are assumed to be non-negative integers
template<typename InputIterator1, typename InputIterator2,
		typename OutputIterator>
OutputIterator expand(InputIterator1 first1, InputIterator1 last1,
		InputIterator2 first2, OutputIterator output) {
	typedef typename thrust::iterator_difference<InputIterator1>::type difference_type;

	difference_type input_size = thrust::distance(first1, last1);
	difference_type output_size = thrust::reduce(first1, last1);

	// scan the counts to obtain output offsets for each input element
	thrust::device_vector<difference_type> output_offsets(input_size, 0);
	thrust::exclusive_scan(first1, last1, output_offsets.begin());

	// scatter the nonzero counts into their corresponding output positions
	thrust::device_vector<difference_type> output_indices(output_size, 0);
	thrust::scatter_if(thrust::counting_iterator < difference_type > (0),
			thrust::counting_iterator < difference_type > (input_size),
			output_offsets.begin(), first1, output_indices.begin());

	// compute max-scan over the output indices, filling in the holes
	thrust::inclusive_scan(output_indices.begin(), output_indices.end(),
			output_indices.begin(), thrust::maximum<difference_type>());

	// gather input values according to index array (output = first2[output_indices])
	OutputIterator output_end = output;
	thrust::advance(output_end, output_size);
	thrust::gather(output_indices.begin(), output_indices.end(), first2,
			output);

	// return output + output_size
	thrust::advance(output, output_size);
	return output;
}

SceNodes::SceNodes(uint maxTotalCellCount, uint maxNodeInCell) {
	maxCellCount = maxTotalCellCount;
	maxNodeOfOneCell = maxNodeInCell;

	currentActiveCellCount = 0;
	// for now it is initialized as 3.
	maxNodePerECM = 3;
	// for now it is initialized as 0.
	maxECMCount = 0;
	maxTotalECMNodeCount = maxECMCount * maxNodePerECM;
	currentActiveECM = 0;

	// will need to change this value after we have more detail about ECM
	maxTotalCellNodeCount = maxTotalCellCount * maxNodeOfOneCell;

	//cellRanks.resize(maxTotalNodeCount);
	//nodeRanks.resize(maxTotalNodeCount);
	uint maxTotalNodeCount = maxTotalCellNodeCount + maxTotalECMNodeCount;
	nodeIsActive.resize(maxTotalNodeCount);
	nodeLocX.resize(maxTotalNodeCount);
	nodeLocY.resize(maxTotalNodeCount);
	nodeLocZ.resize(maxTotalNodeCount);
	nodeVelX.resize(maxTotalNodeCount);
	nodeVelY.resize(maxTotalNodeCount);
	nodeVelZ.resize(maxTotalNodeCount);
	//thrust::counting_iterator<uint> countingBegin(0);
	//thrust::counting_iterator<uint> countingEnd(maxTotalNodeCount);

	// following block of code is depreciated due to a simplification of node global notation
	//thrust::transform(
	//		make_zip_iterator(
	//				make_tuple(cellRanks.begin(), nodeRanks.begin(),
	//						countingBegin)),
	//		make_zip_iterator(
	//				make_tuple(cellRanks.end(), nodeRanks.end(), countingEnd)),
	//		make_zip_iterator(
	//				make_tuple(cellRanks.begin(), nodeRanks.begin(),
	//						countingBegin)), InitFunctor(maxCellCount));

	ConfigParser parser;
	std::string configFileName = "sceCell.config";
	// globalConfigVars should be a global variable and defined in the main function.
	globalConfigVars = parser.parseConfigFile(configFileName);
	static const double U0 =
			globalConfigVars.getConfigValue("InterCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_U0_DivFactor").toDouble();
	static const double V0 =
			globalConfigVars.getConfigValue("InterCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_V0_DivFactor").toDouble();
	static const double k1 =
			globalConfigVars.getConfigValue("InterCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k1_DivFactor").toDouble();
	static const double k2 =
			globalConfigVars.getConfigValue("InterCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k2_DivFactor").toDouble();
	static const double interLinkEffectiveRange =
			globalConfigVars.getConfigValue("InterCellLinkBreakRange").toDouble();

	sceInterParaCPU[0] = U0;
	sceInterParaCPU[1] = V0;
	sceInterParaCPU[2] = k1;
	sceInterParaCPU[3] = k2;
	sceInterParaCPU[4] = interLinkEffectiveRange;

	static const double U0_Intra =
			globalConfigVars.getConfigValue("IntraCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_U0_DivFactor").toDouble();
	static const double V0_Intra =
			globalConfigVars.getConfigValue("IntraCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_V0_DivFactor").toDouble();
	static const double k1_Intra =
			globalConfigVars.getConfigValue("IntraCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k1_DivFactor").toDouble();
	static const double k2_Intra =
			globalConfigVars.getConfigValue("IntraCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k2_DivFactor").toDouble();
	sceIntraParaCPU[0] = U0_Intra;
	sceIntraParaCPU[1] = V0_Intra;
	sceIntraParaCPU[2] = k1_Intra;
	sceIntraParaCPU[3] = k2_Intra;

	cudaMemcpyToSymbol(sceInterPara, sceInterParaCPU, 5 * sizeof(double));
	cudaMemcpyToSymbol(sceIntraPara, sceIntraParaCPU, 4 * sizeof(double));
	//std::cout << "U0 :" << sceIntraParaCPU[0] << std::endl;
	//std::cout << "V0 :" << sceIntraParaCPU[1] << std::endl;
	//std::cout << "k1 :" << sceIntraParaCPU[2] << std::endl;
	//std::cout << "k2 :" << sceIntraParaCPU[3] << std::endl;
	//std::cout << "Force parameters initialized" << std::endl;

	//double constantsCopiesFromDevice[4];
	//cudaMemcpyFromSymbol(constantsCopiesFromDevice, sceInterPara,
	//		4 * sizeof(double));
	//std::cout << "U0 From Device:" << constantsCopiesFromDevice[0] << std::endl;
	//std::cout << "V0 From Device:" << constantsCopiesFromDevice[1] << std::endl;
	//std::cout << "k1 From Device:" << constantsCopiesFromDevice[2] << std::endl;
	//std::cout << "k2 From Device:" << constantsCopiesFromDevice[3] << std::endl;
}

SceNodes::SceNodes(uint maxTotalCellCount, uint maxNodeInCell,
		uint maxTotalECMCount, uint maxNodeInECM) {
	maxCellCount = maxTotalCellCount;
	maxNodeOfOneCell = maxNodeInCell;
	maxNodePerECM = maxNodeInECM;
	maxECMCount = maxTotalECMCount;
	currentActiveCellCount = 0;
	// for now it is initialized as 3.
	maxNodePerECM = 3;
	// for now it is initialized as 0.
	maxECMCount = 0;
	maxTotalECMNodeCount = maxECMCount * maxNodePerECM;
	currentActiveECM = 0;

	// will need to change this value after we have more detail about ECM
	maxTotalCellNodeCount = maxTotalCellCount * maxNodeOfOneCell;

	//cellRanks.resize(maxTotalNodeCount);
	//nodeRanks.resize(maxTotalNodeCount);
	uint maxTotalNodeCount = maxTotalCellNodeCount + maxTotalECMNodeCount;
	nodeIsActive.resize(maxTotalNodeCount);
	nodeLocX.resize(maxTotalNodeCount);
	nodeLocY.resize(maxTotalNodeCount);
	nodeLocZ.resize(maxTotalNodeCount);
	nodeVelX.resize(maxTotalNodeCount);
	nodeVelY.resize(maxTotalNodeCount);
	nodeVelZ.resize(maxTotalNodeCount);
	//thrust::counting_iterator<uint> countingBegin(0);
	//thrust::counting_iterator<uint> countingEnd(maxTotalNodeCount);

	// following block of code is depreciated due to a simplification of node global notation
	//thrust::transform(
	//		make_zip_iterator(
	//				make_tuple(cellRanks.begin(), nodeRanks.begin(),
	//						countingBegin)),
	//		make_zip_iterator(
	//				make_tuple(cellRanks.end(), nodeRanks.end(), countingEnd)),
	//		make_zip_iterator(
	//				make_tuple(cellRanks.begin(), nodeRanks.begin(),
	//						countingBegin)), InitFunctor(maxCellCount));

	ConfigParser parser;
	std::string configFileName = "sceCell.config";
	// globalConfigVars should be a global variable and defined in the main function.
	globalConfigVars = parser.parseConfigFile(configFileName);
	static const double U0 =
			globalConfigVars.getConfigValue("InterCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_U0_DivFactor").toDouble();
	static const double V0 =
			globalConfigVars.getConfigValue("InterCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_V0_DivFactor").toDouble();
	static const double k1 =
			globalConfigVars.getConfigValue("InterCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k1_DivFactor").toDouble();
	static const double k2 =
			globalConfigVars.getConfigValue("InterCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k2_DivFactor").toDouble();
	static const double interLinkEffectiveRange =
			globalConfigVars.getConfigValue("InterCellLinkBreakRange").toDouble();

	sceInterParaCPU[0] = U0;
	sceInterParaCPU[1] = V0;
	sceInterParaCPU[2] = k1;
	sceInterParaCPU[3] = k2;
	sceInterParaCPU[4] = interLinkEffectiveRange;

	static const double U0_Intra =
			globalConfigVars.getConfigValue("IntraCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_U0_DivFactor").toDouble();
	static const double V0_Intra =
			globalConfigVars.getConfigValue("IntraCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_V0_DivFactor").toDouble();
	static const double k1_Intra =
			globalConfigVars.getConfigValue("IntraCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k1_DivFactor").toDouble();
	static const double k2_Intra =
			globalConfigVars.getConfigValue("IntraCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k2_DivFactor").toDouble();
	sceIntraParaCPU[0] = U0_Intra;
	sceIntraParaCPU[1] = V0_Intra;
	sceIntraParaCPU[2] = k1_Intra;
	sceIntraParaCPU[3] = k2_Intra;

	cudaMemcpyToSymbol(sceInterPara, sceInterParaCPU, 5 * sizeof(double));
	cudaMemcpyToSymbol(sceIntraPara, sceIntraParaCPU, 4 * sizeof(double));
}

void SceNodes::addNewlyDividedCells(
		thrust::device_vector<double> &nodeLocXNewCell,
		thrust::device_vector<double> &nodeLocYNewCell,
		thrust::device_vector<double> &nodeLocZNewCell,
		thrust::device_vector<bool> &nodeIsActiveNewCell) {

	uint shiftSize = nodeLocXNewCell.size();
	assert(shiftSize % maxNodeOfOneCell == 0);
	uint addCellCount = shiftSize / maxNodeOfOneCell;

	uint shiftStartPos = currentActiveCellCount * maxNodeOfOneCell;
	uint shiftEndPos = shiftStartPos + currentActiveECM * maxNodePerECM;
	uint ECMStartPos = shiftStartPos + shiftSize;
	// reason using this tmp vector is that GPU copying does not guarantee copying sequence.
	// will cause undefined behavior if copy directly.
	//std::cout << "shift start position = " << shiftStartPos << ", end pos = "
	//		<< shiftEndPos << std::endl;
	thrust::device_vector<double> tmpPosXECM(nodeLocX.begin() + shiftStartPos,
			nodeLocX.begin() + shiftEndPos);
	thrust::device_vector<double> tmpPosYECM(nodeLocY.begin() + shiftStartPos,
			nodeLocY.begin() + shiftEndPos);
	thrust::device_vector<double> tmpPosZECM(nodeLocZ.begin() + shiftStartPos,
			nodeLocZ.begin() + shiftEndPos);
	thrust::device_vector<bool> tmpIsActive(
			nodeIsActive.begin() + shiftStartPos,
			nodeIsActive.begin() + shiftEndPos);

	thrust::copy(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocXNewCell.begin(),
							nodeLocYNewCell.begin(), nodeLocZNewCell.begin(),
							nodeIsActiveNewCell.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocXNewCell.end(),
							nodeLocYNewCell.end(), nodeLocZNewCell.end(),
							nodeIsActiveNewCell.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin(), nodeIsActive.begin()))
					+ shiftStartPos);

	thrust::copy(
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpPosXECM.begin(), tmpPosYECM.begin(),
							tmpPosZECM.begin(), tmpIsActive.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpPosXECM.end(), tmpPosYECM.end(),
							tmpPosZECM.end(), tmpIsActive.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin(), nodeIsActive.begin()))
					+ ECMStartPos);
	currentActiveCellCount = currentActiveCellCount + addCellCount;
}

void SceNodes::buildBuckets2D(double minX, double maxX, double minY,
		double maxY, double bucketSize) {
	int totalActiveNodes = currentActiveCellCount * maxNodeOfOneCell;
//std::cout << "total number of active nodes:" << totalActiveNodes
//		<< std::endl;
	bucketKeys.resize(totalActiveNodes);
	bucketValues.resize(totalActiveNodes);
	thrust::counting_iterator<uint> countingIterBegin(0);
	thrust::counting_iterator<uint> countingIterEnd(totalActiveNodes);

// takes counting iterator and coordinates
// return tuple of keys and values

// transform the points to their bucket indices
	thrust::transform(
			make_zip_iterator(
					make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin(), nodeIsActive.begin(),
							countingIterBegin)),
			make_zip_iterator(
					make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin(), nodeIsActive.begin(),
							countingIterBegin)) + totalActiveNodes,
			make_zip_iterator(
					make_tuple(bucketKeys.begin(), bucketValues.begin())),
			pointToBucketIndex2D(minX, maxX, minY, maxY, bucketSize));

// sort the points by their bucket index
	thrust::sort_by_key(bucketKeys.begin(), bucketKeys.end(),
			bucketValues.begin());
// for those nodes that are inactive, we key value of UINT_MAX will be returned.
// we need to removed those keys along with their values.
	int numberOfOutOfRange = thrust::count(bucketKeys.begin(), bucketKeys.end(),
			UINT_MAX);
	bucketKeys.erase(bucketKeys.end() - numberOfOutOfRange, bucketKeys.end());
	bucketValues.erase(bucketValues.end() - numberOfOutOfRange,
			bucketValues.end());
}

__device__
double computeDist(double &xPos, double &yPos, double &zPos, double &xPos2,
		double &yPos2, double &zPos2) {
	return sqrt(
			(xPos - xPos2) * (xPos - xPos2) + (yPos - yPos2) * (yPos - yPos2)
					+ (zPos - zPos2) * (zPos - zPos2));
}

__device__
void calculateAndAddECMForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes) {

}

__device__
void calculateAndAddInterForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes) {
	double linkLength = computeDist(xPos, yPos, zPos, xPos2, yPos2, zPos2);
	double forceValue = 0;
	if (linkLength > sceInterPara[4]) {
		forceValue = 0;
	} else {
		forceValue = -sceInterPara[0] / sceInterPara[2]
				* exp(-linkLength / sceInterPara[2])
				+ sceInterPara[1] / sceInterPara[3]
						* exp(-linkLength / sceInterPara[3]);
	}
	xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
	yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
	zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
}

__device__
void calculateAndAddIntraForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes) {
	double linkLength = computeDist(xPos, yPos, zPos, xPos2, yPos2, zPos2);
	double forceValue = -sceIntraPara[0] / sceIntraPara[2]
			* exp(-linkLength / sceIntraPara[2])
			+ sceIntraPara[1] / sceIntraPara[3]
					* exp(-linkLength / sceIntraPara[3]);
	xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
	yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
	zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
//xRes = xRes + (xPos2 - xPos);
//yRes = yRes + (yPos2 - yPos);
//zRes = zRes + (zPos2 - zPos);
//xRes = xRes + 1.0;
//yRes = yRes + 2.0;
//zRes = zRes + 3.0;
//xRes = xRes + forceValue;
//yRes = yRes + forceValue;
//zRes = zRes + forceValue;
//xRes = xRes + xPos;
//yRes = yRes + yPos;
//zRes = zRes + zPos;
}

__device__ bool bothNodesCellNode(uint nodeGlobalRank1, uint nodeGlobalRank2,
		uint cellNodesThreshold) {
	if (nodeGlobalRank1 < cellNodesThreshold
			&& nodeGlobalRank2 < cellNodesThreshold) {
		return true;
	} else {
		return false;
	}
}

__device__ bool isSameCell(uint nodeGlobalRank1, uint nodeGlobalRank2,
		uint nodeCountPerCell) {
	if (nodeGlobalRank1 / nodeCountPerCell
			== nodeGlobalRank2 / nodeCountPerCell) {
		return true;
	} else {
		return false;
	}
}

void SceNodes::extendBuckets2D(uint numOfBucketsInXDim,
		uint numOfBucketsInYDim) {
	static const uint extensionFactor2D = 9;
	uint valuesCount = bucketValues.size();
	bucketKeysExpanded.resize(valuesCount * extensionFactor2D);
	bucketValuesIncludingNeighbor.resize(valuesCount * extensionFactor2D);
	thrust::constant_iterator<uint> first(extensionFactor2D);
	thrust::constant_iterator<uint> last = first + valuesCount;

	expand(first, last,
			make_zip_iterator(
					make_tuple(bucketKeys.begin(), bucketValues.begin())),
			make_zip_iterator(
					make_tuple(bucketKeysExpanded.begin(),
							bucketValuesIncludingNeighbor.begin())));

	thrust::counting_iterator<uint> countingBegin(0);
	thrust::counting_iterator<uint> countingEnd = countingBegin
			+ valuesCount * extensionFactor2D;

//std::cout << "number of values for array holding extended value= "
//		<< valuesCount * extensionFactor2D << std::endl;
//thrust::for_each(
//		thrust::make_zip_iterator(
//				make_tuple(bucketKeysExpanded.begin(), countingBegin)),
//		thrust::make_zip_iterator(
//				make_tuple(bucketKeysExpanded.end(), countingEnd)),
//		NeighborFunctor2D(numOfBucketsInXDim, numOfBucketsInYDim));

	thrust::transform(
			make_zip_iterator(
					make_tuple(bucketKeysExpanded.begin(), countingBegin)),
			make_zip_iterator(
					make_tuple(bucketKeysExpanded.end(), countingEnd)),
			make_zip_iterator(
					make_tuple(bucketKeysExpanded.begin(), countingBegin)),
			NeighborFunctor2D(numOfBucketsInXDim, numOfBucketsInYDim));

	int numberOfOutOfRange = thrust::count(bucketKeysExpanded.begin(),
			bucketKeysExpanded.end(), UINT_MAX);
//std::cout << "number out of range = " << numberOfOutOfRange << std::endl;
	int sizeBeforeShrink = bucketKeysExpanded.size();
	int numberInsideRange = sizeBeforeShrink - numberOfOutOfRange;
	thrust::sort_by_key(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
			bucketValuesIncludingNeighbor.begin());
	bucketKeysExpanded.erase(bucketKeysExpanded.begin() + numberInsideRange,
			bucketKeysExpanded.end());
	bucketValuesIncludingNeighbor.erase(
			bucketValuesIncludingNeighbor.begin() + numberInsideRange,
			bucketValuesIncludingNeighbor.end());
}
void SceNodes::applySceForces(uint numOfBucketsInXDim,
		uint numOfBucketsInYDim) {
	uint totalBucketCount = numOfBucketsInXDim * numOfBucketsInYDim;
	thrust::device_vector<unsigned int> keyBegin(totalBucketCount);
	thrust::device_vector<unsigned int> keyEnd(totalBucketCount);
	thrust::counting_iterator<unsigned int> search_begin(0);
	thrust::lower_bound(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
			search_begin, search_begin + totalBucketCount, keyBegin.begin());
	thrust::upper_bound(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
			search_begin, search_begin + totalBucketCount, keyEnd.begin());
	uint* valueAddress = thrust::raw_pointer_cast(
			&bucketValuesIncludingNeighbor[0]);

	thrust::host_vector<uint> keyBeginFromGPU = keyBegin;
	thrust::host_vector<uint> keyEndFromGPU = keyEnd;
	thrust::host_vector<uint> bucketKeysFromGPU = bucketKeys;
//for (uint i = 0; i < keyBeginFromGPU.size(); i++) {
//	std::cout << "key begin: " << keyBeginFromGPU[i] << "key end:"
//			<< keyEndFromGPU[i] << std::endl;
//}

//for (uint i = 0; i < bucketKeysFromGPU.size(); i++) {
//	std::cout << "bucket key: " << bucketKeysFromGPU[i] << std::endl;
//}

	double* nodeLocXAddress = thrust::raw_pointer_cast(&nodeLocX[0]);
	double* nodeLocYAddress = thrust::raw_pointer_cast(&nodeLocY[0]);
	double* nodeLocZAddress = thrust::raw_pointer_cast(&nodeLocZ[0]);
	thrust::transform(
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator(keyBegin.begin(),
									bucketKeys.begin()),
							make_permutation_iterator(keyEnd.begin(),
									bucketKeys.begin()), bucketValues.begin(),
							make_permutation_iterator(nodeLocX.begin(),
									bucketValues.begin()),
							make_permutation_iterator(nodeLocY.begin(),
									bucketValues.begin()),
							make_permutation_iterator(nodeLocZ.begin(),
									bucketValues.begin()))),
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator(keyBegin.begin(),
									bucketKeys.end()),
							make_permutation_iterator(keyEnd.begin(),
									bucketKeys.end()), bucketValues.end(),
							make_permutation_iterator(nodeLocX.begin(),
									bucketValues.end()),
							make_permutation_iterator(nodeLocY.begin(),
									bucketValues.end()),
							make_permutation_iterator(nodeLocZ.begin(),
									bucketValues.end()))),
			make_zip_iterator(
					make_tuple(
							make_permutation_iterator(nodeVelX.begin(),
									bucketValues.begin()),
							make_permutation_iterator(nodeVelY.begin(),
									bucketValues.begin()),
							make_permutation_iterator(nodeVelZ.begin(),
									bucketValues.begin()))),
			AddSceForce(valueAddress, nodeLocXAddress, nodeLocYAddress,
					nodeLocZAddress, maxTotalCellNodeCount, maxNodeOfOneCell,
					maxNodePerECM));
}

void SceNodes::calculateAndApplySceForces(double minX, double maxX, double minY,
		double maxY, double bucketSize) {
	const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
	const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
	buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);
	applySceForces(numberOfBucketsInXDim, numberOfBucketsInYDim);
}
/*
 void SceNodes::buildPairsFromBucketsAndExtendedBuckets(uint numOfBucketsInXDim,
 uint numOfBucketsInYDim) {
 uint totalBucketCount = numOfBucketsInXDim * numOfBucketsInYDim;
 // "+1" to avoid potential bug when all buckets are occupied
 thrust::device_vector<uint> tmpBucketKeys(totalBucketCount + 1, UINT_MAX);
 thrust::device_vector<uint> tmpBucketValues(totalBucketCount + 1, UINT_MAX);

 thrust::device_vector<uint> tmpExtendedBucketKeys(totalBucketCount + 1,
 UINT_MAX);
 thrust::device_vector<uint> tmpExtendedBucketValues(totalBucketCount + 1,
 UINT_MAX);

 //thrust::counting_iterator<uint> countBegin(0);
 //thrust::counting_iterator<uint> countEnd = countBegin + bucketKeys.size();
 thrust::reduce_by_key(bucketKeys.begin(), bucketKeys.end(),
 thrust::constant_iterator < uint > (1), tmpBucketKeys.begin(),
 tmpBucketValues.begin());
 // keep those unique values. reason of "-1" is that we have all numbers initialized as UINT_MAX
 tmpBucketKeys.erase(
 thrust::unique(tmpBucketKeys.begin(), tmpBucketKeys.end()) - 1,
 tmpBucketKeys.end());
 int keySize = tmpBucketKeys.size();
 tmpBucketValues.erase(tmpBucketValues.begin() + keySize,
 tmpBucketValues.end());

 thrust::reduce_by_key(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
 thrust::constant_iterator < uint > (1),
 tmpExtendedBucketKeys.begin(), tmpExtendedBucketValues.begin());
 // keep those unique values. reason of "-1" is that we have all numbers initialized as UINT_MAX
 tmpExtendedBucketKeys.erase(
 thrust::unique(tmpExtendedBucketKeys.begin(),
 tmpExtendedBucketKeys.end()) - 1,
 tmpExtendedBucketKeys.end());
 int extendedKeySize = tmpExtendedBucketKeys.size();
 tmpExtendedBucketValues.erase(
 tmpExtendedBucketValues.begin() + extendedKeySize,
 tmpExtendedBucketValues.end());

 thrust::host_vector<uint> hostKeysDEBUG = tmpBucketKeys;
 thrust::host_vector<uint> hostValuesDEBUG = tmpBucketValues;
 for (uint i = 0; i < hostKeysDEBUG.size(); i++) {
 std::cout << "key:" << hostKeysDEBUG[i] << ", value:"
 << hostValuesDEBUG[i] << std::endl;
 }

 std::cout << "begin output extended key-value pair" << std::endl;
 thrust::host_vector<uint> hostExtendedKeysDEBUG = tmpExtendedBucketKeys;
 thrust::host_vector<uint> hostExtendedValuesDEBUG = tmpExtendedBucketValues;
 for (uint i = 0; i < hostExtendedKeysDEBUG.size(); i++) {
 std::cout << "key:" << hostExtendedKeysDEBUG[i] << ", value:"
 << hostExtendedValuesDEBUG[i] << std::endl;
 }

 thrust::device_vector<uint> fullBucketKeys(totalBucketCount, 0);
 thrust::device_vector<uint> fullBucketValues(totalBucketCount, 0);
 thrust::sequence(fullBucketKeys.begin(), fullBucketKeys.end());
 thrust::scatter(tmpBucketValues.begin(), tmpBucketValues.end(),
 tmpBucketKeys.begin(), fullBucketValues.begin());
 thrust::host_vector<uint> hostKeysDEBUG2 = fullBucketKeys;
 thrust::host_vector<uint> hostValuesDEBUG2 = fullBucketValues;
 for (uint i = 0; i < totalBucketCount; i++) {
 std::cout << "key:" << hostKeysDEBUG2[i] << ", value:"
 << hostValuesDEBUG2[i] << std::endl;
 }

 thrust::device_vector<uint> fullExtendedBucketKeys(totalBucketCount, 0);
 thrust::device_vector<uint> fullExtendedBucketValues(totalBucketCount, 0);
 thrust::sequence(fullExtendedBucketKeys.begin(),
 fullExtendedBucketKeys.end());
 thrust::scatter(tmpExtendedBucketValues.begin(),
 tmpExtendedBucketValues.end(), tmpExtendedBucketKeys.begin(),
 fullExtendedBucketValues.begin());
 thrust::host_vector<uint> hostExtendedKeysDEBUG2 = fullExtendedBucketKeys;
 thrust::host_vector<uint> hostExtendedValuesDEBUG2 =
 fullExtendedBucketValues;
 for (uint i = 0; i < totalBucketCount; i++) {
 std::cout << "key:" << hostExtendedKeysDEBUG2[i] << ", value:"
 << hostExtendedValuesDEBUG2[i] << std::endl;
 }
 int numberOfKeysOfOurInterest = thrust::count_if(
 thrust::make_zip_iterator(
 thrust::make_tuple(fullBucketValues.begin(),
 fullExtendedBucketValues.begin())),
 thrust::make_zip_iterator(
 thrust::make_tuple(fullBucketValues.end(),
 fullExtendedBucketValues.end())), bothNoneZero());
 std::cout << "number of keys of our interest= " << numberOfKeysOfOurInterest
 << std::endl;
 thrust::device_vector<uint> usefulKeys(numberOfKeysOfOurInterest);
 thrust::device_vector<uint> usefulValues(numberOfKeysOfOurInterest);
 thrust::device_vector<uint> usefulExtendedValues(numberOfKeysOfOurInterest);
 thrust::copy_if(
 thrust::make_zip_iterator(
 thrust::make_tuple(fullBucketValues.begin(),
 fullExtendedBucketValues.begin(),
 fullExtendedBucketKeys.begin())),
 thrust::make_zip_iterator(
 thrust::make_tuple(fullBucketValues.end(),
 fullExtendedBucketValues.end(),
 fullExtendedBucketKeys.end())),
 thrust::make_zip_iterator(
 thrust::make_tuple(usefulValues.begin(),
 usefulExtendedValues.begin(), usefulKeys.begin())),
 bothNoneZero2());

 thrust::host_vector<uint> hostUsefulKeys = usefulKeys;
 thrust::host_vector<uint> hostUsefulValues = usefulValues;
 thrust::host_vector<uint> hostUsefulExtendedValues = usefulExtendedValues;
 for (uint i = 0; i < numberOfKeysOfOurInterest; i++) {
 std::cout << "key:" << hostUsefulKeys[i] << ", value1:"
 << hostUsefulValues[i] << ", value2: "
 << hostUsefulExtendedValues[i] << std::endl;
 }

 thrust::device_vector<unsigned int> bucket_begin(totalBucketCount);
 thrust::device_vector<unsigned int> bucket_end(totalBucketCount);

 thrust::counting_iterator<unsigned int> search_begin(0);
 thrust::lower_bound(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
 search_begin, search_begin + totalBucketCount,
 bucket_begin.begin());

 // find the end of each bucket's list of points
 thrust::upper_bound(bucketKeysExpanded.begin(), bucketKeysExpanded.end(),
 search_begin, search_begin + totalBucketCount, bucket_end.begin());
 thrust::host_vector<uint> hostBucketBegin = bucket_begin;
 thrust::host_vector<uint> hostBucketEnd = bucket_end;
 for (uint i = 0; i < totalBucketCount; i++) {
 std::cout << "begin index:" << hostBucketBegin[i] << ", end index:"
 << hostBucketEnd[i] << std::endl;
 }

 thrust::device_vector<unsigned int> selectedBucketIndexBegin(
 numberOfKeysOfOurInterest);
 thrust::device_vector<unsigned int> selectedBucketIndexEnd(
 numberOfKeysOfOurInterest);
 thrust::copy(
 thrust::make_permutation_iterator(bucket_begin.begin(),
 usefulKeys.begin()),
 thrust::make_permutation_iterator(bucket_begin.begin(),
 usefulKeys.end()), selectedBucketIndexBegin.begin());

 thrust::copy(
 thrust::make_permutation_iterator(bucket_end.begin(),
 usefulKeys.begin()),
 thrust::make_permutation_iterator(bucket_end.begin(),
 usefulKeys.end()), selectedBucketIndexEnd.begin());
 thrust::host_vector<uint> selectedBucketIndexBeginFromGPU =
 selectedBucketIndexBegin;
 thrust::host_vector<uint> selectedBucketIndexEndFromGPU =
 selectedBucketIndexEnd;
 for (uint i = 0; i < numberOfKeysOfOurInterest; i++) {
 std::cout << i << " th useful key, begin index = "
 << selectedBucketIndexBeginFromGPU[i] << ", end index = "
 << selectedBucketIndexEndFromGPU[i] << std::endl;
 }

 uint modifiedExtendedValuesSize = bucketKeysExpanded.size();
 thrust::device_vector<int> auxSeq1(numberOfKeysOfOurInterest);
 thrust::device_vector<int> auxSeq2(numberOfKeysOfOurInterest);
 thrust::device_vector<int> auxSeq3(modifiedExtendedValuesSize);
 thrust::device_vector<int> auxSeq4(modifiedExtendedValuesSize);
 //thrust::device_vector<int> auxSeq5(modifiedExtendedValuesSize);

 thrust::sequence(auxSeq1.begin(), auxSeq1.end(), 1, 1);
 thrust::sequence(auxSeq2.begin(), auxSeq2.end(), -1, -1);
 thrust::scatter(auxSeq1.begin(), auxSeq1.end(),
 selectedBucketIndexBegin.begin(), auxSeq3.begin());
 thrust::scatter(auxSeq2.begin(), auxSeq2.end(),
 selectedBucketIndexEnd.begin(), auxSeq4.begin());
 thrust::inclusive_scan(auxSeq3.begin(), auxSeq3.end(), auxSeq3.begin(),
 thrust::maximum<uint>());
 thrust::inclusive_scan(auxSeq4.begin(), auxSeq4.end(), auxSeq4.begin(),
 thrust::minimum<int>());
 thrust::transform(auxSeq3.begin(), auxSeq3.end(), auxSeq4.begin(),
 auxSeq3.begin(), thrust::plus<uint>());
 thrust::host_vector<int> copyAuxIndex = auxSeq3;
 //thrust::host_vector<int> seq4 = auxSeq4;
 std::cout << "array size:" << modifiedExtendedValuesSize << std::endl;
 for (uint i = 0; i < copyAuxIndex.size(); i++) {
 std::cout << copyAuxIndex[i] << ", ";
 }
 std::cout << std::endl;
 //for (uint i = 0; i < copyAuxIndex.size(); i++) {
 //	std::cout << seq4[i] << ", ";
 //}
 //std::cout << std::endl;

 uint shrinkedExtendedValueSize = thrust::count(auxSeq3.begin(),
 auxSeq3.end(), 1);
 std::cout << "shrinked size = " << shrinkedExtendedValueSize << std::endl;
 thrust::device_vector<uint> shrinkedExtendedKeys(shrinkedExtendedValueSize);
 thrust::device_vector<uint> shrinkedExtendedValues(
 shrinkedExtendedValueSize);
 thrust::copy_if(
 thrust::make_zip_iterator(
 thrust::make_tuple(bucketKeysExpanded.begin(),
 bucketValuesIncludingNeighbor.begin())),
 thrust::make_zip_iterator(
 thrust::make_tuple(bucketKeysExpanded.end(),
 bucketValuesIncludingNeighbor.end())),
 auxSeq3.begin(),
 thrust::make_zip_iterator(
 thrust::make_tuple(shrinkedExtendedKeys.begin(),
 shrinkedExtendedValues.begin())), isOne());
 thrust::host_vector<uint> shrinkedExtendedKeysFromGPU = shrinkedExtendedKeys;
 thrust::host_vector<uint> shrinkedExtendedValuesFromGPU =
 shrinkedExtendedValues;
 for (uint i = 0; i < shrinkedExtendedValueSize; i++) {
 std::cout << "key:" << shrinkedExtendedKeysFromGPU[i] << ", value = "
 << shrinkedExtendedValuesFromGPU[i] << std::endl;
 }
 }
 */
void SceNodes::move(double dt) {
	thrust::transform(
			make_zip_iterator(
					make_tuple(nodeVelX.begin(), nodeVelY.begin(),
							nodeVelZ.begin())),
			make_zip_iterator(
					make_tuple(nodeVelX.end(), nodeVelY.end(), nodeVelZ.end())),
			make_zip_iterator(
					make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin())),
			make_zip_iterator(
					make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin())), AddFunctor(dt));
}
