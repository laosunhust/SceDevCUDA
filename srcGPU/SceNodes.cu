#include "SceNodes.h"

__constant__ double sceInterPara[5];
__constant__ double sceIntraPara[4];
double sceInterParaCPU[5];
double sceIntraParaCPU[4];

__constant__ double sceDiffPara[5];
double sceDiffParaCPU[5];

// __constant__ uint sceNodeAuxInfo[5];

__constant__ uint ProfilebeginPos;
__constant__ uint ECMbeginPos;
__constant__ uint cellNodeBeginPos;
__constant__ uint nodeCountPerECM;
__constant__ uint nodeCountPerCell;

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
	nodeIsActive.resize(maxTotalNodeCount, 0);
	nodeLocX.resize(maxTotalNodeCount, 0.0);
	nodeLocY.resize(maxTotalNodeCount, 0.0);
	nodeLocZ.resize(maxTotalNodeCount, 0.0);
	nodeVelX.resize(maxTotalNodeCount, 0.0);
	nodeVelY.resize(maxTotalNodeCount, 0.0);
	nodeVelZ.resize(maxTotalNodeCount, 0.0);
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
	std::cout << "start creating SceNodes object" << std::endl;
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
	std::cout << "before resizing vectors" << std::endl;
	uint maxTotalNodeCount = maxTotalCellNodeCount + maxTotalECMNodeCount;
	std::cout << "maxTotalNodeCount = " << maxTotalNodeCount << std::endl;
	//thrust::host_vector<bool> nodeIsActiveHost
	nodeIsActive.resize(maxTotalNodeCount);
	std::cout << "nodeIsActive resize complete" << std::endl;
	nodeLocX.resize(maxTotalNodeCount);
	nodeLocY.resize(maxTotalNodeCount);
	nodeLocZ.resize(maxTotalNodeCount);
	nodeVelX.resize(maxTotalNodeCount);
	nodeVelY.resize(maxTotalNodeCount);
	nodeVelZ.resize(maxTotalNodeCount);
	std::cout << "after resizing vectors" << std::endl;
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

	//ConfigParser parser;
	//std::string configFileName = "sceCell.config";
	// globalConfigVars should be a global variable and defined in the main function.
	//globalConfigVars = parser.parseConfigFile(configFileName);
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

	std::cout << "in SceNodes, before cuda memory copy to symbol:" << std::endl;
	cudaMemcpyToSymbol(sceInterPara, sceInterParaCPU, 5 * sizeof(double));
	cudaMemcpyToSymbol(sceIntraPara, sceIntraParaCPU, 4 * sizeof(double));
	std::cout << "finished SceNodes:" << std::endl;
}

SceNodes::SceNodes(uint totalBdryNodeCount, uint maxProfileNodeCount,
		uint maxTotalECMCount, uint maxNodeInECM, uint maxTotalCellCount,
		uint maxNodeInCell) {
	std::cout << "start creating SceNodes object" << std::endl;
	maxCellCount = maxTotalCellCount;
	maxNodeOfOneCell = maxNodeInCell;
	maxNodePerECM = maxNodeInECM;
	maxECMCount = maxTotalECMCount;
	currentActiveCellCount = 0;
	maxTotalECMNodeCount = maxECMCount * maxNodePerECM;
	currentActiveECM = 0;

	// will need to change this value after we have more detail about ECM
	maxTotalCellNodeCount = maxTotalCellCount * maxNodeOfOneCell;

	//cellRanks.resize(maxTotalNodeCount);
	//nodeRanks.resize(maxTotalNodeCount);
	//std::cout << "before resizing vectors" << std::endl;
	uint maxTotalNodeCount = totalBdryNodeCount + maxProfileNodeCount
			+ maxTotalECMNodeCount + maxTotalCellNodeCount;
	//std::cout << "maxTotalNodeCount = " << maxTotalNodeCount << std::endl;
	//thrust::host_vector<bool> nodeIsActiveHost

	nodeLocX.resize(maxTotalNodeCount);
	nodeLocY.resize(maxTotalNodeCount);
	nodeLocZ.resize(maxTotalNodeCount);
	nodeVelX.resize(maxTotalNodeCount);
	nodeVelY.resize(maxTotalNodeCount);
	nodeVelZ.resize(maxTotalNodeCount);
	nodeCellType.resize(maxTotalNodeCount);
	nodeCellRank.resize(maxTotalNodeCount);
	nodeIsActive.resize(maxTotalNodeCount);

	startPosProfile = totalBdryNodeCount;
	startPosECM = startPosProfile + maxProfileNodeCount;
	startPosCells = startPosECM + maxTotalECMNodeCount;

	thrust::host_vector<CellType> hostTmpVector(maxTotalNodeCount);
	thrust::host_vector<bool> hostTmpVector2(maxTotalNodeCount);
	for (int i = 0; i < maxTotalNodeCount; i++) {
		if (i < startPosProfile) {
			hostTmpVector[i] = Boundary;
		} else if (i < startPosECM) {
			hostTmpVector[i] = Profile;
		} else if (i < startPosCells) {
			hostTmpVector[i] = ECM;
		} else {
			// all initialized as FNM
			hostTmpVector[i] = FNM;
		}
		nodeIsActive[i] = false;
	}
	nodeCellType = hostTmpVector;
	nodeIsActive = hostTmpVector2;
	copyParaToGPUConstMem();
}

void SceNodes::copyParaToGPUConstMem() {
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

	//std::cout << "in SceNodes, before cuda memory copy to symbol:" << std::endl;
	cudaMemcpyToSymbol(sceInterPara, sceInterParaCPU, 5 * sizeof(double));
	cudaMemcpyToSymbol(sceIntraPara, sceIntraParaCPU, 4 * sizeof(double));
	cudaMemcpyToSymbol(ProfilebeginPos, &startPosProfile, sizeof(uint));
	cudaMemcpyToSymbol(ECMbeginPos, &startPosECM, sizeof(uint));
	cudaMemcpyToSymbol(cellNodeBeginPos, &startPosCells, sizeof(uint));
	cudaMemcpyToSymbol(nodeCountPerECM, &maxNodePerECM, sizeof(uint));
	cudaMemcpyToSymbol(nodeCountPerCell, &maxNodeOfOneCell, sizeof(uint));
	//std::cout << "finished SceNodes:" << std::endl;
}

void SceNodes::addNewlyDividedCells(
		thrust::device_vector<double> &nodeLocXNewCell,
		thrust::device_vector<double> &nodeLocYNewCell,
		thrust::device_vector<double> &nodeLocZNewCell,
		thrust::device_vector<bool> &nodeIsActiveNewCell) {

	uint shiftSize = nodeLocXNewCell.size();
	assert(shiftSize % maxNodeOfOneCell == 0);
	uint addCellCount = shiftSize / maxNodeOfOneCell;

	uint shiftStartPos = startPosCells
			+ currentActiveCellCount * maxNodeOfOneCell;
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

void SceNodes::addNewlyDividedCells(
		thrust::device_vector<double> &nodeLocXNewCell,
		thrust::device_vector<double> &nodeLocYNewCell,
		thrust::device_vector<double> &nodeLocZNewCell,
		thrust::device_vector<bool> &nodeIsActiveNewCell,
		thrust::device_vector<CellType> &nodeCellTypeNewCell) {

	// data validation
	uint nodesSize = nodeLocXNewCell.size();
	assert(nodesSize % maxNodeOfOneCell == 0);
	uint addCellCount = nodesSize / maxNodeOfOneCell;

	// position that we will add newly divided cells.
	uint shiftStartPosNewCell = startPosCells
			+ currentActiveCellCount * maxNodeOfOneCell;

	thrust::copy(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocXNewCell.begin(),
							nodeLocYNewCell.begin(), nodeLocZNewCell.begin(),
							nodeIsActiveNewCell.begin(),
							nodeCellTypeNewCell.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocXNewCell.end(),
							nodeLocYNewCell.end(), nodeLocZNewCell.end(),
							nodeIsActiveNewCell.end(),
							nodeCellTypeNewCell.end())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodeLocX.begin(), nodeLocY.begin(),
							nodeLocZ.begin(), nodeIsActive.begin(),
							nodeCellType.begin())) + shiftStartPosNewCell);

	// total number of cells has increased.
	currentActiveCellCount = currentActiveCellCount + addCellCount;
}

void SceNodes::buildBuckets2D(double minX, double maxX, double minY,
		double maxY, double bucketSize) {
	int totalActiveNodes = startPosCells
			+ currentActiveCellCount * maxNodeOfOneCell;

	// TODO: change number of total active nodes
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
void calculateAndAddProfileForce(double &xPos, double &yPos, double &zPos,
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
		if (forceValue > 0) {
			//forceValue = 0;
			forceValue = forceValue * 0.3;
		}
	}
	if (linkLength > 1.0e-12) {
		xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
		yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
		zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
	}

}

__device__
void calculateAndAddInterForceDiffType(double &xPos, double &yPos, double &zPos,
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
		if (forceValue > 0) {
			//forceValue = 0;
			forceValue = forceValue * 0.3;
		}
	}
	if (linkLength > 1.0e-12) {
		xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
		yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
		zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
	}
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
	if (linkLength > 1.0e-12) {
		xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
		yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
		zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
	}
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

__device__ bool isSameCell(uint nodeGlobalRank1, uint nodeGlobalRank2) {
	if ((nodeGlobalRank1 - cellNodeBeginPos) / nodeCountPerCell
			== (nodeGlobalRank2 - cellNodeBeginPos) / nodeCountPerCell) {
		return true;
	} else {
		return false;
	}
}

__device__ bool isSameECM(uint nodeGlobalRank1, uint nodeGlobalRank2) {
	if ((nodeGlobalRank1 - ECMbeginPos) / nodeCountPerECM
			== (nodeGlobalRank2 - ECMbeginPos) / nodeCountPerECM) {
		return true;
	} else {
		return false;
	}
}

__device__ bool isNeighborECMNodes(uint nodeGlobalRank1, uint nodeGlobalRank2) {
	// this means that two nodes are from the same ECM
	if ((nodeGlobalRank1 - ECMbeginPos) / nodeCountPerECM
			== (nodeGlobalRank2 - ECMbeginPos) / nodeCountPerECM) {
		// this means that two nodes are actually close to each other
		// seems to be strange because of unsigned int.
		if ((nodeGlobalRank1 > nodeGlobalRank2
				&& nodeGlobalRank1 - nodeGlobalRank2 == 1)
				|| (nodeGlobalRank2 > nodeGlobalRank1
						&& nodeGlobalRank2 - nodeGlobalRank1 == 1)) {
			return true;
		}
	}
	return false;
}

__device__ bool isNeighborProfileNodes(uint nodeGlobalRank1,
		uint nodeGlobalRank2) {
	if ((nodeGlobalRank1 > nodeGlobalRank2
			&& nodeGlobalRank1 - nodeGlobalRank2 == 1)
			|| (nodeGlobalRank2 > nodeGlobalRank1
					&& nodeGlobalRank2 - nodeGlobalRank1 == 1)) {
		return true;
	}
	return false;
}

__device__ bool ofSameType(uint cellType1, uint cellType2) {
	if (cellType1 == cellType2) {
		return true;
	} else {
		return false;
	}
}

__device__
void handleForceBetweenNodes(uint &nodeRank1, CellType &type1, uint &nodeRank2,
		CellType &type2, double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes, double* _nodeLocXAddress, double* _nodeLocYAddress,
		double* _nodeLocZAddress) {
	// this means that both nodes come from cells
	if ((type1 == MX || type1 == FNM) && (type2 == MX || type2 == FNM)) {
		// this means that nodes come from different type of cell, apply differential adhesion
		if (type1 != type2) {
			// TODO: apply differential adhesion here.
			// It should be a different type of inter force.
			calculateAndAddInterForce(xPos, yPos, zPos,
					_nodeLocXAddress[nodeRank2], _nodeLocYAddress[nodeRank2],
					_nodeLocZAddress[nodeRank2], xRes, yRes, zRes);
		} else {
			// TODO: this function needs to be modified.
			// (1) nodeCountPerCell need to be stored in constant memory.
			// (2) begin address of cell nodes need to be stored in constant memory.
			if (isSameCell(nodeRank1, nodeRank2)) {
				calculateAndAddIntraForce(xPos, yPos, zPos,
						_nodeLocXAddress[nodeRank2],
						_nodeLocYAddress[nodeRank2],
						_nodeLocZAddress[nodeRank2], xRes, yRes, zRes);
			} else {
				calculateAndAddInterForce(xPos, yPos, zPos,
						_nodeLocXAddress[nodeRank2],
						_nodeLocYAddress[nodeRank2],
						_nodeLocZAddress[nodeRank2], xRes, yRes, zRes);
			}
		}
	}
	// this means that both nodes come from ECM and from same ECM
	else if (type1 == ECM && type2 == ECM && isSameECM(nodeRank1, nodeRank2)) {
		if (isNeighborECMNodes(nodeRank1, nodeRank2)) {
			// TODO: need to create another two vectors that holds the neighbor information for ECM.
			// TODO: alternatively, try to store ECM begin address and number of node per ECM in constant memory.
			// TODO: implement this function.
			calculateAndAddECMForce(xPos, yPos, zPos,
					_nodeLocXAddress[nodeRank2], _nodeLocYAddress[nodeRank2],
					_nodeLocZAddress[nodeRank2], xRes, yRes, zRes);
		}
		// if both nodes belong to same ECM but are not neighbors they shouldn't interact.
	}
	// this means that both nodes come from profile ( Epithilum layer).
	else if (type1 == Profile && type2 == Profile) {
		if (isNeighborProfileNodes(nodeRank1, nodeRank2)) {
			// TODO: need a set of parameters for calculating linking force between profile nodes
			// TODO: implement this function.
			calculateAndAddProfileForce(xPos, yPos, zPos,
					_nodeLocXAddress[nodeRank2], _nodeLocYAddress[nodeRank2],
					_nodeLocZAddress[nodeRank2], xRes, yRes, zRes);
		}
		// if both nodes belong to Profile but are not neighbors they shouldn't interact.

	} else {
		// for now, we assume that interaction between other nodes are the same as inter-cell force.
		calculateAndAddInterForce(xPos, yPos, zPos, _nodeLocXAddress[nodeRank2],
				_nodeLocYAddress[nodeRank2], _nodeLocZAddress[nodeRank2], xRes,
				yRes, zRes);
	}
}

void SceNodes::extendBuckets2D(uint numOfBucketsInXDim,
		uint numOfBucketsInYDim) {
	static const uint extensionFactor2D = 9;
	uint valuesCount = bucketValues.size();
	bucketKeysExpanded.resize(valuesCount * extensionFactor2D);
	bucketValuesIncludingNeighbor.resize(valuesCount * extensionFactor2D);

	/**
	 * beginning of constant iterator
	 */
	thrust::constant_iterator<uint> first(extensionFactor2D);
	/**
	 * end of constant iterator.
	 * the plus sign only indicate movement of position, not value.
	 * e.g. movement is 5 and first iterator is initialized as 9
	 * result array is [9,9,9,9,9];
	 */
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

	double* nodeLocXAddress = thrust::raw_pointer_cast(&nodeLocX[0]);
	double* nodeLocYAddress = thrust::raw_pointer_cast(&nodeLocY[0]);
	double* nodeLocZAddress = thrust::raw_pointer_cast(&nodeLocZ[0]);
	uint* nodeRankAddress = thrust::raw_pointer_cast(&nodeCellRank[0]);
	CellType* nodeTypeAddress = thrust::raw_pointer_cast(&nodeCellType[0]);
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
					nodeLocZAddress, nodeRankAddress, nodeTypeAddress,
					maxTotalCellNodeCount, maxNodeOfOneCell, maxNodePerECM));
}

void SceNodes::calculateAndApplySceForces(double minX, double maxX, double minY,
		double maxY, double bucketSize) {
	const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
	const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
	std::cout << "in SceNodes, before build buckets 2D:" << std::endl;
	buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	std::cout << "in SceNodes, before extend buckets 2D:" << std::endl;
	extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);
	std::cout << "in SceNodes, before apply sce forces:" << std::endl;
	applySceForces(numberOfBucketsInXDim, numberOfBucketsInYDim);
}

