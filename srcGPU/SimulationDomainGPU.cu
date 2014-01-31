#include "SimulationDomainGPU.h"

using namespace std;

/**
 * Constructor.
 */
SimulationDomainGPU::SimulationDomainGPU() {
	cout << "before allocation memory" << endl;
	thrust::host_vector<int> aa;
	aa.resize(50000);
	//thrust::device_vector<int> bb = aa;
	thrust::device_vector<double> cc(5000);
	cout << "after allocate memory" << endl;

	cout << "start to create simulatonDomainGPU object" << endl;

	uint maxCellInDomain = globalConfigVars.getConfigValue(
			string("MaxCellInDomain")).toInt();
	uint maxNodePerCell =
			globalConfigVars.getConfigValue("MaxNodePerCell").toDouble();
	uint maxECMInDomain =
			globalConfigVars.getConfigValue("MaxECMInDomain").toDouble();
	uint maxNodePerECM =
			globalConfigVars.getConfigValue("MaxNodePerECM").toDouble();

	minX = globalConfigVars.getConfigValue("DOMAIN_XMIN").toDouble();
	maxX = globalConfigVars.getConfigValue("DOMAIN_XMAX").toDouble();
	minY = globalConfigVars.getConfigValue("DOMAIN_YMIN").toDouble();
	maxY = globalConfigVars.getConfigValue("DOMAIN_YMAX").toDouble();

	gridSpacing =
			globalConfigVars.getConfigValue("Cell_Center_Interval").toDouble();

	growthGridXDim = globalConfigVars.getConfigValue("GrowthGridXDim").toInt();
	growthGridYDim = globalConfigVars.getConfigValue("GrowthGridYDim").toInt();
	growthGridSpacing =
			globalConfigVars.getConfigValue("GrowthGridSpacing").toDouble();
	growthGridLowerLeftPtX = globalConfigVars.getConfigValue(
			"GrowthGridLowerLeftPtX").toDouble();
	growthGridLowerLeftPtY = globalConfigVars.getConfigValue(
			"GrowthGridLowerLeftPtY").toDouble();

	growthMorCenterXCoord = globalConfigVars.getConfigValue(
			"GrowthMorCenterXCoord").toDouble();
	growthMorCenterYCoord = globalConfigVars.getConfigValue(
			"GrowthMorCenterYCoord").toDouble();
	growthMorHighConcen =
			globalConfigVars.getConfigValue("GrowthMorHighConcen").toDouble();
	growthMorLowConcen =
			globalConfigVars.getConfigValue("GrowthMorLowConcen").toDouble();
	growthMorDiffSlope =
			globalConfigVars.getConfigValue("GrowthMorDiffSlope").toDouble();

	growthMorCenterXCoordMX = globalConfigVars.getConfigValue(
			"GrowthMorCenterXCoordMX").toDouble();
	growthMorCenterYCoordMX = globalConfigVars.getConfigValue(
			"GrowthMorCenterYCoordMX").toDouble();
	growthMorHighConcenMX = globalConfigVars.getConfigValue(
			"GrowthMorHighConcenMX").toDouble();
	growthMorLowConcenMX = globalConfigVars.getConfigValue(
			"GrowthMorLowConcenMX").toDouble();
	growthMorDiffSlopeMX = globalConfigVars.getConfigValue(
			"GrowthMorDiffSlopeMX").toDouble();

	intraLinkDisplayRange = globalConfigVars.getConfigValue(
			"IntraLinkDisplayRange").toDouble();

	cout << "after reading values from config file" << endl;
	cout << "key parameters are : maxCellInDomain = " << maxCellInDomain
			<< "maxNodePerCell = " << maxNodePerCell << "maxECMInDomain = "
			<< maxECMInDomain << "maxNodePerECM = " << maxNodePerECM << endl;
	nodes = SceNodes(maxCellInDomain, maxNodePerCell, maxECMInDomain,
			maxNodePerECM);
	cout << "after created nodes object" << endl;
	cells = SceCells(&nodes);

	cout << "after created nodes and cells object" << endl;
	growthMap = GrowthDistriMap(growthGridXDim, growthGridYDim,
			growthGridSpacing);
	growthMap.initialize(growthGridLowerLeftPtX, growthGridLowerLeftPtY,
			growthMorCenterXCoord, growthMorCenterYCoord, growthMorHighConcen,
			growthMorLowConcen, growthMorDiffSlope);

	cout << "after created growthMap1" << endl;
	growthMap2 = GrowthDistriMap(growthGridXDim, growthGridYDim,
			growthGridSpacing);
	growthMap2.initialize(growthGridLowerLeftPtX, growthGridLowerLeftPtY,
			growthMorCenterXCoordMX, growthMorCenterYCoordMX,
			growthMorHighConcenMX, growthMorLowConcenMX, growthMorDiffSlopeMX);
	cout << "after created growthMap2" << endl;
}

/*
 * we have to initialize three types of cells:
 * first is Boundary (B),
 * second is FNM (F),
 * third is MX (M).
 * like this:
 * B-B-B-B-B-B-B-B-B-F-F-F-F-F-F-F-F-M-M-M-M-M-M-M-M
 * Rule:
 * 1, each input vector must be divided exactly by (max node per cell)
 * 2, sum of number of cells from all input vectors must be size of cellTypes
 *    so that all cells will have its own type
 * First part is error checking.
 * Second part is the actual initialization
 *
 */
void SimulationDomainGPU::initialCellsOfThreeTypes(
		std::vector<CellType> cellTypes,
		std::vector<uint> numOfInitActiveNodesOfCells,
		std::vector<double> initBdryCellNodePosX,
		std::vector<double> initBdryCellNodePosY,
		std::vector<double> initFNMCellNodePosX,
		std::vector<double> initFNMCellNodePosY,
		std::vector<double> initMXCellNodePosX,
		std::vector<double> initMXCellNodePosY) {
	/*
	 * first step: error checking.
	 * we need to first check if inputs are valid
	 */

	cout << "begin init cells of three types" << endl;
	// get max node per cell. should be defined previously.
	uint maxNodePerCell = nodes.maxNodeOfOneCell;

	// obtain sizes of the input arrays
	uint bdryNodeCountX = initBdryCellNodePosX.size();
	uint bdryNodeCountY = initBdryCellNodePosY.size();
	uint FNMNodeCountX = initFNMCellNodePosX.size();
	uint FNMNodeCountY = initFNMCellNodePosY.size();
	uint MXNodeCountX = initMXCellNodePosX.size();
	uint MXNodeCountY = initMXCellNodePosY.size();
	cout << "size of all node vectors:" << bdryNodeCountX << ", "
			<< bdryNodeCountY << ", " << FNMNodeCountX << ", " << FNMNodeCountY
			<< ", " << MXNodeCountX << ", " << MXNodeCountY << endl;
	// array size of cell type array
	uint cellTypeSize = cellTypes.size();
	// array size of initial active node count of cells array.
	uint initNodeCountSize = numOfInitActiveNodesOfCells.size();
	// two sizes must match.
	assert(cellTypeSize == initNodeCountSize);
	// size of X and Y must match.
	assert(bdryNodeCountX == bdryNodeCountY);
	assert(FNMNodeCountX == FNMNodeCountY);
	assert(MXNodeCountX == MXNodeCountY);
	// size of inputs must be divided exactly by max node per cell.
	uint bdryRemainder = bdryNodeCountX % maxNodePerCell;
	uint fnmRemainder = FNMNodeCountX % maxNodePerCell;
	uint mxRemainder = MXNodeCountX % maxNodePerCell;
	uint bdryQuotient = bdryNodeCountX / maxNodePerCell;
	uint fnmQuotient = FNMNodeCountX / maxNodePerCell;
	uint mxQuotient = MXNodeCountX / maxNodePerCell;

	// remainder must be zero.
	assert((bdryRemainder == 0) && (fnmRemainder == 0) && (mxRemainder == 0));
	// size of cellType array and sum of all cell types must match.
	assert(bdryQuotient + fnmQuotient + mxQuotient == cellTypeSize);

	// make sure the cell types follow format requirement.
	int counter = 0;
	while (counter < cellTypeSize) {
		if (counter < bdryQuotient) {
			assert(cellTypes[counter] == Boundary);
		} else {
			int tmp = counter - bdryQuotient;
			if (tmp < fnmQuotient) {
				assert(cellTypes[counter] == FNM);
			} else {
				assert(cellTypes[counter] == MX);
			}
		}
		counter++;
	}

	nodes.setCurrentActiveCellCount(cellTypeSize);
	cells.currentActiveCellCount = cellTypeSize;

	// copy input of initial active node of cells to our actual data location
	thrust::copy(numOfInitActiveNodesOfCells.begin(),
			numOfInitActiveNodesOfCells.end(),
			cells.activeNodeCountOfThisCell.begin());

	// copy x and y position of nodes of boundary cells to actual node position.
	thrust::copy(initBdryCellNodePosX.begin(), initBdryCellNodePosX.end(),
			nodes.nodeLocX.begin());
	thrust::copy(initBdryCellNodePosY.begin(), initBdryCellNodePosY.end(),
			nodes.nodeLocY.begin());

	// find the begining position of FNM cells.
	uint beginAddressOfFNM = bdryNodeCountX;
	// copy x and y position of nodes of FNM cells to actual node position.
	thrust::copy(initFNMCellNodePosX.begin(), initFNMCellNodePosX.end(),
			nodes.nodeLocX.begin() + beginAddressOfFNM);
	thrust::copy(initFNMCellNodePosY.begin(), initFNMCellNodePosY.end(),
			nodes.nodeLocY.begin() + beginAddressOfFNM);

	// find the begining position of MX cells.
	uint beginAddressOfMX = beginAddressOfFNM + FNMNodeCountX;
	thrust::copy(initMXCellNodePosX.begin(), initMXCellNodePosX.end(),
			nodes.nodeLocX.begin() + beginAddressOfMX);
	thrust::copy(initMXCellNodePosY.begin(), initMXCellNodePosY.end(),
			nodes.nodeLocY.begin() + beginAddressOfMX);

	// set cell types
	thrust::device_vector<CellType> cellTypesToPass = cellTypes;

	// copy initial active node count info to GPU
	thrust::copy(numOfInitActiveNodesOfCells.begin(),
			numOfInitActiveNodesOfCells.end(),
			cells.activeNodeCountOfThisCell.begin());
	// set isActiveInfo
	// allocate space for isActive info
	uint sizeOfTmpVector = maxNodePerCell * initNodeCountSize;
	thrust::host_vector<bool> isActive(sizeOfTmpVector, false);

	for (int i = 0; i < initNodeCountSize; i++) {
		int j = 0;
		int index;
		while (j < numOfInitActiveNodesOfCells[i]) {
			index = i * maxNodePerCell + j;
			isActive[index] = true;
			j++;
		}
	}
	thrust::copy(isActive.begin(), isActive.end(), nodes.nodeIsActive.begin());

	// set cell types
	cells.setCellTypes(cellTypesToPass);
}

void SimulationDomainGPU::initializeCells(std::vector<double> initCellNodePosX,
		std::vector<double> initCellNodePosY, std::vector<double> centerPosX,
		std::vector<double> centerPosY, uint cellSpaceForBdry) {
	uint numberOfInitActiveCells = centerPosX.size();
	uint numberOfInitActiveNodePerCell = initCellNodePosX.size();
	uint maxNodePerCell = nodes.maxNodeOfOneCell;
	uint sizeOfTmpVector = numberOfInitActiveCells * maxNodePerCell;

	thrust::host_vector<double> xPos(sizeOfTmpVector, 0.0);
	thrust::host_vector<double> yPos(sizeOfTmpVector, 0.0);
	thrust::host_vector<bool> isActive(sizeOfTmpVector, false);

	/**
	 * following lines make sure that initCellNode has (0.0,0.0) as center position.
	 */
	double centerXPosOfInitCell = 0.0;
	double centerYPosOfInitCell = 0.0;
	for (uint i = 0; i < numberOfInitActiveNodePerCell; i++) {
		centerXPosOfInitCell += initCellNodePosX[i];
		centerYPosOfInitCell += initCellNodePosY[i];
	}
	centerXPosOfInitCell = centerXPosOfInitCell / numberOfInitActiveNodePerCell;
	centerYPosOfInitCell = centerYPosOfInitCell / numberOfInitActiveNodePerCell;
	for (uint i = 0; i < numberOfInitActiveNodePerCell; i++) {
		initCellNodePosX[i] = initCellNodePosX[i] - centerXPosOfInitCell;
		initCellNodePosY[i] = initCellNodePosY[i] - centerYPosOfInitCell;
	}

	for (uint i = 0; i < sizeOfTmpVector; i++) {
		uint cellRank = i / maxNodePerCell;
		uint nodeRank = i % maxNodePerCell;
		if (nodeRank < numberOfInitActiveNodePerCell) {
			xPos[i] = initCellNodePosX[nodeRank] + centerPosX[cellRank];
			yPos[i] = initCellNodePosY[nodeRank] + centerPosY[cellRank];
			isActive[i] = true;
		}
	}

	// active cell count includes cell spaces that are reserved for boundary.
	nodes.setCurrentActiveCellCount(numberOfInitActiveCells);
	cells.currentActiveCellCount = numberOfInitActiveCells;
	// cell space for boundary means number of cell spaces that are reserved for bdry.
	nodes.setCellSpaceForBdry(cellSpaceForBdry);
	cells.setCellSpaceForBdry(cellSpaceForBdry);
	thrust::host_vector<uint> cellActiveNodeCounts(numberOfInitActiveCells,
			numberOfInitActiveNodePerCell);
	thrust::copy(cellActiveNodeCounts.begin(), cellActiveNodeCounts.end(),
			cells.activeNodeCountOfThisCell.begin());
	thrust::copy(xPos.begin(), xPos.end(), nodes.nodeLocX.begin());
	thrust::copy(yPos.begin(), yPos.end(), nodes.nodeLocY.begin());
	thrust::copy(isActive.begin(), isActive.end(), nodes.nodeIsActive.begin());

}

/**
 * cell types need to initialized.
 * Seperated from initializeCells function.
 */
void SimulationDomainGPU::initializeCellTypes(std::vector<CellType> cellTypes) {
	uint cellTypeInputSize = cellTypes.size();
	assert(cells.currentActiveCellCount == cellTypeInputSize);
	thrust::device_vector<CellType> cellTypesToPass = cellTypes;
	cells.setCellTypes(cellTypesToPass);
}

void SimulationDomainGPU::runAllLogic(double dt) {
	nodes.calculateAndApplySceForces(minX, maxX, minY, maxY, gridSpacing);
//nodes.move(dt);
	//cells.growAndDivide(dt, growthMap.growthFactorMag,
	//		growthMap.growthFactorDirXComp, growthMap.growthFactorDirYComp,
	//		growthMap.gridDimensionX, growthMap.gridDimensionY,
	//		growthMap.gridSpacing);
	cells.growAndDivide(dt, growthMap, growthMap2);
}

void SimulationDomainGPU::outputVtkFilesWithColor(std::string scriptNameBase,
		int rank) {

	uint activeTotalNodeCount = nodes.currentActiveCellCount
			* nodes.maxNodeOfOneCell;

	thrust::host_vector<uint> hostActiveCountOfCells(
			nodes.currentActiveCellCount);

	thrust::host_vector<double> hostTmpVectorLocX(activeTotalNodeCount);
	thrust::host_vector<double> hostTmpVectorLocY(activeTotalNodeCount);
	thrust::host_vector<double> hostTmpVectorLocZ(activeTotalNodeCount);
	thrust::host_vector<bool> hostTmpVectorIsActive(activeTotalNodeCount);

	std::vector<uint> prefixSum(nodes.currentActiveCellCount);
	std::vector<uint> prefixSumLinks(nodes.currentActiveCellCount);

	thrust::copy(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes.nodeLocX.begin(),
							nodes.nodeLocY.begin(), nodes.nodeLocZ.begin(),
							nodes.nodeIsActive.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes.nodeLocX.begin(),
							nodes.nodeLocY.begin(), nodes.nodeLocZ.begin(),
							nodes.nodeIsActive.begin())) + activeTotalNodeCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(hostTmpVectorLocX.begin(),
							hostTmpVectorLocY.begin(),
							hostTmpVectorLocZ.begin(),
							hostTmpVectorIsActive.begin())));

	thrust::copy(cells.activeNodeCountOfThisCell.begin(),
			cells.activeNodeCountOfThisCell.begin()
					+ cells.currentActiveCellCount,
			hostActiveCountOfCells.begin());

	int i, j, k;
	int totalNNum = thrust::reduce(hostActiveCountOfCells.begin(),
			hostActiveCountOfCells.end());
	std::vector < std::pair<uint, uint> > links;
	uint tmpRes = 0;
	for (i = 0; i < nodes.currentActiveCellCount; i++) {
		prefixSum[i] = tmpRes;
		tmpRes = tmpRes + hostActiveCountOfCells[i];
	}
// using string stream is probably not the best solution,
// but I can't use c++ 11 features for backward compatibility
	std::stringstream ss;
	ss << std::setw(5) << std::setfill('0') << rank;
	std::string scriptNameRank = ss.str();
	std::string vtkFileName = scriptNameBase + scriptNameRank + ".vtk";
	std::cout << "start to create vtk file" << vtkFileName << std::endl;
	std::ofstream fs;
	fs.open(vtkFileName.c_str());

//int totalNNum = getTotalNodeCount();
//int LNum = 0;
//int NNum;
	fs << "# vtk DataFile Version 3.0" << std::endl;
	fs << "Lines and points representing subcelluar element cells "
			<< std::endl;
	fs << "ASCII" << std::endl;
	fs << std::endl;
	fs << "DATASET UNSTRUCTURED_GRID" << std::endl;
	fs << "POINTS " << totalNNum << " float" << std::endl;

	uint counterForLink = 0;
	for (i = 0; i < nodes.currentActiveCellCount; i++) {
		uint activeNodeCount = hostActiveCountOfCells[i];
		for (j = 0; j < activeNodeCount; j++) {
			uint pos = i * nodes.maxNodeOfOneCell + j;
			fs << hostTmpVectorLocX[pos] << " " << hostTmpVectorLocY[pos] << " "
					<< hostTmpVectorLocZ[pos] << std::endl;
		}
		uint tmpCount = 0;
		for (j = 0; j < activeNodeCount; j++) {
			for (k = j; k < activeNodeCount; k++) {
				if (j == k) {
					continue;
				} else {
					uint pos1 = prefixSum[i] + j;
					uint pos2 = prefixSum[i] + k;
					uint pos1InVec = i * nodes.maxNodeOfOneCell + j;
					uint pos2InVec = i * nodes.maxNodeOfOneCell + k;
					if (compuDist(hostTmpVectorLocX[pos1InVec],
							hostTmpVectorLocY[pos1InVec],
							hostTmpVectorLocZ[pos1InVec],
							hostTmpVectorLocX[pos2InVec],
							hostTmpVectorLocY[pos2InVec],
							hostTmpVectorLocZ[pos2InVec])
							<= intraLinkDisplayRange) {
						links.push_back(std::make_pair<uint, uint>(pos1, pos2));
						tmpCount++;
					} else {
						continue;
					}
				}
			}
		}
		prefixSumLinks[i] = counterForLink;
		counterForLink = counterForLink + tmpCount;
	}
	fs << std::endl;
	fs << "CELLS " << counterForLink << " " << 3 * counterForLink << std::endl;
	uint linkSize = links.size();
	for (uint i = 0; i < linkSize; i++) {
		fs << 2 << " " << links[i].first << " " << links[i].second << std::endl;
	}
	uint LNum = links.size();
	fs << "CELL_TYPES " << LNum << endl;
	for (i = 0; i < LNum; i++) {
		fs << "3" << endl;
	}
	fs << "POINT_DATA " << totalNNum << endl;
	fs << "SCALARS point_scalars float" << endl;
	fs << "LOOKUP_TABLE default" << endl;
	for (i = 0; i < nodes.currentActiveCellCount; i++) {
		uint activeNodeCount = hostActiveCountOfCells[i];
		for (j = 0; j < activeNodeCount; j++) {
			fs << i << endl;
		}
	}

	fs.flush();
	fs.close();
}
