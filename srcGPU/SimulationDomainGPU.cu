#include "SimulationDomainGPU.h"

using namespace std;

/**
 * Constructor.
 */
SimulationDomainGPU::SimulationDomainGPU() {
	uint maxCellInDomain =
			globalConfigVars.getConfigValue("MaxCellInDomain").toDouble();
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

	intraLinkDisplayRange = globalConfigVars.getConfigValue(
			"IntraLinkDisplayRange").toDouble();

	nodes = SceNodes(maxCellInDomain, maxNodePerCell, maxECMInDomain,
			maxNodePerECM);
	cells = SceCells(&nodes);
	growthMap = GrowthDistriMap(growthGridXDim, growthGridYDim,
			growthGridSpacing);
	growthMap.initialize(growthGridLowerLeftPtX, growthGridLowerLeftPtY,
			growthMorCenterXCoord, growthMorCenterYCoord, growthMorHighConcen,
			growthMorLowConcen, growthMorDiffSlope);
}

void SimulationDomainGPU::initializeCells(std::vector<double> initCellNodePosX,
		std::vector<double> initCellNodePosY, std::vector<double> centerPosX,
		std::vector<double> centerPosY) {
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

	nodes.setCurrentActiveCellCount(numberOfInitActiveCells);
	cells.currentActiveCellCount = numberOfInitActiveCells;
	thrust::host_vector<uint> cellActiveNodeCounts(numberOfInitActiveCells,
			numberOfInitActiveNodePerCell);
	thrust::copy(cellActiveNodeCounts.begin(), cellActiveNodeCounts.end(),
			cells.activeNodeCountOfThisCell.begin());
	thrust::copy(xPos.begin(), xPos.end(), nodes.nodeLocX.begin());
	thrust::copy(yPos.begin(), yPos.end(), nodes.nodeLocY.begin());
	thrust::copy(isActive.begin(), isActive.end(), nodes.nodeIsActive.begin());

}

void SimulationDomainGPU::runAllLogic(double dt) {
	nodes.calculateAndApplySceForces(minX, maxX, minY, maxY, gridSpacing);
	//nodes.move(dt);
	cells.growAndDivide(dt, growthMap.growthFactorMag,
			growthMap.growthFactorDirXComp, growthMap.growthFactorDirYComp,
			growthMap.gridDimensionX, growthMap.gridDimensionY,
			growthMap.gridSpacing);
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
