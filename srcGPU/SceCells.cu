#include "SceCells.h"

__constant__ uint GridDimension[2];
__constant__ double gridSpacing;

double epsilon = 1.0e-12;

SceCells::SceCells(SceNodes* nodesInput) {
	addNodeDistance =
			globalConfigVars.getConfigValue("DistanceForAddingNode").toDouble();
	minDistanceToOtherNode = globalConfigVars.getConfigValue(
			"MinDistanceToOtherNode").toDouble();
	cellInitLength =
			globalConfigVars.getConfigValue("CellInitLength").toDouble();
	cellFinalLength =
			globalConfigVars.getConfigValue("CellFinalLength").toDouble();
	elongationCoefficient = globalConfigVars.getConfigValue(
			"ElongateCoefficient").toDouble();

	maxNodeOfOneCell = nodesInput->getMaxNodeOfOneCell();
	maxCellCount = nodesInput->getMaxCellCount();
	maxTotalCellNodeCount = nodesInput->getMaxTotalCellNodeCount();
	currentActiveCellCount = nodesInput->getCurrentActiveCellCount();
	nodes = nodesInput;
	growthProgress.resize(maxCellCount, 0.0);
	expectedLength.resize(maxCellCount, cellInitLength);
	lengthDifference.resize(maxCellCount, 0.0);
	smallestDistance.resize(maxCellCount);
	biggestDistance.resize(maxCellCount);
	activeNodeCountOfThisCell.resize(maxCellCount);
	lastCheckPoint.resize(maxCellCount, 0.0);
	isDivided.resize(maxCellCount);
	isScheduledToGrow.resize(maxCellCount, false);
	centerCoordX.resize(maxCellCount);
	centerCoordY.resize(maxCellCount);
	centerCoordZ.resize(maxCellCount);
	cellRanksTmpStorage.resize(maxCellCount);
	growthSpeed.resize(maxCellCount, 0.0);
	growthXDir.resize(maxCellCount);
	growthYDir.resize(maxCellCount);

	xCoordTmp.resize(maxTotalCellNodeCount);
	yCoordTmp.resize(maxTotalCellNodeCount);
	zCoordTmp.resize(maxTotalCellNodeCount);
	cellRanks.resize(maxTotalCellNodeCount);
	activeXPoss.resize(maxTotalCellNodeCount);
	activeYPoss.resize(maxTotalCellNodeCount);
	activeZPoss.resize(maxTotalCellNodeCount);
	distToCenterAlongGrowDir.resize(maxTotalCellNodeCount);

	growThreshold = 1.0 / (maxNodeOfOneCell - maxNodeOfOneCell / 2);
}

void SceCells::distributeIsActiveInfo() {
	uint totalNodeCountForActiveCells = currentActiveCellCount
			* maxNodeOfOneCell;
	thrust::counting_iterator<uint> countingBegin(0);
	thrust::counting_iterator<uint> countingEnd(totalNodeCountForActiveCells);
	thrust::transform(
			thrust::make_transform_iterator(countingBegin,
					ModuloFunctor(maxNodeOfOneCell)),
			thrust::make_transform_iterator(countingEnd,
					ModuloFunctor(maxNodeOfOneCell)),
			thrust::make_permutation_iterator(activeNodeCountOfThisCell.begin(),
					make_transform_iterator(countingBegin,
							DivideFunctor(maxNodeOfOneCell))),
			nodes->nodeIsActive.begin(), thrust::less<uint>());
}

void SceCells::computeCenterPos() {
	uint totalNodeCountForActiveCells = currentActiveCellCount
			* maxNodeOfOneCell;
	thrust::counting_iterator<uint> countingBegin(0);
	thrust::counting_iterator<uint> countingEnd(totalNodeCountForActiveCells);
	uint totalNumberOfActiveNodes = thrust::reduce(
			activeNodeCountOfThisCell.begin(),
			activeNodeCountOfThisCell.begin() + currentActiveCellCount);

	thrust::copy_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeLocZ.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeLocZ.begin()))
					+ totalNodeCountForActiveCells, nodes->nodeIsActive.begin(),
			thrust::make_zip_iterator(
					thrust::make_tuple(cellRanks.begin(), activeXPoss.begin(),
							activeYPoss.begin(), activeZPoss.begin())),
			isTrue());

	thrust::reduce_by_key(cellRanks.begin(),
			cellRanks.begin() + totalNumberOfActiveNodes,
			thrust::make_zip_iterator(
					thrust::make_tuple(activeXPoss.begin(), activeYPoss.begin(),
							activeZPoss.begin())), cellRanksTmpStorage.begin(),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), centerCoordZ.begin())),
			thrust::equal_to<uint>(), CVec3Add());
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), centerCoordZ.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), centerCoordZ.begin()))
					+ currentActiveCellCount, activeNodeCountOfThisCell.begin(),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), centerCoordZ.begin())),
			CVec3Divide());
}

void SceCells::grow2DSimplified(double dt,
		thrust::device_vector<double> &growthFactorMag,
		thrust::device_vector<double> &growthFactorDirXComp,
		thrust::device_vector<double> &growthFactorDirYComp,
		uint GridDimensionX, uint GridDimensionY, double GridSpacing) {
	//first step: assign the growth magnitude and direction info that was calculated outside
	//to internal values
	double* growthFactorMagAddress = thrust::raw_pointer_cast(
			&growthFactorMag[0]);
	double* growthFactorDirXAddress = thrust::raw_pointer_cast(
			&growthFactorDirXComp[0]);
	double* growthFactorDirYAddress = thrust::raw_pointer_cast(
			&growthFactorDirYComp[0]);
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin())) + currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(growthSpeed.begin(), growthXDir.begin(),
							growthYDir.begin())),
			LoadGridDataToNode(GridDimensionX, GridDimensionY, GridSpacing,
					growthFactorMagAddress, growthFactorDirXAddress,
					growthFactorDirYAddress));
	//second step: use the growth magnitude and dt to update growthProgress
	thrust::transform(growthSpeed.begin(),
			growthSpeed.begin() + currentActiveCellCount,
			growthProgress.begin(), growthProgress.begin(), SaxpyFunctor(dt));
	//third step: use lastCheckPoint and growthProgress to decide whether add point or not
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())) + currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							lastCheckPoint.begin())), PtCondiOp(growThreshold));
	// fourth step: also add a point if scheduled to grow.
	// This step does not guarantee success ; If adding new point failed, it will not change
	// isScheduleToGrow and activeNodeCount;
	bool* nodeIsActiveAddress = thrust::raw_pointer_cast(
			&(nodes->nodeIsActive[0]));
	double* nodeXPosAddress = thrust::raw_pointer_cast(&(nodes->nodeLocX[0]));
	double* nodeYPosAddress = thrust::raw_pointer_cast(&(nodes->nodeLocY[0]));
	thrust::counting_iterator<uint> countingBegin(0);
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin)),
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin)) + currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin())),
			AddPtOp(maxNodeOfOneCell, addNodeDistance, minDistanceToOtherNode,
					nodeIsActiveAddress, nodeXPosAddress, nodeYPosAddress));
	// fifth step: use growthProgress and growthXDir&growthYDir to compute
	// expected length along the growth direction.
	thrust::transform(growthProgress.begin(),
			growthProgress.begin() + currentActiveCellCount,
			expectedLength.begin(),
			CompuTarLen(cellInitLength, cellFinalLength));
	// sixth step:  reducing the smallest value and biggest value
	// a cell's node to its center point
	uint totalNodeCountForActiveCells = currentActiveCellCount
			* maxNodeOfOneCell;

	// compute direction of each node to its corresponding cell center
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(centerCoordX.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(centerCoordY.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeIsActive.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(centerCoordX.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(centerCoordY.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeIsActive.begin()))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), CompuDist());
	// because distance will be zero if the node is inactive, it will not be max nor min
	thrust::reduce_by_key(
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), cellRanksTmpStorage.begin(),
			smallestDistance.begin(), thrust::equal_to<uint>(),
			thrust::minimum<double>());
	thrust::reduce_by_key(
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), cellRanksTmpStorage.begin(),
			biggestDistance.begin(), thrust::equal_to<uint>(),
			thrust::maximum<double>());
	// seventh step: compute the current length and then
	// compute its difference with expected length
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(expectedLength.begin(),
							smallestDistance.begin(), biggestDistance.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(expectedLength.begin(),
							smallestDistance.begin(), biggestDistance.begin()))
					+ currentActiveCellCount, lengthDifference.begin(),
			CompuDiff());
// eighth step: use the difference that just computed and growthXDir&growthYDir
// to apply stretching force (velocity) on nodes of all cells
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(distToCenterAlongGrowDir.begin(),
							make_permutation_iterator(lengthDifference.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))))),
			thrust::make_zip_iterator(
					thrust::make_tuple(distToCenterAlongGrowDir.begin(),
							make_permutation_iterator(lengthDifference.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell)))))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			ApplyStretchForce(elongationCoefficient));
}

