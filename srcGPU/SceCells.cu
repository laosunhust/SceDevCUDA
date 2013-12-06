#include "SceCells.h"

__constant__ uint GridDimension[2];
__constant__ double gridSpacing;

double epsilon = 1.0e-12;

/**
 * constructor for SceCells.
 * takes SceNodes, which is a pre-allocated multi-array, as input argument.
 * This might be strange from a design perspective but has a better performance
 * while running on parallel.
 */
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

/**
 * Mark cell node as either active or inactive.
 * left part of the node array will be active and right part will be inactive.
 * the threshold is defined by array @activeNodeCountOfThisCell.
 * e.g. activeNodeCountOfThisCell = {2,3} and  maxNodeOfOneCell = 5,
 *
 */
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

/**
 * This method computes center of all cells.
 * more efficient then simply iterating the cell because of parallel reducing.
 */
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

/**
 * This is a simplified method for cell growth.
 * first step: assign the growth magnitude and direction info that was calculated outside
 *     to internal values
 *
 * second step: use the growth magnitude and dt to update growthProgress
 *
 * third step: use lastCheckPoint and growthProgress to decide whether add point or not
 *
 * fourth step: use growthProgress and growthXDir&growthYDir to compute
 *     expected length along the growth direction.
 *
 * fifth step:  reducing the smallest value and biggest value
 *     a cell's node to its center point
 *
 * sixth step: compute the current length and then
 *     compute its difference with expected length
 *
 * seventh step: use the difference that just computed and growthXDir&growthYDir
 *     to apply stretching force (velocity) on nodes of all cells
 *
 * eighth step: cell move according to the velocity computed
 *
 * ninth step: also add a point if scheduled to grow.
 *     This step does not guarantee success ; If adding new point failed, it will not change
 *     isScheduleToGrow and activeNodeCount;
 */
void SceCells::grow2DSimplified(double dt,
		thrust::device_vector<double> &growthFactorMag,
		thrust::device_vector<double> &growthFactorDirXComp,
		thrust::device_vector<double> &growthFactorDirYComp,
		uint GridDimensionX, uint GridDimensionY, double GridSpacing) {
	//first step: assign the growth magnitude and direction info that was calculated outside
	//to internal values
	thrust::counting_iterator<uint> countingBegin(0);
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
	// fourth step: use growthProgress and growthXDir&growthYDir to compute
	// expected length along the growth direction.
	thrust::transform(growthProgress.begin(),
			growthProgress.begin() + currentActiveCellCount,
			expectedLength.begin(),
			CompuTarLen(cellInitLength, cellFinalLength));
	// fifth step:  reducing the smallest value and biggest value
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
	// sixth step: compute the current length and then
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
	// seventh step: use the difference that just computed and growthXDir&growthYDir
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

	// eighth step: move the cell nodes according to velocity, if the node is active
	thrust::transform_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin()))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeLocX.begin(),
							nodes->nodeLocY.begin())),
			nodes->nodeIsActive.begin(),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeLocX.begin(),
							nodes->nodeLocY.begin())), SaxpyFunctorDim2(dt),
			isTrue());

	// ninth step: also add a point if scheduled to grow.
	// This step does not guarantee success ; If adding new point failed, it will not change
	// isScheduleToGrow and activeNodeCount;
	bool* nodeIsActiveAddress = thrust::raw_pointer_cast(
			&(nodes->nodeIsActive[0]));
	double* nodeXPosAddress = thrust::raw_pointer_cast(&(nodes->nodeLocX[0]));
	double* nodeYPosAddress = thrust::raw_pointer_cast(&(nodes->nodeLocY[0]));

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
}

/**
 * 2D version of cell division.
 * Division process is done by creating two temporary vectors to hold the node information
 * that are going to divide.
 *
 * step 1: based on lengthDifference, expectedLength and growthProgress,
 *     we could obtain whether this cell is ready to divide and assign value to isDivided.
 *
 * step 2. copy those cells that will divide in to the temp vectors created
 *
 * step 3. For each cell in the temp vectors, we sort its nodes by its distance to the
 * corresponding cell center.
 * This step is not very effcient when the number of cells going to divide is big.
 * but this is unlikely to happen because cells will divide according to external chemical signaling
 * and each will have different divide progress.
 *
 * step 4. copy the right part of each cell of the sorted array (temp1) to left part of each cell of
 * another array
 *
 * step 5. transform isActive vector of both temp1 and temp2, making only left part of each cell active.
 *
 * step 6. insert temp2 to the end of the cell array
 *
 * step 7. copy temp1 to the previous position of the cell array.
 *
 * step 8. add activeCellCount of the system.
 */
void SceCells::divide2DSimplified() {
	//TODO: step 1

	// step 2 : copy all cell rank and distance to its corresponding center with divide flag = 1
	uint totalNodeCountForActiveCells = currentActiveCellCount
			* maxNodeOfOneCell;
	uint toBeDivideCount = thrust::reduce(isDivided.begin(),
			isDivided.begin() + currentActiveCellCount);
	uint nodeStorageCount = currentActiveCellCount * maxNodeOfOneCell;
	thrust::device_vector<bool> tmpIsActiveHold1(nodeStorageCount, true);
	thrust::device_vector<double> tmpDistToCenter1(nodeStorageCount, 0.0);
	thrust::device_vector<uint> tmpCellRankHold1(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpXValueHold1(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpYValueHold1(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpZValueHold1(nodeStorageCount, 0.0);

	thrust::device_vector<bool> tmpIsActiveHold2(nodeStorageCount, false);
	thrust::device_vector<double> tmpDistToCenter2(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpXValueHold2(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpYValueHold2(nodeStorageCount, 0.0);
	thrust::device_vector<double> tmpZValueHold2(nodeStorageCount, 0.0);

	thrust::counting_iterator<uint> countingBegin(0);

	// step 2
	thrust::copy_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							distToCenterAlongGrowDir.begin(),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeLocZ.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							distToCenterAlongGrowDir.begin(),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeLocZ.begin()))
					+ totalNodeCountForActiveCells,
			thrust::make_permutation_iterator(isDivided.begin(),
					make_transform_iterator(countingBegin,
							DivideFunctor(maxNodeOfOneCell))),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpCellRankHold1.begin(),
							tmpDistToCenter1.begin(), tmpXValueHold1.begin(),
							tmpYValueHold1.begin(), tmpZValueHold1.begin())),
			isTrue());
	//step 3
	for (uint i = 0; i < toBeDivideCount; i++) {
		thrust::sort_by_key(tmpDistToCenter1.begin() + i * maxNodeOfOneCell,
				tmpDistToCenter1.begin() + (i + 1) * maxNodeOfOneCell,
				thrust::make_zip_iterator(
						thrust::make_tuple(
								tmpXValueHold1.begin() + i * maxNodeOfOneCell,
								tmpYValueHold1.begin() + i * maxNodeOfOneCell,
								tmpZValueHold1.begin()
										+ i * maxNodeOfOneCell)));
	}
	thrust::scatter_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold2.begin(),
							tmpYValueHold2.begin(), tmpZValueHold2.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold2.begin(),
							tmpYValueHold2.begin(), tmpZValueHold2.begin()))
					+ nodeStorageCount,
			make_transform_iterator(countingBegin,
					LeftShiftFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					IsRightSide(maxNodeOfOneCell)),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold2.begin(),
							tmpYValueHold2.begin(), tmpZValueHold2.begin())));
	thrust::transform(tmpIsActiveHold1.begin(),
			tmpIsActiveHold1.begin() + nodeStorageCount,
			tmpIsActiveHold1.begin(), IsLeftSide(maxNodeOfOneCell));
	thrust::transform(tmpIsActiveHold2.begin(),
			tmpIsActiveHold2.begin() + nodeStorageCount,
			tmpIsActiveHold2.begin(), IsLeftSide(maxNodeOfOneCell));
	/// call SceNodes function to add newly divided cells
	nodes->addNewlyDividedCells(tmpXValueHold2, tmpYValueHold2, tmpZValueHold2,
			tmpIsActiveHold2);
	/*
	 thrust::copy(
	 thrust::make_zip_iterator(
	 thrust::make_tuple(tmpIsActiveHold2.begin(),
	 tmpXValueHold2.begin(), tmpYValueHold2.begin(),
	 tmpZValueHold2.begin())),
	 thrust::make_zip_iterator(
	 thrust::make_tuple(tmpIsActiveHold2.end(),
	 tmpXValueHold2.end(), tmpYValueHold2.end(),
	 tmpZValueHold2.end())),
	 thrust::make_zip_iterator(
	 thrust::make_tuple(nodes->nodeIsActive.begin(),
	 nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
	 nodes->nodeLocZ.begin()))
	 + totalNodeCountForActiveCells);
	 */
	//step 7
	thrust::scatter(
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpIsActiveHold1.begin(),
							tmpXValueHold1.begin(), tmpYValueHold1.begin(),
							tmpZValueHold1.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpIsActiveHold1.end(),
							tmpXValueHold1.end(), tmpYValueHold1.end(),
							tmpZValueHold1.end())),
			thrust::make_transform_iterator(
					thrust::make_zip_iterator(
							thrust::make_tuple(countingBegin,
									tmpCellRankHold1.begin())),
					CompuPos(maxNodeOfOneCell)),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeIsActive.begin(),
							nodes->nodeLocX.begin(), nodes->nodeLocY.begin(),
							nodes->nodeLocZ.begin())));

	//step 8
	currentActiveCellCount = currentActiveCellCount + toBeDivideCount;
	nodes->setCurrentActiveCellCount(currentActiveCellCount);
}

/**
 * All cells first grow, then divide.
 * @param dt timestep of the system
 * @param growthFactorMag magnitude of growth
 * @param growthFactorDirXComp x component of the direction of growth
 * @param growthFactorDirYComp y component of the direction of growth
 * @param GridDimensionX number of points in X direction
 * @param GridDimensionY number of points in Y direction
 * @param GridSpacing spacing of the growth grid
 */
void SceCells::growAndDivide(double dt,
		thrust::device_vector<double> &growthFactorMag,
		thrust::device_vector<double> &growthFactorDirXComp,
		thrust::device_vector<double> &growthFactorDirYComp,
		uint GridDimensionX, uint GridDimensionY, double GridSpacing) {
	grow2DSimplified(dt, growthFactorMag, growthFactorDirXComp,
			growthFactorDirYComp, GridDimensionX, GridDimensionY, GridSpacing);
	divide2DSimplified();
}
