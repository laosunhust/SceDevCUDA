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
	chemoCoefficient =
			globalConfigVars.getConfigValue("ChemoCoefficient").toDouble();
	isDivideCriticalRatio = globalConfigVars.getConfigValue(
			"IsDivideCrticalRatio").toDouble();

	maxNodeOfOneCell = nodesInput->getMaxNodeOfOneCell();
	maxCellCount = nodesInput->getMaxCellCount();
	maxTotalCellNodeCount = nodesInput->getMaxTotalCellNodeCount();
	currentActiveCellCount = nodesInput->getCurrentActiveCellCount();
	//cellSpaceForBdry = nodesInput->getCellSpaceForBdry();

	nodes = nodesInput;
	growthProgress.resize(maxCellCount, 0.0);
	expectedLength.resize(maxCellCount, cellInitLength);
	lengthDifference.resize(maxCellCount, 0.0);
	smallestDistance.resize(maxCellCount);
	biggestDistance.resize(maxCellCount);
	activeNodeCountOfThisCell.resize(maxCellCount);
	lastCheckPoint.resize(maxCellCount, 0.0);
	isDivided.resize(maxCellCount);
	//TODO: add cell type initialization
	cellTypes.resize(maxCellCount, MX);
	isScheduledToGrow.resize(maxCellCount, false);
	centerCoordX.resize(maxCellCount);
	centerCoordY.resize(maxCellCount);
	centerCoordZ.resize(maxCellCount);
	cellRanksTmpStorage.resize(maxCellCount);
	growthSpeed.resize(maxCellCount, 0.0);
	growthXDir.resize(maxCellCount);
	growthYDir.resize(maxCellCount);

	//xCoordTmp.resize(maxTotalCellNodeCount);
	//yCoordTmp.resize(maxTotalCellNodeCount);
	//zCoordTmp.resize(maxTotalCellNodeCount);
	cellRanks.resize(maxTotalCellNodeCount);
	activeXPoss.resize(maxTotalCellNodeCount);
	activeYPoss.resize(maxTotalCellNodeCount);
	activeZPoss.resize(maxTotalCellNodeCount);
	distToCenterAlongGrowDir.resize(maxTotalCellNodeCount);

	// reason for adding a small term here is to avoid scenario when checkpoint might add many times
	// up to 0.99999999 which is theoretically 1.0 but not in computer memory. If we don't include
	// this small term we might risk adding one more node.
	growThreshold = 1.0 / (maxNodeOfOneCell - maxNodeOfOneCell / 2) + epsilon;
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
 *     please note that a cell should not grow if its type is boundary.
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

	// if the celltype is boundary, cell should not grow at all.
	thrust::transform_if(cellTypes.begin(),
			cellTypes.begin() + currentActiveCellCount, growthSpeed.begin(),
			GetZero(), IsBoundary());

	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before second step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */

	//second step: use the growth magnitude and dt to update growthProgress
	thrust::transform(growthSpeed.begin(),
			growthSpeed.begin() + currentActiveCellCount,
			growthProgress.begin(), growthProgress.begin(),
			SaxpyFunctorWithMaxOfOne(dt));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before thrid step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
	//third step: use lastCheckPoint and growthProgress to decide whether add point or not
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())) + currentActiveCellCount,
			isScheduledToGrow.begin(), PtCondiOp(growThreshold));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before fourth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
	// fourth step: use growthProgress and growthXDir&growthYDir to compute
	// expected length along the growth direction.
	thrust::transform(growthProgress.begin(),
			growthProgress.begin() + currentActiveCellCount,
			expectedLength.begin(),
			CompuTarLen(cellInitLength, cellFinalLength));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before fifth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before sixth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin())),
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin()))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			ApplyStretchForce(elongationCoefficient));

	// eighth step: move the cell nodes according to velocity, if the node is active

	// move cell nodes
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
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before ninth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
							countingBegin, lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin, lastCheckPoint.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							lastCheckPoint.begin())),
			AddPtOp(maxNodeOfOneCell, addNodeDistance, minDistanceToOtherNode,
					nodeIsActiveAddress, nodeXPosAddress, nodeYPosAddress,
					time(NULL), growThreshold));

}

/**
 * This is a method for cell growth. growth is influened by two chemical fields.
 * first step: assign the growth magnitude and direction info that was calculated outside
 *     to internal values
 *     please note that a cell should not grow if its type is boundary.
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
void SceCells::grow2DTwoRegions(double dt, GrowthDistriMap &region1,
		GrowthDistriMap &region2) {
	//first step: assign the growth magnitude and direction info that was calculated outside
	//to internal values

	thrust::counting_iterator<uint> countingBegin(0);
	double* growthFactorMagAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorMag[0]));
	double* growthFactorDirXAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorDirXComp[0]));
	double* growthFactorDirYAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorDirYComp[0]));

	double* growthFactorMagAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorMag[0]));
	double* growthFactorDirXAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorDirXComp[0]));
	double* growthFactorDirYAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorDirYComp[0]));

	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), cellTypes.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), cellTypes.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(growthSpeed.begin(), growthXDir.begin(),
							growthYDir.begin())),
			LoadChemDataToNode(region1.gridDimensionX, region1.gridDimensionY,
					region1.gridSpacing, growthFactorMagAddress,
					growthFactorDirXAddress, growthFactorDirYAddress,
					region2.gridDimensionX, region2.gridDimensionY,
					region2.gridSpacing, growthFactorMagAddress2,
					growthFactorDirXAddress2, growthFactorDirYAddress2));

	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before second step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */

	//second step: use the growth magnitude and dt to update growthProgress
	thrust::transform(growthSpeed.begin(),
			growthSpeed.begin() + currentActiveCellCount,
			growthProgress.begin(), growthProgress.begin(),
			SaxpyFunctorWithMaxOfOne(dt));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before thrid step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
	//third step: use lastCheckPoint and growthProgress to decide whether add point or not
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())) + currentActiveCellCount,
			isScheduledToGrow.begin(), PtCondiOp(growThreshold));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before fourth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
	// fourth step: use growthProgress and growthXDir&growthYDir to compute
	// expected length along the growth direction.
	thrust::transform(growthProgress.begin(),
			growthProgress.begin() + currentActiveCellCount,
			expectedLength.begin(),
			CompuTarLen(cellInitLength, cellFinalLength));
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before fifth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before sixth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin())),
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin()))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			ApplyStretchForce(elongationCoefficient));

	//this is only an attempt. add chemotaxis to cells
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(growthSpeed.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(growthSpeed.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin(), nodes->nodeVelY.begin()))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			ApplyChemoVel(chemoCoefficient));
	// eighth step: move the cell nodes according to velocity, if the node is active
	// move cell nodes
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
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeIsActive.begin(),
							make_permutation_iterator(cellTypes.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))))),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeLocX.begin(),
							nodes->nodeLocY.begin())), SaxpyFunctorDim2(dt),
			isActiveNoneBdry());
	/*
	 xTmp = nodes->nodeLocX;
	 for (uint i = 0; i < xTmp.size(); i++) {
	 if (isnan(xTmp[i])) {
	 std::cout << "nan detected before ninth step in grow" << std::endl;
	 exit(0);
	 }
	 }
	 */
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
							countingBegin, lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin, lastCheckPoint.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							lastCheckPoint.begin())),
			AddPtOp(maxNodeOfOneCell, addNodeDistance, minDistanceToOtherNode,
					nodeIsActiveAddress, nodeXPosAddress, nodeYPosAddress,
					time(NULL), growThreshold));

}

/**
 * 2D version of cell division.
 * Division process is done by creating two temporary vectors to hold the node information
 * that are going to divide.
 *
 * step 1: based on lengthDifference, expectedLength and growthProgress,
 *     this process determines whether a certain cell is ready to divide and then assign
 *     a boolean value to isDivided.
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
 *
 * step 9. mark isDivide of all cells to false.
 */

//TODO: also pay attention to number of active nodes per cell. This seems to be omitted.
void SceCells::divide2DSimplified() {
	// step 1
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(lengthDifference.begin(),
							expectedLength.begin(), growthProgress.begin(),
							activeNodeCountOfThisCell.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(lengthDifference.begin(),
							expectedLength.begin(), growthProgress.begin(),
							activeNodeCountOfThisCell.begin()))
					+ currentActiveCellCount, isDivided.begin(),
			CompuIsDivide(isDivideCriticalRatio, maxNodeOfOneCell));

	// step 2 : copy all cell rank and distance to its corresponding center with divide flag = 1
	uint totalNodeCountForActiveCells = currentActiveCellCount
			* maxNodeOfOneCell;
	// sum all bool values which indicate whether the cell is going to divide.
	// toBeDivideCount is the total number of cells going to divide.
	uint toBeDivideCount = thrust::reduce(isDivided.begin(),
			isDivided.begin() + currentActiveCellCount, (uint) (0));
	std::cout << "total number of cells to divide: " << toBeDivideCount
			<< std::endl;
	uint nodeStorageCount = toBeDivideCount * maxNodeOfOneCell;
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

	thrust::device_vector<CellType> tmpCellTypes(toBeDivideCount);

	thrust::counting_iterator<uint> countingBegin(0);

	// step 2 , continued
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

	// step 2, continued, copy cell type to new cells
	thrust::copy_if(cellTypes.begin(),
			cellTypes.begin() + currentActiveCellCount, isDivided.begin(),
			tmpCellTypes.begin(), isTrue());
	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<double> hostTmpDist = tmpDistToCenter1;
	 thrust::host_vector<double> hostXCoord = tmpXValueHold1;
	 thrust::host_vector<double> hostYCoord = tmpYValueHold1;
	 thrust::host_vector<double> hostZCoord = tmpZValueHold1;
	 std::cout << "In the begining, numbers:" << std::endl;
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << "(" << hostTmpDist[i] << ",,(" << hostXCoord[i] << ","
	 << hostYCoord[i] << "," << hostZCoord[i] << ")) # "
	 << std::endl;
	 }
	 //int jj;
	 //std::cin >> jj;
	 }
	 */

	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<double> hostTmpDist = tmpDistToCenter1;
	 thrust::host_vector<double> hostXCoord = tmpXValueHold1;
	 thrust::host_vector<double> hostYCoord = tmpYValueHold1;
	 thrust::host_vector<double> hostZCoord = tmpZValueHold1;
	 std::cout << "before sorting, numbers:" << std::endl;
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << "(" << hostTmpDist[i] << ",,(" << hostXCoord[i] << ","
	 << hostYCoord[i] << "," << hostZCoord[i] << ")) # "
	 << std::endl;
	 }
	 }
	 */
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
	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<double> hostTmpDist = tmpDistToCenter1;
	 thrust::host_vector<double> hostXCoord = tmpXValueHold1;
	 thrust::host_vector<double> hostYCoord = tmpYValueHold1;
	 thrust::host_vector<double> hostZCoord = tmpZValueHold1;
	 std::cout << "after sorting, numbers:" << std::endl;
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << "(" << hostTmpDist[i] << ",,(" << hostXCoord[i] << ","
	 << hostYCoord[i] << "," << hostZCoord[i] << ")) # "
	 << std::endl;
	 }

	 }
	 */
	//step 4.
	thrust::scatter_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold1.begin(),
							tmpYValueHold1.begin(), tmpZValueHold1.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold1.end(),
							tmpYValueHold1.end(), tmpZValueHold1.end())),
			make_transform_iterator(countingBegin,
					LeftShiftFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					IsRightSide(maxNodeOfOneCell)),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold2.begin(),
							tmpYValueHold2.begin(), tmpZValueHold2.begin())));

	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<double> hostTmpDist = tmpDistToCenter1;
	 thrust::host_vector<double> hostXCoord = tmpXValueHold1;
	 thrust::host_vector<double> hostYCoord = tmpYValueHold1;
	 thrust::host_vector<double> hostZCoord = tmpZValueHold1;
	 std::cout << "after step 4, numbers:" << std::endl;
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << "(" << hostTmpDist[i] << ",,(" << hostXCoord[i] << ","
	 << hostYCoord[i] << "," << hostZCoord[i] << ")) # "
	 << std::endl;
	 }

	 }
	 if (toBeDivideCount == 2) {
	 exit(0);
	 }
	 */
	//step 5.
	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<bool> isActive1Host = tmpIsActiveHold1;
	 thrust::host_vector<bool> isActive2Host = tmpIsActiveHold2;
	 std::cout << "before transform, active state 1:" << std::endl;
	 std::cout << "(";
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << isActive1Host[i] << ", ";
	 }
	 std::cout << ")" << std::endl;
	 std::cout << "before transform, active state 2:" << std::endl;
	 std::cout << "(";
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << isActive1Host[2] << ", ";
	 }
	 std::cout << ")" << std::endl;
	 }
	 */
	thrust::transform(countingBegin, countingBegin + nodeStorageCount,
			tmpIsActiveHold1.begin(), IsLeftSide(maxNodeOfOneCell));
	thrust::transform(countingBegin, countingBegin + nodeStorageCount,
			tmpIsActiveHold2.begin(), IsLeftSide(maxNodeOfOneCell));

	/*
	 if (toBeDivideCount != 0) {
	 thrust::host_vector<bool> isActive1Host = tmpIsActiveHold1;
	 thrust::host_vector<bool> isActive2Host = tmpIsActiveHold2;
	 std::cout << "after transform, active state 1:" << std::endl;
	 std::cout << "(";
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << isActive2Host[i] << ", ";
	 }
	 std::cout << ")" << std::endl;
	 std::cout << "after transform, active state 2:" << std::endl;
	 std::cout << "(";
	 for (uint i = 0; i < nodeStorageCount; i++) {
	 std::cout << isActive2Host[i] << ", ";
	 }
	 std::cout << ")" << std::endl;
	 }
	 */
	if (toBeDivideCount != 0) {
		std::cout << "before insert, active cell count in nodes:"
				<< nodes->getCurrentActiveCellCount() << std::endl;
	}
	/// step 6. call SceNodes function to add newly divided cells
	nodes->addNewlyDividedCells(tmpXValueHold2, tmpYValueHold2, tmpZValueHold2,
			tmpIsActiveHold2);

	//if (toBeDivideCount != 0) {
	//	std::cout << "after insert, active cell count in nodes:"
	//			<< nodes->getCurrentActiveCellCount() << std::endl;
	//}
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
	thrust::constant_iterator<uint> initCellCount(maxNodeOfOneCell / 2);
	thrust::constant_iterator<double> initGrowthProgress(0.0);
	thrust::scatter(initCellCount, initCellCount + toBeDivideCount,
			tmpCellRankHold1.begin(), activeNodeCountOfThisCell.begin());

	thrust::scatter(initGrowthProgress, initGrowthProgress + toBeDivideCount,
			tmpCellRankHold1.begin(), growthProgress.begin());
	thrust::scatter(initGrowthProgress, initGrowthProgress + toBeDivideCount,
			tmpCellRankHold1.begin(), lastCheckPoint.begin());

	thrust::fill(activeNodeCountOfThisCell.begin() + currentActiveCellCount,
			activeNodeCountOfThisCell.begin() + currentActiveCellCount
					+ toBeDivideCount, maxNodeOfOneCell / 2);

	//step 8
	currentActiveCellCount = currentActiveCellCount + toBeDivideCount;
	nodes->setCurrentActiveCellCount(currentActiveCellCount);
	//step 9.
	thrust::fill(isDivided.begin(), isDivided.begin() + currentActiveCellCount,
			false);

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
void SceCells::growAndDivide(double dt, GrowthDistriMap &region1,
		GrowthDistriMap &region2) {
	//thrust::host_vector<double> xTmp = nodes->nodeLocX;
	//for (uint i = 0; i < xTmp.size(); i++) {
	//	if (isnan(xTmp[i])) {
	//		std::cout << "nan detected before compute center position"
	//				<< std::endl;
	//		exit(0);
	//	}
	//}
	std::cout << "In SceCells, before compute center:" << std::endl;
	computeCenterPos();
	//xTmp = nodes->nodeLocX;
	//for (uint i = 0; i < xTmp.size(); i++) {
	//	if (isnan(xTmp[i])) {
	//		std::cout << "nan detected before grow 2D "
	//				<< std::endl;
	//		exit(0);
	//	}
	//}
	std::cout << "In SceCells, before grow 2D:" << std::endl;
	//grow2DSimplified(dt, growthFactorMag, growthFactorDirXComp,
	//		growthFactorDirYComp, GridDimensionX, GridDimensionY, GridSpacing);
	grow2DTwoRegions(dt, region1, region2);
	//xTmp = nodes->nodeLocX;
	//for (uint i = 0; i < xTmp.size(); i++) {
	//if (isnan(xTmp[i])) {
	//		std::cout << "nan detected before distri info"
	//				<< std::endl;
	//		exit(0);
	//	}
	//}
	std::cout << "In SceCells, before distribute is active info:" << std::endl;
	distributeIsActiveInfo();
	std::cout << "In SceCells, before divide 2D:" << std::endl;

	//for (uint i = 0; i < xTmp.size(); i++) {
	//	if (isnan(xTmp[i])) {
	//		std::cout << "nan detected before division" << std::endl;
	//		exit(0);
	//	}
	//}
	divide2DSimplified();
	std::cout << "In SceCells, before distribute is active info:" << std::endl;
	distributeIsActiveInfo();
}

/**
 * constructor for SceCells_M.
 * takes SceNodes, which is a pre-allocated multi-array, as input argument.
 * This might be strange from a design perspective but has a better performance
 * while running on parallel.
 */
SceCells_M::SceCells_M(SceNodes* nodesInput) :
		maxNodeOfOneCell(nodesInput->getMaxNodeOfOneCell()), countingBegin(0), initCellCount(
				maxNodeOfOneCell / 2), initGrowthProgress(0.0) {

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
	chemoCoefficient =
			globalConfigVars.getConfigValue("ChemoCoefficient").toDouble();
	isDivideCriticalRatio = globalConfigVars.getConfigValue(
			"IsDivideCrticalRatio").toDouble();

	beginPosOfBdry = 0;
	//maxNodeOfBdry = nodesInput->cellSpaceForBdry;

	beginPosOfEpi = 1;
	maxNodeOfEpi = nodesInput->maxProfileNodeCount; // represents maximum number of nodes of epithilum layer.
	beginPosOfEpiNode = nodesInput->startPosProfile;

	maxNodeOfECM = nodesInput->maxNodePerECM; // represents maximum number of nodes per ECM
	// represents begining position of ECM (in node perspective) value is 2
	beginPosOfECM = 2;
	// represents begining position of ECM (in cell perspective)
	beginPosOfECMNode = nodesInput->startPosECM;
	maxECMCount = nodesInput->maxECMCount;  // represents maximum number of ECM.

	beginPosOfCells = beginPosOfECM + maxECMCount; // represents begining position of cells (in cell perspective)
	beginPosOfCellsNode = nodesInput->startPosCells; // represents begining position of cells (in node perspective)
	maxNodeOfOneCell = nodesInput->getMaxNodeOfOneCell();
	// maxCellCount only take FNM and MX cells into consideration.
	maxCellCount = nodesInput->getMaxCellCount();
	// maxCellCountAll counts all cells include pseduo cells
	maxCellCountAll = nodesInput->getMaxCellCountAll();
	maxTotalNodeCountCellOnly = nodesInput->getMaxTotalCellNodeCount();
	currentActiveCellCount = nodesInput->getCurrentActiveCellCount();

	// cellSpaceForBdry = nodesInput->getCellSpaceForBdry();

	nodes = nodesInput;
	growthProgress.resize(maxCellCount, 0.0);
	expectedLength.resize(maxCellCount, cellInitLength);
	lengthDifference.resize(maxCellCount, 0.0);
	smallestDistance.resize(maxCellCount);
	biggestDistance.resize(maxCellCount);
	activeNodeCountOfThisCell.resize(maxCellCount);
	lastCheckPoint.resize(maxCellCount, 0.0);
	isDivided.resize(maxCellCount);
	cellTypes.resize(maxCellCount, MX);
	isScheduledToGrow.resize(maxCellCount, false);
	centerCoordX.resize(maxCellCount);
	centerCoordY.resize(maxCellCount);
	centerCoordZ.resize(maxCellCount);
	cellRanksTmpStorage.resize(maxCellCount);
	growthSpeed.resize(maxCellCount, 0.0);
	growthXDir.resize(maxCellCount);
	growthYDir.resize(maxCellCount);

	//xCoordTmp.resize(maxTotalCellNodeCount);
	//yCoordTmp.resize(maxTotalCellNodeCount);
	//zCoordTmp.resize(maxTotalCellNodeCount);
	cellRanks.resize(maxTotalNodeCountCellOnly);
	activeXPoss.resize(maxTotalNodeCountCellOnly);
	activeYPoss.resize(maxTotalNodeCountCellOnly);
	activeZPoss.resize(maxTotalNodeCountCellOnly);
	distToCenterAlongGrowDir.resize(maxTotalNodeCountCellOnly);

	// reason for adding a small term here is to avoid scenario when checkpoint might add many times
	// up to 0.99999999 which is theoretically 1.0 but not in computer memory. If we don't include
	// this small term we might risk adding one more node.
	growThreshold = 1.0 / (maxNodeOfOneCell - maxNodeOfOneCell / 2) + epsilon;

	nodeIsActiveAddress = thrust::raw_pointer_cast(
			&(nodes->nodeIsActive[beginPosOfCellsNode]));
	nodeXPosAddress = thrust::raw_pointer_cast(
			&(nodes->nodeLocX[beginPosOfCellsNode]));
	nodeYPosAddress = thrust::raw_pointer_cast(
			&(nodes->nodeLocY[beginPosOfCellsNode]));

}

/**
 * This is a method for cell growth. growth is influened by two chemical fields.
 * first step: assign the growth magnitude and direction info that was calculated outside
 *     to internal values
 *     please note that a cell should not grow if its type is boundary.
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
void SceCells_M::grow2DTwoRegions(double d_t, GrowthDistriMap &region1,
		GrowthDistriMap &region2) {

	dt = d_t;

	// obtain pointer address for first region
	growthFactorMagAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorMag[0]));
	growthFactorDirXAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorDirXComp[0]));
	growthFactorDirYAddress = thrust::raw_pointer_cast(
			&(region1.growthFactorDirYComp[0]));

	// obtain pointer address for second region
	growthFactorMagAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorMag[0]));
	growthFactorDirXAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorDirXComp[0]));
	growthFactorDirYAddress2 = thrust::raw_pointer_cast(
			&(region2.growthFactorDirYComp[0]));

	totalNodeCountForActiveCells = currentActiveCellCount * maxNodeOfOneCell;

	copyGrowInfoFromGridToCells(region1, region2);
	updateGrowthProgress();
	decideIsScheduleToGrow();
	computeCellTargetLength();
	computeDistToCellCenter();
	findMinAndMaxDistToCenter();
	computeLenDiffExpCur();
	stretchCellGivenLenDiff();
	cellChemotaxis();
	addPointIfScheduledToGrow();
}

/**
 * we need to copy the growth information from grid for chemical to cell nodes.
 *
 * checked.
 */
void SceCells_M::copyGrowInfoFromGridToCells(GrowthDistriMap &region1,
		GrowthDistriMap &region2) {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), cellTypes.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(centerCoordX.begin(),
							centerCoordY.begin(), cellTypes.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(growthSpeed.begin(), growthXDir.begin(),
							growthYDir.begin())),
			LoadChemDataToNode(region1.gridDimensionX, region1.gridDimensionY,
					region1.gridSpacing, growthFactorMagAddress,
					growthFactorDirXAddress, growthFactorDirYAddress,
					region2.gridDimensionX, region2.gridDimensionY,
					region2.gridSpacing, growthFactorMagAddress2,
					growthFactorDirXAddress2, growthFactorDirYAddress2));
}

/**
 * Use the growth magnitude and dt to update growthProgress.
 */
void SceCells_M::updateGrowthProgress() {
	thrust::transform(growthSpeed.begin(),
			growthSpeed.begin() + currentActiveCellCount,
			growthProgress.begin(), growthProgress.begin(),
			SaxpyFunctorWithMaxOfOne(dt));
}

/**
 * Decide if the cells are going to add a node or not.
 * Use lastCheckPoint and growthProgress to decide whether add point or not
 */
void SceCells_M::decideIsScheduleToGrow() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(growthProgress.begin(),
							lastCheckPoint.begin())) + currentActiveCellCount,
			isScheduledToGrow.begin(), PtCondiOp(growThreshold));
}

/**
 * Calculate target length of cell given the cell growth progress.
 * length is along the growth direction.
 */
void SceCells_M::computeCellTargetLength() {
	thrust::transform(growthProgress.begin(),
			growthProgress.begin() + currentActiveCellCount,
			expectedLength.begin(),
			CompuTarLen(cellInitLength, cellFinalLength));
}

/**
 * Compute distance of each node to its corresponding cell center.
 * The distantce could be either positive or negative, depending on the pre-defined
 * growth direction.
 */
void SceCells_M::computeDistToCellCenter() {
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
							nodes->nodeLocX.begin() + beginPosOfCellsNode,
							nodes->nodeLocY.begin() + beginPosOfCellsNode,
							nodes->nodeIsActive.begin() + beginPosOfCellsNode)),
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
							nodes->nodeLocX.begin() + beginPosOfCellsNode,
							nodes->nodeLocY.begin() + beginPosOfCellsNode,
							nodes->nodeIsActive.begin() + beginPosOfCellsNode))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), CompuDist());
}

/**
 * For nodes of each cell, find the maximum and minimum distance to the center.
 * We will then calculate the current length of a cell along its growth direction
 * using max and min distance to the center.
 */
void SceCells_M::findMinAndMaxDistToCenter() {
	thrust::reduce_by_key(
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), cellRanksTmpStorage.begin(),
			smallestDistance.begin(), thrust::equal_to<uint>(),
			thrust::minimum<double>());
	// for nodes of each cell, find the maximum distance from the node to the corresponding
	// cell center along the pre-defined growth direction.
	thrust::reduce_by_key(
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					DivideFunctor(maxNodeOfOneCell))
					+ totalNodeCountForActiveCells,
			distToCenterAlongGrowDir.begin(), cellRanksTmpStorage.begin(),
			biggestDistance.begin(), thrust::equal_to<uint>(),
			thrust::maximum<double>());
}

/**
 * Compute the difference for cells between their expected length and current length.
 */
void SceCells_M::computeLenDiffExpCur() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(expectedLength.begin(),
							smallestDistance.begin(), biggestDistance.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(expectedLength.begin(),
							smallestDistance.begin(), biggestDistance.begin()))
					+ currentActiveCellCount, lengthDifference.begin(),
			CompuDiff());
}

/**
 * Use the difference that just computed and growthXDir&growthYDir
 * to apply stretching force (velocity) on nodes of all cells
 */
void SceCells_M::stretchCellGivenLenDiff() {
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode)),
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
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode)),
			ApplyStretchForce(elongationCoefficient));
}
/**
 * This is just an attempt. Cells move according to chemicals.
 */
void SceCells_M::cellChemotaxis() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(growthSpeed.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode)),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_permutation_iterator(growthSpeed.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthXDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							make_permutation_iterator(growthYDir.begin(),
									make_transform_iterator(countingBegin,
											DivideFunctor(maxNodeOfOneCell))),
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode))
					+ totalNodeCountForActiveCells,
			thrust::make_zip_iterator(
					thrust::make_tuple(
							nodes->nodeVelX.begin() + beginPosOfCellsNode,
							nodes->nodeVelY.begin() + beginPosOfCellsNode)),
			ApplyChemoVel(chemoCoefficient));
}
/**
 * Adjust the velocities of nodes.
 * For example, velocity of boundary nodes must be zero.
 */
void SceCells_M::adjustNodeVel() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin(),
							nodes->nodeIsActive.begin(),
							nodes->nodeCellType.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin(),
							nodes->nodeIsActive.begin(),
							nodes->nodeCellType.begin()))
					+ totalNodeCountForActiveCells + beginPosOfCellsNode,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())), VelocityModifier());
}
/**
 * Move nodes according to the velocity we just adjusted.
 */
void SceCells_M::moveNodes() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeVelX.begin(),
							nodes->nodeVelY.begin()))
					+ totalNodeCountForActiveCells + beginPosOfCellsNode,
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeLocX.begin(),
							nodes->nodeLocY.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(nodes->nodeLocX.begin(),
							nodes->nodeLocY.begin())), SaxpyFunctorDim2(dt));
}
/**
 * Add a point to a cell if it is scheduled to grow.
 * This step does not guarantee success ; If adding new point failed, it will not change
 * isScheduleToGrow and activeNodeCount;
 */
void SceCells_M::addPointIfScheduledToGrow() {
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin, lastCheckPoint.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							centerCoordX.begin(), centerCoordY.begin(),
							countingBegin, lastCheckPoint.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin(),
							lastCheckPoint.begin())),
			AddPtOp(maxNodeOfOneCell, addNodeDistance, minDistanceToOtherNode,
					nodeIsActiveAddress, nodeXPosAddress, nodeYPosAddress,
					time(NULL), growThreshold));
}

/**
 * To run all the cell level logics.
 * First step we got center positions of cells.
 * Grow.
 */
void SceCells_M::runAllCellLevelLogics(double dt, GrowthDistriMap &region1,
		GrowthDistriMap &region2) {
	computeCenterPos();
	grow2DTwoRegions(dt, region1, region2);
	distributeIsActiveInfo();
	divide2DSimplified();
	distributeIsActiveInfo();
	allComponentsMove();
}

void SceCells_M::allComponentsMove() {
	adjustNodeVel();
	moveNodes();
}

/**
 * Mark cell node as either active or inactive.
 * left part of the node array will be active and right part will be inactive.
 * the threshold is defined by array @activeNodeCountOfThisCell.
 * e.g. activeNodeCountOfThisCell = {2,3} and  maxNodeOfOneCell = 5,
 *
 * @Checked.
 */
void SceCells_M::distributeIsActiveInfo() {
	//uint totalNodeCountForActiveCells = currentActiveCellCount
	//		* maxNodeOfOneCell;
	thrust::counting_iterator < uint > countingBegin(0);
	thrust::counting_iterator<uint> countingEnd(totalNodeCountForActiveCells);
	thrust::transform(
			thrust::make_transform_iterator(countingBegin,
					ModuloFunctor(maxNodeOfOneCell)),
			thrust::make_transform_iterator(countingEnd,
					ModuloFunctor(maxNodeOfOneCell)),
			thrust::make_permutation_iterator(activeNodeCountOfThisCell.begin(),
					make_transform_iterator(countingBegin,
							DivideFunctor(maxNodeOfOneCell))),
			nodes->nodeIsActive.begin() + beginPosOfCells,
			thrust::less<uint>());
}

void SceCells_M::distributeIsCellRank() {
	//uint totalNodeCountForActiveCells = currentActiveCellCount
	//		* maxNodeOfOneCell;
	thrust::counting_iterator < uint > countingBegin(0);
	thrust::counting_iterator<uint> countingCellEnd(
			totalNodeCountForActiveCells);

	thrust::counting_iterator<uint> countingECMEnd(countingECMEnd);

	// only computes the cell ranks of cells. the rest remain unchanged.
	thrust::transform(countingBegin, countingCellEnd,
			nodes->nodeCellRank.begin() + beginPosOfCells,
			DivideFunctor(maxNodeOfOneCell));
}

/**
 * This method computes center of all cells.
 * more efficient then simply iterating the cell because of parallel reducing.
 *
 * @Checked.
 */
void SceCells_M::computeCenterPos() {
//uint totalNodeCountForActiveCells = currentActiveCellCount
//		* maxNodeOfOneCell;
	thrust::counting_iterator < uint > countingBegin(0);
	thrust::counting_iterator<uint> countingEnd(totalNodeCountForActiveCells);
	uint totalNumberOfActiveNodes = thrust::reduce(
			activeNodeCountOfThisCell.begin(),
			activeNodeCountOfThisCell.begin() + currentActiveCellCount);

	thrust::copy_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							nodes->nodeLocX.begin() + beginPosOfCells,
							nodes->nodeLocY.begin() + beginPosOfCells,
							nodes->nodeLocZ.begin() + beginPosOfCells)),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							nodes->nodeLocX.begin() + beginPosOfCells,
							nodes->nodeLocY.begin() + beginPosOfCells,
							nodes->nodeLocZ.begin() + beginPosOfCells))
					+ totalNodeCountForActiveCells,
			nodes->nodeIsActive.begin() + beginPosOfCells,
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
 * 2D version of cell division.
 * Division process is done by creating two temporary vectors to hold the node information
 * that are going to divide.
 *
 * step 1: based on lengthDifference, expectedLength and growthProgress,
 *     this process determines whether a certain cell is ready to divide and then assign
 *     a boolean value to isDivided.
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
 *
 * step 9. mark isDivide of all cells to false.
 */

//TODO: also pay attention to number of active nodes per cell. This seems to be omitted.
void SceCells_M::divide2DSimplified() {
	decideIfGoingToDivide();
	copyCellsPreDivision();
	sortNodesAccordingToDist();
	copyLeftAndRightToSeperateArrays();
	transformIsActiveArrayOfBothArrays();
	addSecondArrayToCellArray();
	copyFirstArrayToPreviousPos();
	updateActiveCellCount();
	markIsDivideFalse();
}

void SceCells_M::decideIfGoingToDivide() {
// step 1
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(lengthDifference.begin(),
							expectedLength.begin(), growthProgress.begin(),
							activeNodeCountOfThisCell.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(lengthDifference.begin(),
							expectedLength.begin(), growthProgress.begin(),
							activeNodeCountOfThisCell.begin()))
					+ currentActiveCellCount, isDivided.begin(),
			CompuIsDivide(isDivideCriticalRatio, maxNodeOfOneCell));
}

void SceCells_M::copyCellsPreDivision() {
// step 2 : copy all cell rank and distance to its corresponding center with divide flag = 1
	totalNodeCountForActiveCells = currentActiveCellCount * maxNodeOfOneCell;
// sum all bool values which indicate whether the cell is going to divide.
// toBeDivideCount is the total number of cells going to divide.
	toBeDivideCount = thrust::reduce(isDivided.begin(),
			isDivided.begin() + currentActiveCellCount, (uint) (0));
	nodeStorageCount = toBeDivideCount * maxNodeOfOneCell;
	tmpIsActiveHold1 = thrust::device_vector<bool>(nodeStorageCount, true);
	tmpDistToCenter1 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpCellRankHold1 = thrust::device_vector < uint > (nodeStorageCount, 0.0);
	tmpXValueHold1 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpYValueHold1 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpZValueHold1 = thrust::device_vector<double>(nodeStorageCount, 0.0);

	tmpIsActiveHold2 = thrust::device_vector<bool>(nodeStorageCount, false);
	tmpDistToCenter2 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpXValueHold2 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpYValueHold2 = thrust::device_vector<double>(nodeStorageCount, 0.0);
	tmpZValueHold2 = thrust::device_vector<double>(nodeStorageCount, 0.0);

	tmpCellTypes = thrust::device_vector < CellType > (toBeDivideCount);

// step 2 , continued
	thrust::copy_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							distToCenterAlongGrowDir.begin(),
							nodes->nodeLocX.begin() + beginPosOfCells,
							nodes->nodeLocY.begin() + beginPosOfCells,
							nodes->nodeLocZ.begin() + beginPosOfCells)),
			thrust::make_zip_iterator(
					thrust::make_tuple(
							make_transform_iterator(countingBegin,
									DivideFunctor(maxNodeOfOneCell)),
							distToCenterAlongGrowDir.begin(),
							nodes->nodeLocX.begin() + beginPosOfCells,
							nodes->nodeLocY.begin() + beginPosOfCells,
							nodes->nodeLocZ.begin() + beginPosOfCells))
					+ totalNodeCountForActiveCells,
			thrust::make_permutation_iterator(isDivided.begin(),
					make_transform_iterator(countingBegin,
							DivideFunctor(maxNodeOfOneCell))),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpCellRankHold1.begin(),
							tmpDistToCenter1.begin(), tmpXValueHold1.begin(),
							tmpYValueHold1.begin(), tmpZValueHold1.begin())),
			isTrue());

// step 2, continued, copy cell type to new cell
	thrust::copy_if(cellTypes.begin(),
			cellTypes.begin() + currentActiveCellCount, isDivided.begin(),
			tmpCellTypes.begin(), isTrue());
}

/**
 * performance wise, this implementation is not the best because I can use only one sort_by_key
 * with speciialized comparision operator. However, This implementation is more robust and won't
 * cause serious delay of the program.
 */
void SceCells_M::sortNodesAccordingToDist() {
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
}

/**
 * scatter_if() is a thrust function.
 * inputIter1 first,
 * inputIter1 last,
 * inputIter2 map,
 * inputIter3 stencil
 * randomAccessIter output
 */
void SceCells_M::copyLeftAndRightToSeperateArrays() {
//step 4.
	thrust::scatter_if(
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold1.begin(),
							tmpYValueHold1.begin(), tmpZValueHold1.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold1.end(),
							tmpYValueHold1.end(), tmpZValueHold1.end())),
			make_transform_iterator(countingBegin,
					LeftShiftFunctor(maxNodeOfOneCell)),
			make_transform_iterator(countingBegin,
					IsRightSide(maxNodeOfOneCell)),
			thrust::make_zip_iterator(
					thrust::make_tuple(tmpXValueHold2.begin(),
							tmpYValueHold2.begin(), tmpZValueHold2.begin())));
}

void SceCells_M::transformIsActiveArrayOfBothArrays() {
	thrust::transform(countingBegin, countingBegin + nodeStorageCount,
			tmpIsActiveHold1.begin(), IsLeftSide(maxNodeOfOneCell));
	thrust::transform(countingBegin, countingBegin + nodeStorageCount,
			tmpIsActiveHold2.begin(), IsLeftSide(maxNodeOfOneCell));
	if (toBeDivideCount != 0) {
		std::cout << "before insert, active cell count in nodes:"
				<< nodes->getCurrentActiveCellCount() << std::endl;
	}
}

void SceCells_M::addSecondArrayToCellArray() {
/// step 6. call SceNodes function to add newly divided cells
	nodes->addNewlyDividedCells(tmpXValueHold2, tmpYValueHold2, tmpZValueHold2,
			tmpIsActiveHold2, tmpCellTypes);
}

void SceCells_M::copyFirstArrayToPreviousPos() {
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
					thrust::make_tuple(
							nodes->nodeIsActive.begin() + beginPosOfCells,
							nodes->nodeLocX.begin() + beginPosOfCells,
							nodes->nodeLocY.begin() + beginPosOfCells,
							nodes->nodeLocZ.begin() + beginPosOfCells)));

	thrust::scatter(initCellCount, initCellCount + toBeDivideCount,
			tmpCellRankHold1.begin(), activeNodeCountOfThisCell.begin());

	thrust::scatter(initGrowthProgress, initGrowthProgress + toBeDivideCount,
			tmpCellRankHold1.begin(), growthProgress.begin());
	thrust::scatter(initGrowthProgress, initGrowthProgress + toBeDivideCount,
			tmpCellRankHold1.begin(), lastCheckPoint.begin());

	thrust::fill(activeNodeCountOfThisCell.begin() + currentActiveCellCount,
			activeNodeCountOfThisCell.begin() + currentActiveCellCount
					+ toBeDivideCount, maxNodeOfOneCell / 2);

}

void SceCells_M::updateActiveCellCount() {
	currentActiveCellCount = currentActiveCellCount + toBeDivideCount;
	nodes->setCurrentActiveCellCount(currentActiveCellCount);
}

void SceCells_M::markIsDivideFalse() {
	thrust::fill(isDivided.begin(), isDivided.begin() + currentActiveCellCount,
			false);
}
