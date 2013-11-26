#include "SceCells.h"

__constant__ uint GridDimension[2];
__constant__ double gridSpacing;

double epsilon = 1.0e-12;

SceCells::SceCells(SceNodes* nodesInput) {
	maxNodeOfOneCell = nodesInput->getMaxNodeOfOneCell();
	maxCellCount = nodesInput->getMaxCellCount();
	maxTotalCellNodeCount = nodesInput->getMaxTotalCellNodeCount();
	currentActiveCellCount = nodesInput->getCurrentActiveCellCount();
	nodes = nodesInput;
	growthProgress.resize(maxCellCount, 0.0);
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
							make_transform_iterator(countingEnd,
									DivideFunctor(maxNodeOfOneCell)),
							nodes->nodeLocX.begin()
									+ totalNodeCountForActiveCells,
							nodes->nodeLocY.begin()
									+ totalNodeCountForActiveCells,
							nodes->nodeLocZ.begin()
									+ totalNodeCountForActiveCells)),
			nodes->nodeIsActive.begin(),
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
	thrust::transform(
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin())),
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin()))
					+ currentActiveCellCount,
			thrust::make_zip_iterator(
					thrust::make_tuple(isScheduledToGrow.begin(),
							activeNodeCountOfThisCell.begin())), AddPtOp());
	// fifth step: use growthProgress and growthXDir&growthYDir to compute
	// expected length along the growth direction.

	// sixth step: compute current length along the growth direction
	// and its difference with expected length

	// seventh step: use the difference that just computed and growthXDir&growthYDir
	// to apply stretching force (velocity) on nodes of all cells
	thrust::transform();
}

