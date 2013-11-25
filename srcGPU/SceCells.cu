#include "SceCells.h"

__constant__ uint GridDimension[2];
__constant__ double gridSpacing;

SceCells::SceCells(SceNodes* nodesInput) {
	maxNodeOfOneCell = nodesInput->getMaxNodeOfOneCell();
	maxCellCount = nodesInput->getMaxCellCount();
	maxTotalCellNodeCount = nodesInput->getMaxTotalCellNodeCount();
	currentActiveCellCount = nodesInput->getCurrentActiveCellCount();
	nodes = nodesInput;
	growthProgress.resize(maxCellCount);
	activeNodeCountOfThisCell.resize(maxCellCount);
	lastCheckPoint.resize(maxCellCount);
	isDivided.resize(maxCellCount);
	centerCoordX.resize(maxCellCount);
	centerCoordY.resize(maxCellCount);
	centerCoordZ.resize(maxCellCount);
	cellRanksTmpStorage.resize(maxCellCount);

	xCoordTmp.resize(maxTotalCellNodeCount);
	yCoordTmp.resize(maxTotalCellNodeCount);
	zCoordTmp.resize(maxTotalCellNodeCount);
	cellRanks.resize(maxTotalCellNodeCount);
	activeXPoss.resize(maxTotalCellNodeCount);
	activeYPoss.resize(maxTotalCellNodeCount);
	activeZPoss.resize(maxTotalCellNodeCount);
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

	//second step: use the growth magnitude and dt to update growthProgress

	//third step: use lastCheckPoint and growthProgress to decide whether add point or not

	//fourth step: use growthProgress and growthXDir&growthYDir and apply stretching force (velocity)
}

