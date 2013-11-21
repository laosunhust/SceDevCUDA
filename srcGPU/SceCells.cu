#include "SceCells.h"

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

	xCoordTmp.resize(maxTotalCellNodeCount);
	yCoordTmp.resize(maxTotalCellNodeCount);
	zCoordTmp.resize(maxTotalCellNodeCount);

}
