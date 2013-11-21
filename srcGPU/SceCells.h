#ifndef SCECELLS_H_
#define SCECELLS_H_

#include "SceNodes.h"

class SceCells {
public:
	// @maxNodeOfOneCell represents maximum number of nodes per cell
	uint maxNodeOfOneCell;
	// @maxCellCount represents maximum number of cells in the system
	uint maxCellCount;
	uint maxTotalCellNodeCount;
	uint currentActiveCellCount;

	SceNodes* nodes;

	// values of these vectors corresponds to each cell.
	// which means these vectors have size of maxCellCount
	thrust::device_vector<double> growthProgress;
	thrust::device_vector<uint> activeNodeCountOfThisCell;
	thrust::device_vector<double> lastCheckPoint;
	thrust::device_vector<bool> isDivided;
	thrust::device_vector<double> centerCoordX;
	thrust::device_vector<double> centerCoordY;
	thrust::device_vector<double> centerCoordZ;

	// these tmp coordinates will be temporary storage for division info
	// their size will be the same with maximum node count.
	thrust::device_vector<double> xCoordTmp;
	thrust::device_vector<double> yCoordTmp;
	thrust::device_vector<double> zCoordTmp;

	SceCells(SceNodes* nodesInput);

	void growAndDivide(double dt);
	void grow(double dt);
	void processDivisionInfoAndAddNewCells();
};

#endif /* SCECELLS_H_ */
