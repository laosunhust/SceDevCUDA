/*
 * SimulationDomain.h
 *
 *  Created on: Nov 3, 2013
 *      Author: Wenzhao
 */

#ifndef SIMULATIONDOMAIN_H_
#define SIMULATIONDOMAIN_H_

#include "SceCellModified.h"
#include <vector>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace BeakSce {

class SimulationDomain {
	vector<SceCell> sceCells;
	vector<vector<CVector> > cellDivisionBuffer;
	int cellCount;
public:
	SimulationDomain();
	void insertNewCell(vector<CVector> &nodeLocations, int cellRank);
	void insertNewBoundary(vector<CVector> &boundaryNodes, int cellRank);
	void allCellsMoveAndRecordDivisionInfo(double dt, CVector direction);
	void processDivisionInfoAndAddNewCells();
	void inefficientlyBuildInterCellLinks();
	void outputVtkFiles(std::string scriptNameBase, int i);
	void outputVtkFilesWithColor(std::string scriptNameBase, int i);

	int getNodeGlobalRank(int cellRank, int nodeRank, vector<int> &auxArray);
	vector<int> generateAuxArray();
	int getTotalNodeCount();
	virtual ~SimulationDomain();
};

} /* namespace BeakSce */

#endif /* SIMULATIONDOMAIN_H_ */
