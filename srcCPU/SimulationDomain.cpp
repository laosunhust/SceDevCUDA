/*
 * SimulationDomain.cpp
 *
 *  Created on: Nov 3, 2013
 *      Author: Wenzhao
 */

#include "SimulationDomain.h"

namespace BeakSce {

SimulationDomain::SimulationDomain() {
	int maxCellInDomain =
			globalConfigVars.getConfigValue("MaxCellInDomain").toInt();
	sceCells.resize(maxCellInDomain);
	cellCount = 0;
}

void SimulationDomain::insertNewCell(vector<CVector> &nodeLocations,
		int cellRank) {
	static const double cellGrowTime = globalConfigVars.getConfigValue(
			"CellDivisionTimeInterval").toDouble();
	static const double cellInitLength = globalConfigVars.getConfigValue(
			"CellInitLength").toDouble();
	static const double cellFinalLength = globalConfigVars.getConfigValue(
			"CellFinalLength").toDouble();
	sceCells[cellRank] = SceCell(cellInitLength, cellFinalLength, cellGrowTime);
	sceCells[cellRank].initializeWithLocations(nodeLocations, cellRank);
	cellCount++;
}

void SimulationDomain::insertNewBoundary(vector<CVector> &boundaryNodes,
		int cellRank) {
	static const double cellGrowTime = globalConfigVars.getConfigValue(
			"CellDivisionTimeInterval").toDouble();
	static const double cellInitLength = globalConfigVars.getConfigValue(
			"CellInitLength").toDouble();
	static const double cellFinalLength = globalConfigVars.getConfigValue(
			"CellFinalLength").toDouble();
	sceCells[cellRank] = SceBoundary(cellInitLength, cellFinalLength,
			cellGrowTime);
	sceCells[cellRank].initializeWithLocations(boundaryNodes, cellRank);
	cellCount++;
}

void SimulationDomain::allCellsMoveAndRecordDivisionInfo(double dt,
		CVector direction) {
	for (int i = 0; i < cellCount; i++) {
		if (sceCells[i].getCellType() == NormalCell) {
			sceCells[i].moveAndExecuteAllLogic(dt, direction,
					cellDivisionBuffer);
		} else if (sceCells[i].getCellType() == Boundary) {
			//SceBoundary* bdry = dynamic_cast<SceBoundary*>(&sceCells[i]);
			//bdry->moveAndExecuteAllLogic(dt, direction, cellDivisionBuffer);
		}
	}
}

void SimulationDomain::processDivisionInfoAndAddNewCells() {
	int cellNumToAdd = cellDivisionBuffer.size();
	for (int i = 0; i < cellNumToAdd; i++) {
		insertNewCell(cellDivisionBuffer[i], cellCount);
	}
	cellDivisionBuffer.clear();
}

void SimulationDomain::inefficientlyBuildInterCellLinks() {
	static const double bondCutoffDist = globalConfigVars.getConfigValue(
			"BondCutoffDist").toDouble();
	double dist;
	//int totalAddedLinks = 0;
	for (int i = 0; i < cellCount; i++) {
		int nodeCountI = sceCells[i].getNodeCount();
		for (int j = 0; j < nodeCountI; j++) {
			CVector locationJ =
					sceCells[i].addressOfIthSceNode(j)->getNodeLoc();
			sceCells[i].addressOfIthSceNode(j)->setInterLinkCount(0);
			for (int k = 0; k < cellCount; k++) {
				if (k != i) {
					int nodeCountK = sceCells[k].getNodeCount();
					for (int l = 0; l < nodeCountK; l++) {
						CVector locationL =
								sceCells[k].addressOfIthSceNode(l)->getNodeLoc();
						dist = (locationJ - locationL).getModul();
						if (dist < bondCutoffDist) {
							sceCells[i].addressOfIthSceNode(j)->addNodeToInterLinkArray(
									sceCells[k].addressOfIthSceNode(l));
							//totalAddedLinks++;
						}
					}
				}
			}
		}
	}
	//cout << "in inefficient build intercell links , we added "
	//		<< totalAddedLinks << "Inter-cell links" << endl;
	//getchar();
}

void SimulationDomain::outputVtkFiles(std::string scriptNameBase, int rank) {
	int i, j, k;
	// using string stream is definitely not the best solution,
	// but I can't use c++ 11 features for backward compatibility
	stringstream ss;
	ss << std::setw(5) << std::setfill('0') << rank;
	string scriptNameRank = ss.str();
	std::string vtkFileName = scriptNameBase + scriptNameRank + ".vtk";
	cout << "start to create vtk file" << vtkFileName << endl;
	ofstream fs;
	fs.open(vtkFileName.c_str());

	int NNum = getTotalNodeCount();
	int LNum = 0;

	fs << "# vtk DataFile Version 3.0" << endl;
	fs << "Lines and points representing subcelluar element cells " << endl;
	fs << "ASCII" << endl;
	fs << endl;
	fs << "DATASET POLYDATA" << endl;
	fs << "POINTS " << NNum << " float" << endl;

	/*
	 vector<CVector> nodePositions = sceCells[0].getNodeLoctions();
	 vector<SceNode> sceNodes = sceCells[0].getSceNodes();
	 for (i = 0; i < NNum; i++) {
	 CVector nodePos = nodePositions[i];
	 LNum = LNum + sceNodes[i].getNumOfIntraLinks();
	 fs << nodePos.x << " " << nodePos.y << " " << nodePos.z << endl;
	 }
	 */

	for (i = 0; i < cellCount; i++) {
		vector<CVector> nodePositions = sceCells[i].getNodeLoctions();
		vector<SceNode> sceNodes = sceCells[i].getSceNodes();
		NNum = sceCells[i].getNodeCount();
		for (j = 0; j < NNum; j++) {
			CVector nodePos = nodePositions[j];
			LNum = LNum + sceNodes[j].getNumOfIntraLinks();
			LNum = LNum + sceNodes[j].getNumOfInterLinks();
			fs << nodePos.x << " " << nodePos.y << " " << nodePos.z << endl;
		}
	}

	vector<int> auxArray = generateAuxArray();
	LNum = LNum / 2;

	fs << endl;
	fs << "LINES " << LNum << " " << 3 * LNum << endl;
	/*
	 for (i = 0; i < NNum; i++) {
	 //cout << "enter node loop: " << endl;
	 vector<SceNode*> linkedNodes = sceNodes[i].getIntraLinkNodes();
	 int numOfLinkedNodes = linkedNodes.size();
	 for (j = 0; j < numOfLinkedNodes; j++) {
	 if (linkedNodes[j]->getNodeRank() < sceNodes[i].getNodeRank()) {
	 //cout << "enter if: " << endl;
	 fs << 2 << " " << linkedNodes[j]->getNodeRank() << " "
	 << sceNodes[i].getNodeRank() << endl;
	 }
	 }
	 }
	 */

	///int totalOutputInterLinks = 0;
	for (i = 0; i < cellCount; i++) {

		vector<CVector> nodePositions = sceCells[i].getNodeLoctions();
		vector<SceNode> sceNodes = sceCells[i].getSceNodes();
		NNum = sceCells[i].getNodeCount();
		for (j = 0; j < NNum; j++) {
			vector<SceNode*> linkedNodesIntra = sceNodes[j].getIntraLinkNodes();
			int numOfLinkedNodesIntra = linkedNodesIntra.size();
			for (k = 0; k < numOfLinkedNodesIntra; k++) {
				if (linkedNodesIntra[k]->getNodeRank()
						< sceNodes[j].getNodeRank()) {
					int globalRank1 = getNodeGlobalRank(
							linkedNodesIntra[k]->getCellRank(),
							linkedNodesIntra[k]->getNodeRank(), auxArray);
					int globalRank2 = getNodeGlobalRank(
							sceNodes[j].getCellRank(),
							sceNodes[j].getNodeRank(), auxArray);
					fs << 2 << " " << globalRank1 << " " << globalRank2 << endl;
				}
			}

			vector<SceNode*> linkedNodesInter = sceNodes[j].getInterLinkNodes();
			int numOfLinkedNodesInter = linkedNodesInter.size();
			for (k = 0; k < numOfLinkedNodesInter; k++) {

				if (linkedNodesInter[k]->getCellRank()
						< sceNodes[j].getCellRank()) {
					//cout << "output inter link to vtk file!" << endl;
					//getchar();
					int globalRank1 = getNodeGlobalRank(
							linkedNodesInter[k]->getCellRank(),
							linkedNodesInter[k]->getNodeRank(), auxArray);

					int globalRank2 = getNodeGlobalRank(
							sceNodes[j].getCellRank(),
							sceNodes[j].getNodeRank(), auxArray);
					//totalOutputInterLinks++;
					fs << 2 << " " << globalRank1 << " " << globalRank2 << endl;
				}
			}

		}
	}
	//cout << "in output Vtk File , we outputed " << totalOutputInterLinks
	//		<< "Inter-cell links" << endl;
	//getchar();
	//getchar();
	fs.flush();
	fs.close();
}

void SimulationDomain::outputVtkFilesWithColor(std::string scriptNameBase,
		int rank) {
	int i, j, k;
	// using string stream is definitely not the best solution,
	// but I can't use c++ 11 features for backward compatibility
	stringstream ss;
	ss << std::setw(5) << std::setfill('0') << rank;
	string scriptNameRank = ss.str();
	std::string vtkFileName = scriptNameBase + scriptNameRank + ".vtk";
	cout << "start to create vtk file" << vtkFileName << endl;
	ofstream fs;
	fs.open(vtkFileName.c_str());

	int totalNNum = getTotalNodeCount();
	int LNum = 0;
	int NNum;

	fs << "# vtk DataFile Version 3.0" << endl;
	fs << "Lines and points representing subcelluar element cells " << endl;
	fs << "ASCII" << endl;
	fs << endl;
	fs << "DATASET UNSTRUCTURED_GRID" << endl;
	fs << "POINTS " << totalNNum << " float" << endl;

	/*
	 vector<CVector> nodePositions = sceCells[0].getNodeLoctions();
	 vector<SceNode> sceNodes = sceCells[0].getSceNodes();
	 for (i = 0; i < NNum; i++) {
	 CVector nodePos = nodePositions[i];
	 LNum = LNum + sceNodes[i].getNumOfIntraLinks();
	 fs << nodePos.x << " " << nodePos.y << " " << nodePos.z << endl;
	 }
	 */

	for (i = 0; i < cellCount; i++) {
		vector<CVector> nodePositions = sceCells[i].getNodeLoctions();
		vector<SceNode> sceNodes = sceCells[i].getSceNodes();
		NNum = sceCells[i].getNodeCount();
		for (j = 0; j < NNum; j++) {
			CVector nodePos = nodePositions[j];
			LNum = LNum + sceNodes[j].getNumOfIntraLinks();
			LNum = LNum + sceNodes[j].getNumOfInterLinks();
			fs << nodePos.x << " " << nodePos.y << " " << nodePos.z << endl;
		}
	}

	vector<int> auxArray = generateAuxArray();
	LNum = LNum / 2;

	fs << endl;
	fs << "CELLS " << LNum << " " << 3 * LNum << endl;

	///int totalOutputInterLinks = 0;
	for (i = 0; i < cellCount; i++) {

		vector<CVector> nodePositions = sceCells[i].getNodeLoctions();
		vector<SceNode> sceNodes = sceCells[i].getSceNodes();
		NNum = sceCells[i].getNodeCount();
		for (j = 0; j < NNum; j++) {
			vector<SceNode*> linkedNodesIntra = sceNodes[j].getIntraLinkNodes();
			int numOfLinkedNodesIntra = linkedNodesIntra.size();
			for (k = 0; k < numOfLinkedNodesIntra; k++) {
				if (linkedNodesIntra[k]->getNodeRank()
						< sceNodes[j].getNodeRank()) {
					int globalRank1 = getNodeGlobalRank(
							linkedNodesIntra[k]->getCellRank(),
							linkedNodesIntra[k]->getNodeRank(), auxArray);
					int globalRank2 = getNodeGlobalRank(
							sceNodes[j].getCellRank(),
							sceNodes[j].getNodeRank(), auxArray);
					fs << 2 << " " << globalRank1 << " " << globalRank2 << endl;
				}
			}

			vector<SceNode*> linkedNodesInter = sceNodes[j].getInterLinkNodes();
			int numOfLinkedNodesInter = linkedNodesInter.size();
			for (k = 0; k < numOfLinkedNodesInter; k++) {

				if (linkedNodesInter[k]->getCellRank()
						< sceNodes[j].getCellRank()) {
					//cout << "output inter link to vtk file!" << endl;
					//getchar();
					int globalRank1 = getNodeGlobalRank(
							linkedNodesInter[k]->getCellRank(),
							linkedNodesInter[k]->getNodeRank(), auxArray);

					int globalRank2 = getNodeGlobalRank(
							sceNodes[j].getCellRank(),
							sceNodes[j].getNodeRank(), auxArray);
					//totalOutputInterLinks++;
					fs << 2 << " " << globalRank1 << " " << globalRank2 << endl;
				}
			}

		}
	}
	fs << "CELL_TYPES " << LNum << endl;
	for (i = 0; i < LNum; i++) {
		fs << "3" << endl;
	}
	fs << "POINT_DATA " << totalNNum << endl;
	fs << "SCALARS point_scalars float" << endl;
	fs << "LOOKUP_TABLE default" << endl;
	for (i = 0; i < cellCount; i++) {
		NNum = sceCells[i].getNodeCount();
		for (j = 0; j < NNum; j++) {
			fs << i << endl;
		}
	}

	//cout << "in output Vtk File , we outputed " << totalOutputInterLinks
	//		<< "Inter-cell links" << endl;
	//getchar();
	//getchar();
	fs.flush();
	fs.close();
}

int SimulationDomain::getNodeGlobalRank(int cellRank, int nodeRank,
		vector<int> &auxArray) {
	return auxArray[cellRank] + nodeRank;
}

/*
 * equivalent to an exclusive scan
 */
vector<int> SimulationDomain::generateAuxArray() {
	vector<int> result;
	result.resize(cellCount);
	if (cellCount == 0) {
		return result;
	} else {
		result[0] = 0;
		for (int i = 1; i < cellCount; i++) {
			result[i] = result[i - 1] + sceCells[i - 1].getNodeCount();
		}
		return result;
	}
}
int SimulationDomain::getTotalNodeCount() {
	int result = 0;
	for (int i = 0; i < cellCount; i++) {
		result = result + sceCells[i].getNodeCount();
	}
	return result;
}

SimulationDomain::~SimulationDomain() {
// TODO Auto-generated destructor stub
}

} /* namespace BeakSce */
