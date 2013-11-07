/*
 * SceCellModified.cpp
 *
 *  Created on: Oct 31, 2013
 *      Author: wsun2
 */

#include "SceCellModified.h"

namespace BeakSce {

SceCell::SceCell() {
	static const int maxNodesInCellCount = globalConfigVars.getConfigValue(
			"MaxNodePerCell").toInt();
	sceNodes.resize(maxNodesInCellCount);
	nodeCount = 0;
	cellAge = 0;
	initSize = -1;
	finalSize = -1;
	cellDivisionInterval = -1;
	finalNodeCount = -1;
	initNodeCount = -1;
	growthCountDown = -1;
	cellType = NormalCell;
}

SceCell::SceCell(double initLength, double finalLength,
		double cellLifeExpectency) {
	static const int maxNodesInCellCount = globalConfigVars.getConfigValue(
			"MaxNodePerCell").toInt();
	static const double NoGrowthInterval = globalConfigVars.getConfigValue(
			"CellNoGrowthInterval").toDouble();
	sceNodes.resize(maxNodesInCellCount);
	nodeCount = 0;
	cellAge = 0;
	initSize = initLength;
	finalSize = finalLength;
	cellDivisionInterval = cellLifeExpectency;
	cellNoGrowingInterval = NoGrowthInterval;
	finalNodeCount = -1;
	initNodeCount = -1;
	growthCountDown = -1;
	cellType = NormalCell;
}

void SceCell::initializeWithLocations(vector<CVector> &nodeLocations,
		int cellRank) {
	static const double finalToInitRatio = globalConfigVars.getConfigValue(
			"FinalToInitNodeCountRatio").toDouble();
	this->cellRank = cellRank;
	nodeCount = nodeLocations.size();
	initNodeCount = nodeCount;
	finalNodeCount = finalToInitRatio * initNodeCount;
	if (cellRank == 1) {
		//finalNodeCount = initNodeCount;
		//growthCountDown = -2;
	}
	//finalNodeCount = initNodeCount;
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].setCellRank(cellRank);
		sceNodes[i].setNodeRank(i);
		sceNodes[i].setNodeLoc(nodeLocations[i]);
	}
	addNearbyPointsToLinkForAllPoints();
}
void SceCell::clearVelocity() {
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].clearVelocity();
	}
}
void SceCell::addAllVelFromLinks() {
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].addAllVelFromLinks();
	}
}

void SceCell::addVelFromElongation(CVector elongateDirection,
		double targetLength) {
	static const double elongationCoefficient = globalConfigVars.getConfigValue(
			"ElongateCoefficient").toDouble();
	elongateDirection = elongateDirection.getUnitVector();
	double minDist = 0.0, maxDist = 0.0;
	CVector cellCenter = getCellCenter();
	CVector dirToCenter;
	double distInElongationDirection;
	for (int i = 0; i < nodeCount; i++) {
		dirToCenter = sceNodes[i].getNodeLoc() - cellCenter;
		distInElongationDirection = dirToCenter * elongateDirection;
		if (distInElongationDirection > maxDist) {
			maxDist = distInElongationDirection;
		}
		if (distInElongationDirection < minDist) {
			minDist = distInElongationDirection;
		}
	}
	double lengthNow = maxDist - minDist;
	double elongationPara = elongationCoefficient * (targetLength - lengthNow);
	for (int i = 0; i < nodeCount; i++) {
		dirToCenter = sceNodes[i].getNodeLoc() - cellCenter;
		distInElongationDirection = dirToCenter * elongateDirection;
		CVector elongationForce = distInElongationDirection * elongationPara
				* elongateDirection;
		sceNodes[i].addVel(elongationForce);
	}

}

void SceCell::move(double dt) {
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].move(dt);
	}
	cellAge = cellAge + dt;
}

void SceCell::divideIgnoreNewCell(CVector direction) {
	CVector cellCenter = getCellCenter();
	CVector dirToCenter;
	direction = direction.getUnitVector();
	double distInDirection;
	vector<AuxStruct> auxStructs;
	vector<CVector> nodeLocationsOfRemainingCell;
	vector<CVector> nodeLocationsOfNewCell;
	for (int i = 0; i < nodeCount; i++) {
		AuxStruct mystruct;
		mystruct.rank = i;
		dirToCenter = sceNodes[i].getNodeLoc() - cellCenter;
		distInDirection = dirToCenter * direction;
		mystruct.value = distInDirection;
		auxStructs.push_back(mystruct);
	}
	sort(auxStructs.begin(), auxStructs.end(), AuxCompOp());
	for (int i = 0; i < nodeCount / 2; i++) {
		nodeLocationsOfRemainingCell.push_back(
				sceNodes[auxStructs[i].rank].getNodeLoc());
	}
	for (int i = nodeCount / 2; i < nodeCount; i++) {
		nodeLocationsOfNewCell.push_back(
				sceNodes[auxStructs[i].rank].getNodeLoc());
	}
	nodeCount = nodeLocationsOfRemainingCell.size();
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].setCellRank(i);
		sceNodes[i].setCellRank(0);
		sceNodes[i].setNodeLoc(nodeLocationsOfRemainingCell[i]);
	}
	cellAge = 0;
	growthCountDown = -1;
}

void SceCell::divideAndPushInfoToBuffer(CVector direction,
		vector<vector<CVector> > &cellDivisionBuffer) {
	CVector cellCenter = getCellCenter();
	CVector dirToCenter;
	direction = direction.getUnitVector();
	double distInDirection;
	vector<AuxStruct> auxStructs;
	vector<CVector> nodeLocationsOfRemainingCell;
	vector<CVector> nodeLocationsOfNewCell;
	for (int i = 0; i < nodeCount; i++) {
		AuxStruct mystruct;
		mystruct.rank = i;
		dirToCenter = sceNodes[i].getNodeLoc() - cellCenter;
		distInDirection = dirToCenter * direction;
		mystruct.value = distInDirection;
		auxStructs.push_back(mystruct);
	}
	sort(auxStructs.begin(), auxStructs.end(), AuxCompOp());
	for (int i = 0; i < nodeCount / 2; i++) {
		nodeLocationsOfRemainingCell.push_back(
				sceNodes[auxStructs[i].rank].getNodeLoc());
	}
	for (int i = nodeCount / 2; i < nodeCount; i++) {
		nodeLocationsOfNewCell.push_back(
				sceNodes[auxStructs[i].rank].getNodeLoc());
	}
	nodeCount = nodeLocationsOfRemainingCell.size();
	//finalNodeCount = nodeCount;
	for (int i = 0; i < nodeCount; i++) {
		sceNodes[i].setCellRank(i);
		sceNodes[i].setCellRank(this->cellRank);
		sceNodes[i].setNodeLoc(nodeLocationsOfRemainingCell[i]);
	}
	cellAge = 0;
	//growthCountDown = -1;
	//growthCountDown = -2;
	cellDivisionBuffer.push_back(nodeLocationsOfNewCell);
}

void SceCell::runDivideLogicIgnoreNewCell(CVector direction) {
	if (nodeCount >= finalNodeCount && cellAge >= cellDivisionInterval) {
		divideIgnoreNewCell(direction);
	}
}
void SceCell::runDivideLogic(CVector direction,
		vector<vector<CVector> > &cellDivisionBuffer) {
	if (nodeCount >= finalNodeCount && cellAge >= cellDivisionInterval) {
		divideAndPushInfoToBuffer(direction, cellDivisionBuffer);
	}
}

void SceCell::loadMesh(std::string meshName) {
	fstream fs;
	int NNum, LNum, ENum;
	int i, temp1, temp2;
	fs.open(meshName.c_str(), ios::in);
	if (!fs.is_open()) {
		std::string errString =
				"Unable to load mesh in string input mode, meshname: "
						+ meshName
						+ " ,possible reason is the file is not located in the project folder \n";
		throw SceCellModifiedException(errString);
	}
	fs >> NNum >> LNum >> ENum;
	vector<CVector> nodeLocations;
	nodeLocations.resize(NNum);

//sceNodes.resize(NNum);

//cout << "NNum = " << NNum << "ENum = " << ENum << "LNum = " << LNum << endl;

	for (i = 0; i < NNum; i++) {
		fs >> nodeLocations[i].x >> nodeLocations[i].y >> nodeLocations[i].z;
		nodeLocations[i] = nodeLocations[i] / 20.0;
		//cout << "node location:" << nodeLocations[i] << endl;
		//sceNodes[i].setNodeLoc(nodeLocations[i]);
		//sceNodes[i].setNodeRank(i);
		//sceNodes[i].setCellRank(0);
	}
	initializeWithLocations(nodeLocations, 0);

	nodeCount = NNum;
	for (i = 0; i < LNum; i++) {

		fs >> temp1 >> temp2;
		temp1 = temp1 - 1;
		temp2 = temp2 - 1;
		//cout << "node1:" << temp1 << "node2: " << temp2 << endl;
		sceNodes[temp1].addNodeToIntraLinkArray(&sceNodes[temp2]);
		sceNodes[temp2].addNodeToIntraLinkArray(&sceNodes[temp1]);
	}

	/*
	 for (i = 0; i < ENum; i++) {
	 int nodeCount = 0;
	 int temp;
	 vector<int> tmpVector;
	 fs >> nodeCount;
	 for (int j = 0; j < nodeCount; j++) {
	 fs >> temp;
	 tmpVector.push_back(temp - 1);
	 }
	 //trianglularElements.push_back(tmpVector);
	 }
	 */
	fs.close();
}

bool SceCell::isScheduledToGrow() {
	if (growthCountDown == -1) {
		return false;
	} else {
		return true;
	}
}
void SceCell::scheduleToGrow(double dt) {
	double timeStartedGrowing = cellAge - cellNoGrowingInterval;
	if (timeStartedGrowing < 0) {
		growthCountDown = (-timeStartedGrowing) / dt;
		return;
	}
	double remainLife = cellDivisionInterval - cellAge;
	if (remainLife < 0) {
		remainLife = 0;
	}
	double remainNodesToAdd = finalNodeCount - nodeCount;
	if (remainNodesToAdd == 0) {
		// -2 means that growth has stopped
		growthCountDown = -2;
	} else if (remainNodesToAdd < 0) {
		throw SceCellModifiedException(
				"number of nodes cannot exceed final NodeCount!");
	} else {
		growthCountDown = (int) (remainLife / dt / remainNodesToAdd);
	}

}
void SceCell::growAsScheduled() {
	if (growthCountDown == 0) {
		addOneNode();
		// -1 means growth needs to be reinitialized
		growthCountDown = -1;
		//cout << "added one node" << endl;
	} else if (growthCountDown > 0) {
		growthCountDown--;
	}
}
void SceCell::runGrowthLogic(double dt) {
	if (!isScheduledToGrow()) {
		scheduleToGrow(dt);
	} else {
		growAsScheduled();
	}
}

void SceCell::addOneNode() {
	static const double distance = globalConfigVars.getConfigValue(
			"DistanceForAddingNode").toDouble();
	static const double minDistanceToOtherNode =
			globalConfigVars.getConfigValue("MinDistanceToOtherNode").toDouble();
	static const double bondCutoffDist = globalConfigVars.getConfigValue(
			"BondCutoffDist").toDouble();
	static const int maxTry = globalConfigVars.getConfigValue(
			"AddingNodeMaxTry").toInt();
	CVector startPoint = getNodeLocClosestToCenter();

	for (int i = 0; i < maxTry; i++) {
		if (tryAddOneNode(startPoint, distance, minDistanceToOtherNode,
				bondCutoffDist)) {
			break;
		}
	}
}

double SceCell::getTargetLengthNow() {
	return initSize + cellAge / cellDivisionInterval * (finalSize - initSize);
}

bool SceCell::tryAddOneNode(CVector startPoint, double distance,
		double minDistanceToOtherNode, double bondCutoffDist) {
// static value of PI
	static const double piValue = acos(-1.0);
	static const double randMax = (double) (RAND_MAX);
// random number from system library
	double randNum = rand() / randMax;
	//cout << "rand number is " << randNum << endl;

// get a random radius in [0.0 , 2*PI)
	double randRad = randNum * 2.0 * piValue;
// get a random radius in [-PI, PI)
	randRad = randRad - piValue;
// X coordinate
	double xCor = distance * cos(randRad);
// Y coordinate
	double yCor = distance * cos(randRad);
// create new point
	CVector ptToBeCreated = startPoint + CVector(xCor, yCor, 0.0);
// check if all the distance from this new point to all existing point
	if (allFurtherThanThreshold(ptToBeCreated, minDistanceToOtherNode)) {
		//cout << "add succeeded at location" << ptToBeCreated << endl;
//		/getchar();
		insertPointToCell(ptToBeCreated, bondCutoffDist);
		return true;
	} else {
		//cout << "add failed at location" << ptToBeCreated << endl;
		return false;
	}
}

bool SceCell::allFurtherThanThreshold(CVector ptToBeCreated,
		double minDistanceToOtherNode) {
	double dist;
	for (int i = 0; i < nodeCount; i++) {
		dist = Modul(sceNodes[i].getNodeLoc() - ptToBeCreated);
		if (dist < minDistanceToOtherNode) {
			return false;
		}
	}
	return true;
}

/*
 * this is wrong because the poointer address is changing
 */
void SceCell::insertPointToCell(CVector ptToBeCreated,
		double maxDistanceForBond) {
	static const int maxNodeCount = globalConfigVars.getConfigValue(
			"MaxNodePerCell").toInt();
	int currentNNum = nodeCount;
	SceNode nodeToBeInserted;
	nodeToBeInserted.setNodeLoc(ptToBeCreated);
	nodeToBeInserted.setNodeRank(currentNNum);
	nodeToBeInserted.setCellRank(cellRank);
	sceNodes[currentNNum] = nodeToBeInserted;

	//cout<<"inserted node location:"<<ptToBeCreated<<endl;
	//getchar();
	/*
	 double dist;
	 for (int i = 0; i < currentNNum; i++) {
	 dist = Modul(sceNodes[i].getNodeLoc() - ptToBeCreated);
	 if (dist < maxDistanceForBond) {
	 sceNodes[i].addNodeToIntraLinkArray(&sceNodes[currentNNum]);
	 sceNodes[currentNNum].addNodeToIntraLinkArray(&sceNodes[i]);
	 }
	 }
	 */
	nodeCount++;
	if (nodeCount > maxNodeCount) {
		throw SceCellModifiedException(
				"current node count of cell exceeds maximum node count allowed!");
	}
}

void SceCell::addNearbyPointsToLinkForAllPoints() {
	static const double bondCutoffDist = globalConfigVars.getConfigValue(
			"BondCutoffDist").toDouble();
	double dist;
	for (int i = 0; i < nodeCount; i++) {
		// because we need to reconstruct link, we clear link count first
		sceNodes[i].setIntraLinkCount(0);
		for (int j = 0; j < nodeCount; j++) {
			if (j != i) {
				dist = Modul(
						sceNodes[i].getNodeLoc() - sceNodes[j].getNodeLoc());
				if (dist < bondCutoffDist) {
					sceNodes[i].addNodeToIntraLinkArray(&sceNodes[j]);
				}
			}
		}
	}
}

void SceCell::outputVtkFiles(std::string scriptNameBase, int rank) {
	int i, j;
// using string stream is definitely not the best solution,
// but I can't use c++ 11 features for backward compatibility
	stringstream ss;
	ss << std::setw(5) << std::setfill('0') << rank;
	string scriptNameRank = ss.str();
	std::string vtkFileName = scriptNameBase + scriptNameRank + ".vtk";
	cout << "start to create vtk file" << vtkFileName << endl;
	ofstream fs;
	fs.open(vtkFileName.c_str());

	int NNum = nodeCount;
	int LNum = 0;

	fs << "# vtk DataFile Version 3.0" << endl;
	fs << "Lines and points representing subcelluar element cells " << endl;
	fs << "ASCII" << endl;
	fs << endl;
	fs << "DATASET POLYDATA" << endl;
	fs << "POINTS " << NNum << " float" << endl;

	for (i = 0; i < NNum; i++) {
		CVector nodePos = sceNodes[i].getNodeLoc();
		LNum = LNum + sceNodes[i].getNumOfIntraLinks();
		fs << nodePos.x << " " << nodePos.y << " " << nodePos.z << endl;
	}

	LNum = LNum / 2;

	fs << endl;
	fs << "LINES " << LNum << " " << 3 * LNum << endl;
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
//getchar();
	fs.flush();
	fs.close();
}

void SceCell::moveAndExecuteAllLogic(double dt, CVector direction,
		vector<vector<CVector> > &cellDivisionBuffer) {
	runGrowthLogic(dt);
	//TODO: this needs to be done in global level in the future, but leave it local as for now
	addNearbyPointsToLinkForAllPoints();
	clearVelocity();
	addAllVelFromLinks();
	addVelFromElongation(direction, getTargetLengthNow());
	move(dt);
	//runDivideLogicIgnoreNewCell(direction);
	runDivideLogic(direction, cellDivisionBuffer);
}

void SceCell::printMaxVel() {
	double maxVel = 0;
	CVector maxVelocity, tmpVelocity;
	for (int i = 0; i < nodeCount; i++) {
		tmpVelocity = sceNodes[i].getNodeVel();
		if (tmpVelocity.getModul() > maxVel) {
			maxVelocity = tmpVelocity;
			maxVel = tmpVelocity.getModul();
		}
	}
	cout << "max velocity in this timestep is " << maxVelocity << endl;
}

int SceCell::getNodeCount() {
	return nodeCount;
}

CVector SceCell::getCellCenter() const {
	CVector result = CVector(0, 0, 0);
	for (int i = 0; i < nodeCount; i++) {
		result = result + sceNodes[i].getNodeLoc();
	}
	result = result / nodeCount;
	return result;
}

CVector SceCell::getNodeLocClosestToCenter() const {
	CVector cellCenter = getCellCenter();
	double minDistance = (sceNodes[0].getNodeLoc() - cellCenter).getModul();
	double tmpDist;
	int minDistNodeRank = 0;
	for (int i = 1; i < nodeCount; i++) {
		tmpDist = (sceNodes[i].getNodeLoc() - cellCenter).getModul();
		if (tmpDist < minDistance) {
			minDistNodeRank = i;
			minDistance = tmpDist;
		}
	}
	return sceNodes[minDistNodeRank].getNodeLoc();
}

SceNode* SceCell::addressOfIthSceNode(int i) {
	return &sceNodes[i];
}

vector<CVector> SceCell::getNodeLoctions() {
	vector<CVector> result;
	for (int i = 0; i < nodeCount; i++) {
		result.push_back(sceNodes[i].getNodeLoc());
	}
	return result;
}

SceCell::~SceCell() {
// TODO Auto-generated destructor stub
}

} /* namespace BeakSce */
