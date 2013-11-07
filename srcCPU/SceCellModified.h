/*
 * SceCellModified.h
 *
 *  Created on: Oct 31, 2013
 *      Author: wsun2
 */

#ifndef SCECELLMODIFIED_H_
#define SCECELLMODIFIED_H_

#include "SceNode.h"
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <climits>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace BeakSce {

enum CellType {
	Boundary, NormalCell
};

struct SceCellModifiedException: public std::exception {
	std::string errorMessage;
	SceCellModifiedException(std::string errMsg) :
			errorMessage(errMsg) {
	}
	~SceCellModifiedException() throw () {
	}
	const char* what() const throw () {
		return errorMessage.c_str();
	}
};

/*
 * Auxiliary data structure for sorting purpose
 */
struct AuxStruct {
	int rank;
	double value;
};

/*
 * Auxiliary comparison operator
 */
struct AuxCompOp {
	inline bool operator()(const AuxStruct& struct1, const AuxStruct& struct2) {
		return (struct1.value < struct2.value);
	}
};

class SceCell {
	int nodeCount;
	int finalNodeCount;
	int initNodeCount;
	int growthCountDown;
	int cellRank;
	vector<SceNode> sceNodes;
	double cellAge;
	double initSize;
	double finalSize;
	double cellDivisionInterval;
	double cellNoGrowingInterval;
	CellType cellType;
public:
	SceCell();
	SceCell(double initLength, double finalLength, double cellLifeExpectency);
	void initializeWithLocations(vector<CVector> &nodeLocations, int cellRank);
	void clearVelocity();
	void addAllVelFromLinks();
	void addVelFromElongation(CVector elongateDirection, double targetLength);
	void move(double dt);

	void loadMesh(std::string meshName);
	bool isScheduledToGrow();
	void scheduleToGrow(double dt);
	void growAsScheduled();
	void runGrowthLogic(double dt);
	void addOneNode();
	bool tryAddOneNode(CVector startPoint, double distance,
			double minDistanceToOtherNode, double bondCutoffDist);
	void addNearbyPointsToLinkForAllPoints();
	bool allFurtherThanThreshold(CVector ptToBeCreated,
			double minDistanceToOtherNode);
	void insertPointToCell(CVector ptToBeCreated, double maxDistanceForBond);
	void divideIgnoreNewCell(CVector direction);
	void divideAndPushInfoToBuffer(CVector direction,
			vector<vector<CVector> > &cellDivisionBuffer);
	void runDivideLogicIgnoreNewCell(CVector direction);
	void runDivideLogic(CVector direction,
			vector<vector<CVector> > &cellDivisionBuffer);
	double getTargetLengthNow();
	SceNode* addressOfIthSceNode(int i);
	virtual void moveAndExecuteAllLogic(double dt, CVector direction,
			vector<vector<CVector> > &cellDivisionBuffer);
	void outputVtkFiles(std::string scriptNameBase, int rank);

	vector<CVector> getNodeLoctions();
	CVector getCellCenter() const;
	CVector getNodeLocClosestToCenter() const;
	int getNodeCount();
	void printMaxVel();

	virtual ~SceCell();

	const vector<SceNode>& getSceNodes() const {
		return sceNodes;
	}

	void setSceNodes(const vector<SceNode>& sceNodes) {
		this->sceNodes = sceNodes;
	}

	CellType getCellType() const {
		return cellType;
	}

	void setCellType(CellType cellType) {
		this->cellType = cellType;
	}
};

class SceBoundary: public SceCell {
public:
	SceBoundary(double initLength, double finalLength,
			double cellLifeExpectency) :
			SceCell(initLength, finalLength, cellLifeExpectency) {
		setCellType(Boundary);
	}
	/*
	 * Overrides this function from super class. do nothing.
	 */
	void moveAndExecuteAllLogic(double dt, CVector direction,
			vector<vector<CVector> > &cellDivisionBuffer) {
	}
};

} /* namespace BeakSce */
#endif /* SCECELLMODIFIED_H_ */
