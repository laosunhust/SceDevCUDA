/*
 * CellInitHelper.cpp
 *
 *  Created on: Sep 22, 2013
 *      Author: wsun2
 */

#include "CellInitHelper.h"

CellInitHelper::CellInitHelper() {
	linEqn1 = StraightLineEquationNoneVertical(CVector(0, 20, 0),
			CVector(10, 40, 0));
	linEqn2 = StraightLineEquationNoneVertical(CVector(0, 10, 0),
			CVector(20, 20, 0));
	linEqn3 = StraightLineEquationNoneVertical(CVector(0, 30, 0),
			CVector(20, 20, 0));
	linEqn4 = StraightLineEquationNoneVertical(CVector(15, 40, 0),
			CVector(20, 20, 0));
}

// Initialize 15 points with coordinate catched directly from snapshot
void CellInitHelper::initPrecisionBoundaryPoints() {

	CVector P1(242, 26, 0);
	CVector P2(232, 26, 0);
	CVector P3(232, 168, 0);
	CVector P4(142, 259, 0);
	CVector P5(142, 307, 0);
	CVector P6(239, 246, 0);
	CVector P7(266, 246, 0);
	CVector P8(303, 278, 0);
	CVector P9(324, 277, 0);
	CVector P10(337, 257, 0);
	CVector P11(346, 250, 0);
	CVector P12(276, 56, 0);
	//CVector P13(292, 264, 0);
	CVector P13(305, 275, 0);
	CVector P14(271, 207, 0);
	CVector P15(249, 97, 0);
	initPoints.clear();
	initPoints.push_back(P1);
	initPoints.push_back(P2);
	initPoints.push_back(P3);
	initPoints.push_back(P4);
	initPoints.push_back(P5);
	initPoints.push_back(P6);
	initPoints.push_back(P7);
	initPoints.push_back(P8);
	initPoints.push_back(P9);
	initPoints.push_back(P10);
	initPoints.push_back(P11);
	initPoints.push_back(P12);
	initPoints.push_back(P13);
	initPoints.push_back(P14);
	initPoints.push_back(P15);
}

void CellInitHelper::transformBoundaryPoints() {
	double ratio = 10.0;
	double xZeroPoint = 142;
	double yBound = 350;
	for (unsigned int i = 0; i < initPoints.size(); i++) {
		initPoints[i].x = (initPoints[i].x - xZeroPoint) / ratio;
		initPoints[i].y = (yBound - initPoints[i].y) / ratio;
	}
}

CVector CellInitHelper::getPointGivenAngle(double currentAngle, double r,
		CVector centerPos) {
	double xPos = centerPos.x + r * cos(currentAngle);
	double yPos = centerPos.y + r * sin(currentAngle);
	return CVector(xPos, yPos, 0);
}

/*
 * bdryNodes is the result vector. Must be exactly divisable by number of nodes per cell.
 * interval controls distance between nodes.
 * beginPoint is not a real boundary and thus not in the boundary nodes list
 * but is also important in initialize our array.
 */
void CellInitHelper::generateBoundaryCellNodesArray(vector<CVector> &bdryNodes,
		double distance) {
	CVector Point1 = CVector(0.0, (350 - 307) / 10.0, 0.0);
	CVector Point2 = CVector(0.0, (350 - 259) / 10.0, 0.0);
	CVector Point3 = CVector((232 - 142) / 10.0, (350 - 168) / 10.0, 0.0);
	CVector Point4 = CVector((232 - 142) / 10.0, (350 - 26) / 10.0, 0.0);
	CVector Point5 = CVector((342 - 142) / 10.0, (350 - 26) / 10.0, 0.0);
	CVector Point6 = CVector((249 - 142) / 10.0, (350 - 97) / 10.0, 0);
	double tmpDiff1 = fabs(Point3.x - Point6.x);
	double tmpDiff2 = fabs(Point3.y - Point6.y);
	double radius = (tmpDiff1 * tmpDiff1 + tmpDiff2 * tmpDiff2) / 2.0
			/ tmpDiff1;
	CVector arcCenter = CVector(-4.97647, 25.3, 0);

	//fstream fs;

	bdryNodes.clear();
	double delta = 1.0e-6;
	CVector dirVector1 = Point2 - Point1;
	CVector dirVector2 = Point3 - Point2;
	CVector dirVector4 = Point5 - Point4;
	double totalDistance1 = Modul(dirVector1);
	double totalDistance2 = Modul(dirVector2);
	assert(Point3.x == Point4.x);
	double arcSpan = fabs(Point3.y - Point4.y);
	double arcSpanAngle = asin(arcSpan / 2.0 / radius) * 2;
	double totalDistance3 = radius * arcSpanAngle;
	double angleIncrement = arcSpanAngle / (totalDistance3 / distance);

	CVector unitVector1 = dirVector1.getUnitVector();
	CVector unitVector2 = dirVector2.getUnitVector();
	CVector unitVector4 = dirVector4.getUnitVector();
	CVector unitIncrease1 = unitVector1 * distance;
	CVector unitIncrease2 = unitVector2 * distance;
	CVector unitIncrease4 = unitVector4 * distance;
	CVector tempPoint;
	bool isOnLine2 = false;
	double distanceFromStart = 0.0;
	int nodeNumCounter = 0;
	int numOfPointsOnLine1 = floor(totalDistance1 / distance) + 1;
	double startDistFromStartOnLine2 = distance
			- (totalDistance1 / distance - (int) (totalDistance1 / distance));
	CVector startPointOnLine2 = Point2
			+ startDistFromStartOnLine2 * unitVector2;
	double actualDistance2 = Modul(Point3 - startPointOnLine2);
	int numOfPointsOnLine2 = floor(actualDistance2 / distance) + 1;
	//int totalNumOfNodes = numOfPointsOnLine1 + numOfPointsOnLine2;
	int totalNumOfNodes = 0;

	tempPoint = Point1;
	while (Modul(tempPoint - Point1) <= Modul(dirVector1)) {
		// fs << tempPoint.x << " " << tempPoint.y << " " << tempPoint.z << endl;
		bdryNodes.push_back(tempPoint);
		tempPoint = tempPoint + unitIncrease1;
		cout << "distance to Point1: " << Modul(tempPoint - Point1) << endl;
		totalNumOfNodes++;
	}
	tempPoint = tempPoint - unitIncrease1;
	double leftOver = Modul(tempPoint - Point2);
	cout << "left over = " << leftOver << endl;
	tempPoint = Point2 + (distance - leftOver) * unitIncrease2;
	while (Modul(tempPoint - Point2) <= Modul(dirVector2)) {
		//fs << tempPoint.x << " " << tempPoint.y << " " << tempPoint.z << endl;
		bdryNodes.push_back(tempPoint);
		tempPoint = tempPoint + unitIncrease2;
		cout << "distance to Point2: " << Modul(tempPoint - Point2) << endl;
		totalNumOfNodes++;
	}

	double theta = -atan(
			fabs(tempPoint.y - arcCenter.y) / fabs(tempPoint.x - arcCenter.x));
	double thetaAlternative = asin((tempPoint.y - arcCenter.y) / radius);
	cout << "starting theta = " << theta << "alternative theta = "
			<< thetaAlternative << endl;
	cout << "starting point = ";
	tempPoint.Print();
	cout << "arcSpanAngle = " << arcSpanAngle / 2.0 << endl;
	//theta = theta + angleIncrement;
	while (theta <= arcSpanAngle / 2.0) {
		tempPoint = getPointGivenAngle(theta, radius, arcCenter);
		theta = theta + angleIncrement;
		totalNumOfNodes++;
		//fs << tempPoint.x << " " << tempPoint.y << " " << tempPoint.z << endl;
		bdryNodes.push_back(tempPoint);
		cout << "in arc stage, target point: ";
		Point4.Print();
		cout << "current theta = " << theta << "and current position: ";
		tempPoint.Print();
	}
	double leftOverAngle = arcSpanAngle / 2.0 - (theta - angleIncrement);
	cout << "angle increment = " << angleIncrement << " and left over angle ="
			<< leftOverAngle << endl;
	double leftOverDistance = leftOverAngle * radius;
	double increaseDist = distance - leftOverDistance;
	CVector startPointOnLine4 = Point4 + increaseDist * unitVector4;
	tempPoint = startPointOnLine4;
	while (Modul(tempPoint - Point4) <= Modul(dirVector4)) {
		//fs << tempPoint.x << " " << tempPoint.y << " " << tempPoint.z << endl;
		bdryNodes.push_back(tempPoint);
		tempPoint = tempPoint + unitIncrease4;
		totalNumOfNodes++;
	}

}

// initialize all boundary lines
void CellInitHelper::initBoundaryLines(double interval) {
	StraightLineEquationNoneVertical *B1 = new StraightLineEquationNoneVertical(
			initPoints[0], initPoints[1]);
	B1->condiType = Down;
	Arc *B2 = new Arc(initPoints[1], initPoints[14], initPoints[2]);
	B2->condiType = Outside;
	StraightLineEquationNoneVertical *B3 = new StraightLineEquationNoneVertical(
			initPoints[2], initPoints[3]);
	B3->condiType = Right;
	StraightLineEquationVertical *B4 = new StraightLineEquationVertical(
			initPoints[3].x);
	B4->condiType = Right;
	StraightLineEquationNoneVertical *B5 = new StraightLineEquationNoneVertical(
			initPoints[4], initPoints[5]);
	B5->condiType = Up;
	StraightLineEquationNoneVertical *B6 = new StraightLineEquationNoneVertical(
			initPoints[5], initPoints[6]);
	B6->condiType = Up;
	StraightLineEquationNoneVertical *B7 = new StraightLineEquationNoneVertical(
			initPoints[6], initPoints[7]);
	B7->condiType = Up;
	StraightLineEquationNoneVertical *B8 = new StraightLineEquationNoneVertical(
			initPoints[7], initPoints[8]);
	B8->condiType = Up;
	StraightLineEquationNoneVertical *B9 = new StraightLineEquationNoneVertical(
			initPoints[8], initPoints[9]);
	B9->condiType = Up;
	StraightLineEquationNoneVertical *B10 =
			new StraightLineEquationNoneVertical(initPoints[9], initPoints[10]);
	B10->condiType = Up;
	StraightLineEquationNoneVertical *B11 =
			new StraightLineEquationNoneVertical(initPoints[10],
					initPoints[11]);
	B11->condiType = Left;
	StraightLineEquationNoneVertical *B12 =
			new StraightLineEquationNoneVertical(initPoints[11], initPoints[0]);
	B12->condiType = Left;

	StraightLineEquationNoneVertical *D1 = new StraightLineEquationNoneVertical(
			initPoints[2], initPoints[13]);
	D1->condiType = Left;
	StraightLineEquationNoneVertical *D2 = new StraightLineEquationNoneVertical(
			initPoints[13], initPoints[12]);
	D2->condiType = Left;

	boundaryLines.clear();
	internalBoundaryLines.clear();
	boundariesForCellCenter.clear();
	boundaryLines.push_back(B1);
	boundaryLines.push_back(B2);
	boundaryLines.push_back(B3);
	boundaryLines.push_back(B4);
	boundaryLines.push_back(B5);
	boundaryLines.push_back(B6);
	boundaryLines.push_back(B7);
	boundaryLines.push_back(B8);
	boundaryLines.push_back(B9);
	boundaryLines.push_back(B10);
	boundaryLines.push_back(B11);
	boundaryLines.push_back(B12);

	internalBoundaryLines.push_back(D1);
	internalBoundaryLines.push_back(D2);

	StraightLineEquationNoneVertical IB1 = B1->getDownOf(interval);
	Arc IB2 = B2->getOutside(interval);
	StraightLineEquationNoneVertical IB3 = B3->getDownOf(interval);
	StraightLineEquationVertical IB4 = B4->getRightOf(interval);
	StraightLineEquationNoneVertical IB5 = B5->getUpOf(interval);
	StraightLineEquationNoneVertical IB6 = B6->getUpOf(interval);
	StraightLineEquationNoneVertical IB7 = B7->getUpOf(interval);
	StraightLineEquationNoneVertical IB8 = B8->getUpOf(interval);
	StraightLineEquationNoneVertical IB9 = B9->getUpOf(interval);
	StraightLineEquationNoneVertical IB10 = B10->getUpOf(interval);
	StraightLineEquationNoneVertical IB11 = B11->getDownOf(interval);
	StraightLineEquationNoneVertical IB12 = B12->getDownOf(interval);

	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB1));
	boundariesForCellCenter.push_back(new Arc(IB2));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB3));
	boundariesForCellCenter.push_back(new StraightLineEquationVertical(IB4));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB5));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB6));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB7));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB8));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB9));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB10));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB11));
	boundariesForCellCenter.push_back(
			new StraightLineEquationNoneVertical(IB12));

}

vector<CVector> CellInitHelper::getCellCentersInside(double interval) {
	// start from top left most point
	double endX = initPoints[10].x;
	double startingY = initPoints[0].y - interval / 2.0;
	vector<CVector> result;
	double currentY = startingY;
	double currentX;
	while (currentY > 0) {
		try {
			int count = 0;
			currentX = getStartingXGivenY(currentY);
			while (currentX < endX) {
				CVector temp(currentX, currentY, 0);
				cout << "Printing temp center location:";
				temp.Print();
				bool isInside = false;
				try {
					isInside = isCellCenterInsidePreciseRegion(temp);
				} catch (CellInitHelperException &e) {
					cout
							<< " got exception from get starting X, error message:!"
							<< e.what() << endl;
				}
				if (isInside) {
					result.push_back(temp);
				}
				//if (count == 0) {
				//	int jj;
				//	cin >> jj;
				//}
				count++;
				currentX = currentX + interval;
			}
			currentY = currentY - interval;
		} catch (CellInitHelperException &e) {
			cout << " got exception from get starting X, error message:!"
					<< e.what() << endl;
			break;
		}
	}
	return result;
}

double CellInitHelper::getStartingXGivenY(double yPos) {
	if (yPos <= initPoints[1].y && yPos > initPoints[2].y) {
		Arc* boundaryArc = static_cast<Arc *>(boundariesForCellCenter[1]);
		double centerX = boundaryArc->centerPos.x;
		double centerY = boundaryArc->centerPos.y;
		double radius = boundaryArc->r;
		double xRes = sqrt(
				radius * radius - (yPos - centerY) * (yPos - centerY))
				+ centerX;
		//CVector result(xRes, yPos, 0);
		//cout << "y lower bound for arc is " << initPoints[2].y
		//		<< "y upper bound for arc is " << initPoints[1].y << endl;
		//cout << "r = " << radius;
		//cout << " center of arc:";
		//boundaryArc->centerPos.Print();
		//cout << "intput y = " << yPos << " output x =" << xRes << endl;
		//int jj;
		//cin >> jj;
		return xRes;
	} else if (yPos <= initPoints[2].y && yPos > initPoints[3].y) {
		StraightLineEquationNoneVertical* boundaryLine =
				static_cast<StraightLineEquationNoneVertical *>(boundariesForCellCenter[2]);
		double xRes = (yPos - boundaryLine->b) / boundaryLine->k;
		//cout << "y lower bound for straight line is " << initPoints[3].y
		//		<< "y upper bound for straight line is " << initPoints[2].y
		//		<< endl;
		//cout << "k = " << boundaryLine->k << " b = " << boundaryLine->b << endl;
		//cout << "intput y = " << yPos << " output x =" << xRes << endl;
		//int jj;
		//cin >> jj;
		//CVector result(xRes, yPos, 0);
		return xRes;
	} else if (yPos <= initPoints[3].y && yPos > initPoints[4].y) {
		StraightLineEquationVertical* boundaryVerticalLine =
				static_cast<StraightLineEquationVertical *>(boundariesForCellCenter[3]);
		double xRes = boundaryVerticalLine->xPos;
		//cout << "y lower bound for vertical line is " << initPoints[4].y
		//		<< "y upper bound for vertical line is " << initPoints[3].y
		//		<< endl;
		//cout << "x value of this vertical line = " << boundaryVerticalLine->xPos
		//		<< endl;
		//cout << "intput y = " << yPos << " output x =" << xRes << endl;
		//int jj;
		//cin >> jj;
		//CVector result(xRes, yPos, 0);
		return xRes;
	} else {
		throw CellInitHelperException(
				"Unexpected error while getting starting position");
	}
}

bool CellInitHelper::isCellCenterInsidePreciseRegion(CVector position) {
	//cout
	//		<< "check the following Point to see if it is inside Simulation Region:";
	//position.Print();
	for (unsigned int i = 0; i < boundariesForCellCenter.size(); i++) {
		//boundariesForCellCenter[i]->printWhoAmI();
		if (!boundariesForCellCenter[i]->isFitCondition(position)) {
			//cout << "The following point does not fit " << i
			//		<< " th boundary line condition";
			//position.Print();
			return false;
		} else {
			//cout << "passed " << i << " th fit condition" << endl;
		}
	}
	return true;
}

bool CellInitHelper::isMXType(CVector position) {
	//cout << "debug: checking position";
	//position.Print();
	if (position.y >= initPoints[2].y) {
		return false;
	}
	if (position.x >= initPoints[12].x) {
		return false;
	}
	for (unsigned int i = 0; i < internalBoundaryLines.size(); i++) {
		if (!internalBoundaryLines[i]->isFitCondition(position)) {
			return false;
		}
	}
	return true;
}

bool CellInitHelper::isInsideFNMRegion(CVector position) {
	bool result = linEqn1.isRightOfLine(position) && (position.y < 40)
			&& linEqn3.isUpOfLine(position) && linEqn4.isLeftOfLine(position);
	return result;
}

bool CellInitHelper::isInsideMXRegion(CVector position) {
	bool result = linEqn1.isRightOfLine(position) && (position.x > 0)
			&& linEqn3.isDownOfLine(position) && linEqn2.isUpOfLine(position);
	return result;
}

double CellInitHelper::getMinDistanceFromFNMBorder(CVector position) {
	double result = linEqn1.getDistance(position);
	double tmp = fabs(40 - position.y);
	if (tmp < result) {
		result = tmp;
	}
// distance of line equation 3 is disabled.
// because the point has to be placed at either region MX or region FNM
//tmp = linEqn3.getDistance(position);
//if (tmp < result) {
//	result = tmp;
//}
	tmp = linEqn4.getDistance(position);
	if (tmp < result) {
		result = tmp;
	}
	return result;
}

double CellInitHelper::getMinDistanceFromMXBorder(CVector position) {
	double result = linEqn1.getDistance(position);
	double tmp = fabs(position.x);
	if (tmp < result) {
		result = tmp;
	}
// distance of line equation 3 is disabled.
// because the point has to be placed at either region MX or region FNM
//tmp = linEqn3.getDistance(position);
//if (tmp < result) {
//	result = tmp;
//}
	tmp = linEqn2.getDistance(position);
	if (tmp < result) {
		result = tmp;
	}
	return result;
}

vector<CellPlacementInfo> CellInitHelper::obtainPreciseCellInfoArray(
		double interval, double deformRatio) {

	vector<CellPlacementInfo> cellPlacementInfoArray;
	cout << "BEGIN: " << cellPlacementInfoArray.size() << endl;
	initPrecisionBoundaryPoints();
	transformBoundaryPoints();
	initBoundaryLines(interval / 1.8);
	CVector cellDeform = CVector(deformRatio, deformRatio, deformRatio);
	vector<CVector> insideCellCenters = getCellCentersInside(interval);
	cout << "INSIDE CELLS: " << insideCellCenters.size() << endl;
	for (unsigned int i = 0; i < insideCellCenters.size(); i++) {
		CVector centerPos = insideCellCenters[i];
		if (isMXType(centerPos)) {
			CellPlacementInfo cellInfo;
			cellInfo.centerLocation = centerPos;
			cellInfo.cellType = MX;
			cellPlacementInfoArray.push_back(cellInfo);
		} else {
			CellPlacementInfo cellInfo;
			cellInfo.centerLocation = centerPos;
			cellInfo.cellType = FNM;
			cellPlacementInfoArray.push_back(cellInfo);
		}
	}
	cout << "cellInfoArraySize: " << cellPlacementInfoArray.size() << endl;
	//int jj;
	//cin >> jj;
	return cellPlacementInfoArray;
}

/**
 * Initialize data for future usage.
 * translate the intermediate data types to final input types that are acceptable by
 * initialCellsOfThreeTypes().
 * last input of initCellNodePoss is the initial position of cell nodes, which could read
 * from mesh file. must be preprocessed to have the cell center of (0,0)
 */
void CellInitHelper::initInputsFromCellInfoArray(vector<CellType> &cellTypes,
		vector<uint> &numOfInitNodesOfCells,
		vector<double> &initBdryCellNodePosX,
		vector<double> &initBdryCellNodePosY,
		vector<double> &initFNMCellNodePosX,
		vector<double> &initFNMCellNodePosY, vector<double> &initMXCellNodePosX,
		vector<double> &initMXCellNodePosY, vector<CVector> &bdryNodes,
		vector<CVector> &FNMCellCenters, vector<CVector> &MXCellCenters,
		vector<CVector> &initCellNodePoss) {

	cellTypes.clear();
	numOfInitNodesOfCells.clear();
	initBdryCellNodePosX.clear();
	initBdryCellNodePosY.clear();
	initFNMCellNodePosX.clear();
	initFNMCellNodePosY.clear();
	initMXCellNodePosX.clear();
	initMXCellNodePosY.clear();

	uint FnmCellCount = FNMCellCenters.size();
	uint MxCellCount = MXCellCenters.size();

	uint maxNodePerCell =
			globalConfigVars.getConfigValue("MaxNodePerCell").toInt();
	uint initNodePerCell = initCellNodePoss.size();
	uint requiredSpaceForBdry = ceil(
			(double) bdryNodes.size() / maxNodePerCell);
	uint requiredTotalSpaceBdry = maxNodePerCell * requiredSpaceForBdry;
	initBdryCellNodePosX.resize(requiredTotalSpaceBdry, 0.0);
	initBdryCellNodePosY.resize(requiredTotalSpaceBdry, 0.0);
	initFNMCellNodePosX.resize(maxNodePerCell * FnmCellCount, 0.0);
	initFNMCellNodePosY.resize(maxNodePerCell * FnmCellCount, 0.0);
	initMXCellNodePosX.resize(maxNodePerCell * MxCellCount, 0.0);
	initMXCellNodePosY.resize(maxNodePerCell * MxCellCount, 0.0);

	for (unsigned int i = 0; i < requiredSpaceForBdry - 1; i++) {
		cellTypes.push_back(Boundary);
		numOfInitNodesOfCells.push_back(maxNodePerCell);
	}
	cout << "Required space for bdry:" << requiredSpaceForBdry << endl;
	cellTypes.push_back(Boundary);
	numOfInitNodesOfCells.push_back(
			bdryNodes.size() - (requiredSpaceForBdry - 1) * maxNodePerCell);
	cout << "after bdry calculation, size of cellType is now "
			<< cellTypes.size() << endl;

	for (uint i = 0; i < FnmCellCount; i++) {
		cellTypes.push_back(FNM);
		numOfInitNodesOfCells.push_back(initNodePerCell);
	}
	cout << "after fnm calculation, size of cellType is now "
			<< cellTypes.size() << endl;

	for (uint i = 0; i < MxCellCount; i++) {
		cellTypes.push_back(MX);
		numOfInitNodesOfCells.push_back(initNodePerCell);
	}

	cout << "after mx calculation, size of cellType is now " << cellTypes.size()
			<< endl;

	cout << "begin init bdry pos:" << endl;

	cout << "size of bdryNodes is " << bdryNodes.size() << endl;
	cout << "size of initBdryCellNodePos = " << initBdryCellNodePosX.size()
			<< endl;

	//int jj;
	//cin>>jj;

	for (uint i = 0; i < bdryNodes.size(); i++) {
		initBdryCellNodePosX[i] = bdryNodes[i].x;
		initBdryCellNodePosY[i] = bdryNodes[i].y;
		//cout << "(" << initBdryCellNodePosX[i] << "," << initBdryCellNodePosY[i]
		//		<< ")" << endl;
	}

	uint index;
	cout << "begin init fnm pos:" << endl;
	cout << "current size is" << initFNMCellNodePosX.size() << endl;
	cout << "try to resize to: " << maxNodePerCell * FnmCellCount << endl;

	cout << "size of fnm pos: " << initFNMCellNodePosX.size() << endl;

	cout.flush();

	for (uint i = 0; i < FnmCellCount; i++) {
		for (uint j = 0; j < initNodePerCell; j++) {
			index = i * maxNodePerCell + j;
			initFNMCellNodePosX[index] = FNMCellCenters[i].x
					+ initCellNodePoss[j].x;
			initFNMCellNodePosY[index] = FNMCellCenters[i].y
					+ initCellNodePoss[j].y;
			cout << "(" << initFNMCellNodePosX[index] << ","
					<< initFNMCellNodePosY[index] << ")" << endl;
		}
	}

	cout << "begin init mx pos:" << endl;

	for (uint i = 0; i < MxCellCount; i++) {
		for (uint j = 0; j < initNodePerCell; j++) {
			index = i * maxNodePerCell + j;
			initMXCellNodePosX[index] = MXCellCenters[i].x
					+ initCellNodePoss[j].x;
			initMXCellNodePosY[index] = MXCellCenters[i].y
					+ initCellNodePoss[j].y;
			cout << "(" << initMXCellNodePosX[index] << ","
					<< initMXCellNodePosY[index] << ")" << endl;
		}
	}

	cout << "finished init inputs" << endl;
}

/**
 * Initialize inputs for five different components.
 */
void CellInitHelper::initInputsV2(SimulationInitData &initData,
		RawDataInput &rawData) {

	// step1: clean those init data.
	initData.cellTypes.clear();
	initData.numOfInitActiveNodesOfCells.clear();
	initData.initBdryCellNodePosX.clear();
	initData.initBdryCellNodePosY.clear();
	initData.initProfileNodePosX.clear();
	initData.initProfileNodePosY.clear();
	initData.initECMNodePosX.clear();
	initData.initECMNodePosY.clear();
	initData.initFNMCellNodePosX.clear();
	initData.initFNMCellNodePosY.clear();
	initData.initMXCellNodePosX.clear();
	initData.initMXCellNodePosY.clear();

	uint FnmCellCount = rawData.FNMCellCenters.size();
	uint MxCellCount = rawData.MXCellCenters.size();
	uint ECMCount = rawData.ECMCenters.size();

	uint maxNodePerCell =
			globalConfigVars.getConfigValue("MaxNodePerCell").toInt();
	uint maxNodePerECM =
			globalConfigVars.getConfigValue("MaxNodePerECM").toInt();

	uint initTotalCellCount = rawData.initCellNodePoss.size();
	uint initTotalECMCount = rawData.ECMCenters.size();
	initData.initBdryCellNodePosX.resize(rawData.bdryNodes.size(), 0.0);
	initData.initBdryCellNodePosY.resize(rawData.bdryNodes.size(), 0.0);
	initData.initProfileNodePosX.resize(rawData.profileNodes.size());
	initData.initProfileNodePosY.resize(rawData.profileNodes.size());
	initData.initECMNodePosX.resize(maxNodePerECM * ECMCount);
	initData.initECMNodePosY.resize(maxNodePerECM * ECMCount);
	initData.initFNMCellNodePosX.resize(maxNodePerCell * FnmCellCount, 0.0);
	initData.initFNMCellNodePosY.resize(maxNodePerCell * FnmCellCount, 0.0);
	initData.initMXCellNodePosX.resize(maxNodePerCell * MxCellCount, 0.0);
	initData.initMXCellNodePosY.resize(maxNodePerCell * MxCellCount, 0.0);

	for (uint i = 0; i < FnmCellCount; i++) {
		initData.cellTypes.push_back(FNM);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	cout << "after fnm calculation, size of cellType is now "
			<< initData.cellTypes.size() << endl;

	for (uint i = 0; i < MxCellCount; i++) {
		initData.cellTypes.push_back(MX);
		initData.numOfInitActiveNodesOfCells.push_back(initTotalCellCount);
	}

	cout << "after mx calculation, size of cellType is now "
			<< initData.cellTypes.size() << endl;

	cout << "begin init bdry pos:" << endl;

	cout << "size of bdryNodes is " << rawData.bdryNodes.size() << endl;
	cout << "size of initBdryCellNodePos = "
			<< initData.initBdryCellNodePosX.size() << endl;

	//int jj;
	//cin>>jj;

	for (uint i = 0; i < rawData.bdryNodes.size(); i++) {
		initData.initBdryCellNodePosX[i] = rawData.bdryNodes[i].x;
		initData.initBdryCellNodePosY[i] = rawData.bdryNodes[i].y;
		//cout << "(" << initBdryCellNodePosX[i] << "," << initBdryCellNodePosY[i]
		//		<< ")" << endl;
	}

	for (uint i = 0; i < rawData.profileNodes.size(); i++) {
		initData.initProfileNodePosX[i] = rawData.profileNodes[i].x;
		initData.initProfileNodePosX[i] = rawData.profileNodes[i].y;
		//cout << "(" << initBdryCellNodePosX[i] << "," << initBdryCellNodePosY[i]
		//		<< ")" << endl;
	}

	uint index;
	cout << "begin init fnm pos:" << endl;
	cout << "current size is" << initData.initFNMCellNodePosX.size() << endl;
	cout << "try to resize to: " << maxNodePerCell * FnmCellCount << endl;

	cout << "size of fnm pos: " << initData.initFNMCellNodePosX.size() << endl;

	cout.flush();

	uint ECMInitNodeCount = rawData.initECMNodePoss.size();
	for (uint i = 0; i < ECMCount; i++) {
		vector<CVector> rotatedCoords = rotate2D(rawData.initECMNodePoss,
				rawData.ECMAngles[i]);
		for (uint j = 0; j < ECMInitNodeCount; j++) {
			index = i * maxNodePerECM + j;
			initData.initECMNodePosX[index] = rawData.ECMCenters[i].x
					+ rotatedCoords[j].x;
			initData.initECMNodePosY[index] = rawData.ECMCenters[i].y
					+ rotatedCoords[j].y;
		}
	}

	for (uint i = 0; i < FnmCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initFNMCellNodePosX[index] = rawData.FNMCellCenters[i].x
					+ rawData.initCellNodePoss[j].x;
			initData.initFNMCellNodePosY[index] = rawData.FNMCellCenters[i].y
					+ rawData.initCellNodePoss[j].y;
			//cout << "(" << initData.initFNMCellNodePosX[index] << ","
			//		<< initData.initFNMCellNodePosY[index] << ")" << endl;
		}
	}

	cout << "begin init mx pos:" << endl;

	for (uint i = 0; i < MxCellCount; i++) {
		for (uint j = 0; j < initTotalCellCount; j++) {
			index = i * maxNodePerCell + j;
			initData.initMXCellNodePosX[index] = rawData.MXCellCenters[i].x
					+ rawData.initCellNodePoss[j].x;
			initData.initMXCellNodePosY[index] = rawData.MXCellCenters[i].y
					+ rawData.initCellNodePoss[j].y;
			//cout << "(" << initData.initMXCellNodePosX[index] << ","
			//		<< initData.initMXCellNodePosY[index] << ")" << endl;
		}
	}

	cout << "finished init inputs" << endl;
}

//cout << "before debug:" << endl;

void CellInitHelper::generateThreeInputCellInfoArrays(
		vector<CVector> &bdryNodes, vector<CVector> &FNMCellCenters,
		vector<CVector> &MXCellCenters, double cellCenterInterval,
		double bdryNodeInterval) {
	bdryNodes.clear();
	FNMCellCenters.clear();
	MXCellCenters.clear();

	generateBoundaryCellNodesArray(bdryNodes, bdryNodeInterval);

	initPrecisionBoundaryPoints();
	transformBoundaryPoints();
	initBoundaryLines(cellCenterInterval / 1.8);
	vector<CVector> insideCellCenters = getCellCentersInside(
			cellCenterInterval);
	cout << "INSIDE CELLS: " << insideCellCenters.size() << endl;

	for (unsigned int i = 0; i < insideCellCenters.size(); i++) {
		CVector centerPos = insideCellCenters[i];
		if (isMXType(centerPos)) {
			MXCellCenters.push_back(centerPos);
		} else {
			FNMCellCenters.push_back(centerPos);
		}
	}
	cout << "Number of boundary nodes: " << bdryNodes.size()
			<< "Number of MX cells: " << MXCellCenters.size()
			<< " and number of FNM cells:" << FNMCellCenters.size() << endl;

}

vector<CVector> CellInitHelper::rotate2D(vector<CVector> &initECMNodePoss,
		double angle) {
	uint inputVectorSize = initECMNodePoss.size();
	CVector centerPosOfInitVector = CVector(0);
	for (uint i = 0; i < inputVectorSize; i++) {
		centerPosOfInitVector = centerPosOfInitVector + initECMNodePoss[i];
	}
	centerPosOfInitVector = centerPosOfInitVector / inputVectorSize;
	for (uint i = 0; i < inputVectorSize; i++) {
		initECMNodePoss[i] = initECMNodePoss[i] - centerPosOfInitVector;
	}
	vector<CVector> result;
	for (uint i = 0; i < inputVectorSize; i++) {
		CVector posNew;
		posNew.x = cos(angle) * initECMNodePoss[i].x
				- sin(angle) * initECMNodePoss[i].y;
		posNew.y = sin(angle) * initECMNodePoss[i].x
				+ cos(angle) * initECMNodePoss[i].y;
		result.push_back(posNew);
	}
	return result;
}

CellInitHelper::~CellInitHelper() {
	/*
	 vector<BoundaryLine *>::iterator it = boundaryLines.begin();
	 while (it != boundaryLines.end()) {
	 delete ((*it));
	 ++it;
	 }

	 it = boundariesForCellCenter.begin();
	 while (it != boundariesForCellCenter.end()) {
	 delete ((*it));
	 ++it;
	 }

	 it = internalBoundaryLines.begin();
	 while (it != internalBoundaryLines.end()) {
	 delete ((*it));
	 ++it;
	 }
	 */
}

