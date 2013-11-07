/*
 * SceNode.cpp
 *
 *  Created on: Oct 31, 2013
 *      Author: wsun2
 */

#include "SceNode.h"

namespace BeakSce {

struct SceNodeException: public std::exception {
	std::string errorMessage;
	SceNodeException(std::string errMsg) :
			errorMessage(errMsg) {
	}
	~SceNodeException() throw () {
	}
	const char* what() const throw () {
		return errorMessage.c_str();
	}
};

SceNode::SceNode() {
	static const int maxIntraCellNodeCount = globalConfigVars.getConfigValue(
			"MaxIntraLinkPerNode").toInt();
	static const int maxInterCellNodeCount = globalConfigVars.getConfigValue(
			"MaxInterLinkPerNode").toInt();
	intraLinkNodes.resize(maxIntraCellNodeCount);
	interLinkNodes.resize(maxInterCellNodeCount);
	nodeRank = -1;
	cellRank = -1;
	intraLinkCount = 0;
	interLinkCount = 0;
}

bool SceNode::isCloserThanDist(SceNode* otherNode, double distance) const {
	CVector dirVec = otherNode->getNodeLoc() - this->nodeLoc;
	if (Modul(dirVec) < distance) {
		return true;
	} else {
		return false;
	}
}

void SceNode::clearVelocity() {
	nodeVel = CVector(0.0, 0.0, 0.0);
}
void SceNode::addVel(CVector &addVel) {
	nodeVel = nodeVel + addVel;
}
void SceNode::addAllVelFromLinks() {
	CVector tmpVel;
	/*
	 * this loop adds force from intraCell links
	 */
	//cout << "start calculate intra link nodes" << endl;
	for (int i = 0; i < intraLinkCount; i++) {
		tmpVel = this->intraLinkForce(intraLinkNodes[i]);
		this->addVel(tmpVel);
	}

	/*
	 * this loop adds force from interCell links
	 */
	//cout << "start calculate inter link nodes" << endl;
	for (int i = 0; i < interLinkCount; i++) {
		tmpVel = this->interLinkForce(interLinkNodes[i]);
		this->addVel(tmpVel);
	}
}
void SceNode::move(double dt) {
	nodeLoc = nodeLoc + nodeVel * dt;
}

void SceNode::addNodeToIntraLinkArray(SceNode* otherNode) {
	static const int maxIntraCount = globalConfigVars.getConfigValue(
			"MaxIntraLinkPerNode").toInt();
	intraLinkNodes[intraLinkCount] = otherNode;
	intraLinkCount++;
	if (intraLinkCount > maxIntraCount) {
		stringstream ss1, ss2;
		ss1 << intraLinkCount;
		ss2 << maxIntraCount;
		string linkCountString = ss1.str();
		string maxCountString = ss2.str();
		string errorMessage = "number of intra cell links is " + linkCountString
				+ ", which exceeds max allowed count: " + maxCountString;
		throw SceNodeException(errorMessage);
	}
}
void SceNode::addNodeToInterLinkArray(SceNode* otherNode) {
	static const int maxInterCount = globalConfigVars.getConfigValue(
			"MaxInterLinkPerNode").toInt();
	interLinkNodes[interLinkCount] = otherNode;
	interLinkCount++;
	if (interLinkCount > maxInterCount) {
		throw SceNodeException(
				"number of inter cell links exceeds maximum allowed");
	}
}

/*
 * all these force calculating related variables are stored here using static keyword
 * because of the need of saving memory in Data struture SceNode.
 */
CVector SceNode::intraLinkForce(SceNode* otherNode) const {
	static const double U0 =
			globalConfigVars.getConfigValue("IntraCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_U0_DivFactor").toDouble();
	static const double V0 =
			globalConfigVars.getConfigValue("IntraCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_V0_DivFactor").toDouble();
	static const double k1 =
			globalConfigVars.getConfigValue("IntraCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k1_DivFactor").toDouble();
	static const double k2 =
			globalConfigVars.getConfigValue("IntraCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("IntraCell_k2_DivFactor").toDouble();

	CVector dirVector = this->nodeLoc - otherNode->getNodeLoc();
	CVector unitDirVector = dirVector.getUnitVector();
	double linkLength = Modul(dirVector);
	double forceValue = -U0 / k1 * exp(-linkLength / k1)
			+ V0 / k2 * exp(-linkLength / k2);
	CVector result = unitDirVector * (-forceValue);
	return result;
}
/*
 * all these force calculating related variables are stored here using static keyword
 * because of the need of saving memory in Data struture SceNode.
 */
CVector SceNode::interLinkForce(SceNode* otherNode) const {

	static const double U0 =
			globalConfigVars.getConfigValue("InterCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_U0_DivFactor").toDouble();
	static const double V0 =
			globalConfigVars.getConfigValue("InterCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_V0_DivFactor").toDouble();
	static const double k1 =
			globalConfigVars.getConfigValue("InterCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k1_DivFactor").toDouble();
	static const double k2 =
			globalConfigVars.getConfigValue("InterCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k2_DivFactor").toDouble();

	CVector dirVector = this->nodeLoc - otherNode->getNodeLoc();
	CVector unitDirVector = dirVector.getUnitVector();
	double linkLength = Modul(dirVector);
	double forceValue = -U0 / k1 * exp(-linkLength / k1)
			+ V0 / k2 * exp(-linkLength / k2);
	if (forceValue > 0) {
		forceValue = 0;
	}
	CVector result = unitDirVector * (-forceValue);
	return result;
}

SceNode::~SceNode() {
// TODO Auto-generated destructor stub
}

} /* namespace BeakSce */
