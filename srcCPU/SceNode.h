/*
 * SceNode.h
 *
 *  Created on: Oct 31, 2013
 *      Author: wsun2
 */

#ifndef SCENODE_H_
#define SCENODE_H_

#include "GeoVector.h"
#include <vector>
#include "ConfigParser.h"
#include <sstream>

namespace BeakSce {

/**
 * Advantages of this data structure:
 * 1) avoid costly atomic operations
 * 2) easy transition to GPU thrust library data structures
 * Disadvantages of this data structure:
 * 1) using array of structure instead of structure of array which is not cache friendly.
 *    However array of structure would significantly increase understandability.
 * 2) less object oriented than Cell based approach
 */
class SceNode {
	int cellRank;
	int nodeRank;
	CVector nodeLoc;
	CVector nodeVel;
	int intraLinkCount;
	int interLinkCount;
	vector<SceNode*> intraLinkNodes;
	vector<SceNode*> interLinkNodes;

public:
	SceNode();
	bool isCloserThanDist(SceNode* otherNode, double distance) const;

	/*
	 * movement related functions
	 */
	void clearVelocity();
	void addVel(CVector &addVel);
	void addAllVelFromLinks();
	void move(double dt);

	/*
	 * geometry related functions
	 */
	void addNodeToIntraLinkArray(SceNode* otherNode);
	void addNodeToInterLinkArray(SceNode* otherNode);

	/*
	 * force related functions
	 */
	CVector intraLinkForce(SceNode* otherNode) const;
	CVector interLinkForce(SceNode* otherNode) const;

	/*
	 * helper functions
	 */
	int getNumOfIntraLinks() {
		return intraLinkCount;
	}
	int getNumOfInterLinks() {
		return interLinkCount;
	}

	virtual ~SceNode();

	int getCellRank() const {
		return cellRank;
	}

	void setCellRank(int cellRank) {
		this->cellRank = cellRank;
	}

	const CVector& getNodeLoc() const {
		return nodeLoc;
	}

	void setNodeLoc(const CVector& nodeLoc) {
		this->nodeLoc = nodeLoc;
	}

	int getNodeRank() const {
		return nodeRank;
	}

	void setNodeRank(int nodeRank) {
		this->nodeRank = nodeRank;
	}

	const CVector& getNodeVel() const {
		return nodeVel;
	}

	void setNodeVel(const CVector& nodeVel) {
		this->nodeVel = nodeVel;
	}

	const vector<SceNode*> getInterLinkNodes() const {
		vector<SceNode*> result = vector<SceNode*>(interLinkNodes.begin(),
				interLinkNodes.begin() + interLinkCount);
		return result;
	}

	void setInterLinkNodes(const vector<SceNode*>& interLinkNodes) {
		this->interLinkNodes = interLinkNodes;
	}

	const vector<SceNode*> getIntraLinkNodes() const {
		vector<SceNode*> result = vector<SceNode*>(intraLinkNodes.begin(),
				intraLinkNodes.begin() + intraLinkCount);
		return result;
	}

	void setIntraLinkNodes(const vector<SceNode*>& intraLinkNodes) {
		this->intraLinkNodes = intraLinkNodes;
	}

	int getInterLinkCount() const {
		return interLinkCount;
	}

	void setInterLinkCount(int interLinkCount) {
		this->interLinkCount = interLinkCount;
	}

	int getIntraLinkCount() const {
		return intraLinkCount;
	}

	void setIntraLinkCount(int intraLinkCount) {
		this->intraLinkCount = intraLinkCount;
	}
};

} /* namespace BeakSce */
#endif /* SCENODE_H_ */
