#ifndef SimulationDomainGPU_H_
#define SimulationDomainGPU_H_

#include "SceNodes.h"
#include "SceCells.h"
#include "GrowthDistriMap.h"
#include <sstream>
#include <iomanip>
#include <fstream>
#include <string>

/**
 * This class is responsible for domain-wise highest level logic, e.g. output animation.
 *
 */
class SimulationDomainGPU {
public:
	SceNodes nodes;
	SceCells cells;
	GrowthDistriMap growthMap;

	double intraLinkDisplayRange;

	double minX;
	double maxX;
	double minY;
	double maxY;
	double gridSpacing;

	uint growthGridXDim;
	uint growthGridYDim;
	double growthGridSpacing;
	double growthGridLowerLeftPtX;
	double growthGridLowerLeftPtY;
	double growthMorCenterXCoord;
	double growthMorCenterYCoord;
	double growthMorHighConcen;
	double growthMorLowConcen;
	double growthMorDiffSlope;

	double compuDist(double x1, double y1, double z1, double x2, double y2,
			double z2) {
		return sqrt(
				(x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
						+ (z1 - z2) * (z1 - z2));
	}

	SimulationDomainGPU();
	void initializeCells(std::vector<double> initCellNodePosX,
			std::vector<double> initCellNodePosY,
			std::vector<double> centerPosX, std::vector<double> centerPosY);
	void runAllLogic(double dt);
	void outputVtkFilesWithColor(std::string scriptNameBase, int rank);
};

#endif
