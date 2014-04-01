#include <vector>

/**
 * This class solves efficiency problem while tracking boundary.
 */
class BoundaryTrack {
	std::vector<LineSegment> lineSegments;
	vector<CVector> nodesCloseToBdry;
	vector<vector<int> > generatedImg;
	void setInitPoint(CVector initPt);
	void setInitProfile(int img[][], int width, int height);
	// put all possible locations to hashset
	void buildHashsetForPossibleNodeLocations();
	void buildHashsetForPossibleCellCenterLocations();
	// pass in all SceNodes filter those nodes that are not close to bdry.
	void keepNodesCloseToBoundary(vector<CVector> &inputNodes);
	void buildImgGivenNodes();
	void generateLineSegUsingMarchingSquare();
public:
	void initialize(CVector initPt, int img[][], int width, int height);
	void update(vector<CVector> &inputCellCenters,
			vector<vector<CVector> > &inputNodes);
};

