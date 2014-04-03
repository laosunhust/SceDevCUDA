#include "BoundaryTrack.h"

void BoundaryTrack::setInitPoint(CVector initPt) {
}

void BoundaryTrack::setInitProfile(int img[][], int width, int height) {
}

void BoundaryTrack::buildHashsetForPossibleNodeLocations() {
}

void BoundaryTrack::buildHashsetForPossibleCellCenterLocations() {
}

void BoundaryTrack::keepNodesCloseToBoundary(vector<CVector>& inputNodes) {
}

void BoundaryTrack::buildImgGivenNodes() {
}

void BoundaryTrack::generateLineSegUsingMarchingSquare() {
}

void BoundaryTrack::initialize(CVector initPt, int img[][], int width,
		int height) {
}

/**
 * During update, we need to do the following:
 * (1) screen the input cell centers. keep the array offset of those centers
 * that might be close to the boundary. Store these offsets as an array.
 * (2) obtain a list of nodes that might be close to the boundary given the
 * array of offsets we obtained from step 1.
 * (3) only keep nodes that are actually near boundary that we found in the
 * previous update step
 * (4) generate an image using the nodes that we obtained. We can do so by
 * drawing each node as an circle and mark all pixels that are covers by circle.
 * (5) run marching square algorithm on the generated image and find new line segments
 * which represents outer profile
 * (6) generate hashset for possible locations of cell center and nodes.
 * These two hashsets will be used next step.
 */
void BoundaryTrack::update(vector<CVector>& inputCellCenters,
		vector<vector<CVector> >& inputNodes) {

}
