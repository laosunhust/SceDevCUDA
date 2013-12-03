#ifndef SCECELLS_H_
#define SCECELLS_H_

#include "SceNodes.h"
#include <thrust/tabulate.h>

struct DivideFunctor: public thrust::unary_function<uint, uint> {
	uint dividend;
	__host__ __device__ DivideFunctor(uint dividendInput) :
			dividend(dividendInput) {
	}
	__host__ __device__ uint operator()(const uint &num) {
		return num / dividend;
	}
};
struct ModuloFunctor: public thrust::unary_function<uint, uint> {
	uint dividend;
	__host__ __device__ ModuloFunctor(uint dividendInput) :
			dividend(dividendInput) {
	}
	__host__ __device__ uint operator()(const uint &num) {
		return num % dividend;
	}
};

struct isTrue {
	__host__ __device__
	bool operator()(bool b) {
		if (b == true) {
			return true;
		} else {
			return false;
		}
	}
};

struct CVec3Add: public thrust::binary_function<CVec3, CVec3, CVec3> {
	__host__ __device__ CVec3 operator()(const CVec3 &vec1, const CVec3 &vec2) {
		return thrust::make_tuple(
				thrust::get < 0 > (vec1) + thrust::get < 0 > (vec2),
				thrust::get < 1 > (vec1) + thrust::get < 1 > (vec2),
				thrust::get < 2 > (vec1) + thrust::get < 2 > (vec2));
	}
};
struct CVec3Divide: public thrust::binary_function<CVec3, double, CVec3> {
	__host__ __device__ CVec3 operator()(const CVec3 &vec1,
			const double &divisor) {
		return thrust::make_tuple(thrust::get < 0 > (vec1) / divisor,
				thrust::get < 1 > (vec1) / divisor,
				thrust::get < 2 > (vec1) / divisor);
	}
};

struct LoadGridDataToNode: public thrust::unary_function<CVec2, CVec3> {
	uint _gridDimensionX;
	uint _gridDimensionY;
	double _gridSpacing;
	double* _gridMagValue;
	double* _gridDirXCompValue;
	double* _gridDirYCompValue;
	__host__ __device__ LoadGridDataToNode(uint gridDimensionX,
			uint gridDimensionY, double gridSpacing, double* gridMagValue,
			double* gridDirXCompValue, double* gridDirYCompValue) :
			_gridDimensionX(gridDimensionX), _gridDimensionY(gridDimensionY), _gridSpacing(
					gridSpacing), _gridMagValue(gridMagValue), _gridDirXCompValue(
					gridDirXCompValue), _gridDirYCompValue(gridDirYCompValue) {
	}
	__host__ __device__ CVec3 operator()(const CVec2 &d2) const {
		double xCoord = thrust::get < 0 > (d2);
		double yCoord = thrust::get < 1 > (d2);
		uint gridLoc = (uint) (xCoord / _gridSpacing)
				+ (uint) (yCoord / _gridSpacing) * _gridDimensionX;
		double magRes = _gridMagValue[gridLoc];
		double xDirRes = _gridDirXCompValue[gridLoc];
		double yDirRes = _gridDirYCompValue[gridLoc];
		return thrust::make_tuple(magRes, xDirRes, yDirRes);
	}
};

struct SaxpyFunctor: public thrust::binary_function<double, double, double> {
	double _dt;
	__host__ __device__ SaxpyFunctor(double dt) :
			_dt(dt) {
	}
	__host__ __device__ double operator()(const double &x, const double &y) {
		return _dt * x + y;
	}
};

struct PtCondiOp: public thrust::unary_function<CVec2, BoolD> {
	double _threshold;
	__host__ __device__ PtCondiOp(double threshold) :
			_threshold(threshold) {
	}
	__host__       __device__ BoolD operator()(const CVec2 &d2) const {
		double progress = thrust::get < 0 > (d2);
		double lastCheckPoint = thrust::get < 1 > (d2);
		bool resBool = false;
		double resLastCheckPoint = lastCheckPoint;
		if (progress - lastCheckPoint >= _threshold) {
			resBool = true;
			resLastCheckPoint = resLastCheckPoint + _threshold;
		}
		return thrust::make_tuple(resBool, resLastCheckPoint);
	}
};

struct AddPtOp: thrust::unary_function<BoolUIDDUI, BoolUI> {
	uint _maxNodeOfOneCell;
	double _addNodeDistance;
	double _minDistanceToOtherNode;
	bool* _nodeIsActiveAddress;
	double* _nodeXPosAddress;
	double* _nodeYPosAddress;

	__host__ __device__ AddPtOp(uint maxNodeOfOneCell, double addNodeDistance,
			double minDistanceToOtherNode, bool* nodeIsActiveAddress,
			double* nodeXPosAddress, double* nodeYPosAddress) :
			_maxNodeOfOneCell(maxNodeOfOneCell), _addNodeDistance(
					addNodeDistance), _minDistanceToOtherNode(
					minDistanceToOtherNode), _nodeIsActiveAddress(
					nodeIsActiveAddress), _nodeXPosAddress(nodeXPosAddress), _nodeYPosAddress(
					nodeYPosAddress) {
	}
	__host__ __device__ BoolUI operator()(const BoolUIDDUI &biddi) {
		const double pI = acos(-1.0);
		bool isScheduledToGrow = thrust::get < 0 > (biddi);
		uint activeNodeCountOfThisCell = thrust::get < 1 > (biddi);
		double cellCenterXCoord = thrust::get < 2 > (biddi);
		double cellCenterYCoord = thrust::get < 3 > (biddi);
		uint cellRank = thrust::get < 4 > (biddi);
		bool isSuccess = true;
		// we need a good seed for random number generator. combine three parts to get a seed
		uint seedP1 = uint(cellCenterXCoord * 1.0e5) % 100;
		seedP1 = (seedP1 ^ 0x761c23c) ^ (seedP1 >> 19);
		uint seedP2 = uint(cellCenterYCoord * 1.0e5) % 100;
		seedP2 = (seedP2 + 0x165667b1) + (seedP2 << 5);
		uint seedP3 = (cellRank + 0x7ed55d16) + (cellRank << 12);
		uint seed = seedP1 + seedP2 + seedP3;
		thrust::default_random_engine rng(seed);
		thrust::uniform_real_distribution<double> u0Pi(0, 2.0 * pI);
		double randomAngle = u0Pi(rng);
		double xOffset = _addNodeDistance * cos(randomAngle);
		double yOffset = _addNodeDistance * sin(randomAngle);
		double xCoordNewPt = cellCenterXCoord + xOffset;
		double yCoordNewPt = cellCenterYCoord + yOffset;
		uint cellNodeStartPos = cellRank * _maxNodeOfOneCell;
		uint cellNodeEndPos = cellNodeStartPos + activeNodeCountOfThisCell;
		for (uint i = cellNodeStartPos; i < cellNodeEndPos; i++) {
			double distance = sqrt(
					(xCoordNewPt - _nodeXPosAddress[i])
							* (xCoordNewPt - _nodeXPosAddress[i])
							+ (yCoordNewPt - _nodeYPosAddress[i])
									* (yCoordNewPt - _nodeYPosAddress[i]));
			if (distance < _minDistanceToOtherNode) {
				isSuccess = false;
				break;
			}
		}

		if (isSuccess) {
			_nodeXPosAddress[cellNodeEndPos] = xCoordNewPt;
			_nodeYPosAddress[cellNodeEndPos] = yCoordNewPt;
			isScheduledToGrow = 0;
			activeNodeCountOfThisCell = activeNodeCountOfThisCell + 1;
		}

		return thrust::make_tuple(isScheduledToGrow, activeNodeCountOfThisCell);
	}
};

struct CompuTarLen: thrust::unary_function<double, double> {
	double _cellInitLength, _cellFinalLength;
	__host__ __device__ CompuTarLen(double initLen, double finalLen) :
			_cellInitLength(initLen), _cellFinalLength(finalLen) {
	}
	__host__ __device__ double operator()(const double &progress) {
		return _cellInitLength + progress * (_cellFinalLength - _cellInitLength);
	}
};

//TODO: complete this function
struct CompuDist: thrust::unary_function<CVec6Bool, double> {
	__host__ __device__ double operator()(const CVec6Bool &vec6b) {
		return 0.0;
	}
};

struct CompuDiff: thrust::unary_function<CVec3, double> {
	__host__ __device__ double operator()(const CVec3 &vec3) {
		double expectedLen = thrust::get < 0 > (vec3);
		// minimum distance of node to its corresponding center along growth direction
		double minDistance = thrust::get < 1 > (vec3);
		double maxDistance = thrust::get < 2 > (vec3);
		return (expectedLen - (maxDistance - minDistance));
	}
};

//TODO: complete this function
struct ApplyStretchForce: thrust::unary_function<CVec4, CVec2> {
	double _elongationCoefficient;
	__host__ __device__ ApplyStretchForce(double elongationCoefficient) :
			_elongationCoefficient(elongationCoefficient) {
	}
	__host__  __device__ CVec2 operator()(const CVec4 &vec4) {
		double distToCenterAlongGrowDir = thrust::get < 0 > (vec4);
		// minimum distance of node to its corresponding center along growth direction
		double lengthDifference = thrust::get < 1 > (vec4);
		double growthXDir = thrust::get < 2 > (vec4);
		double growthYDir = thrust::get < 3 > (vec4);
		return thrust::make_tuple(0.0, 0.0);
	}
};

class SceCells {
public:
	// @maxNodeOfOneCell represents maximum number of nodes per cell
	uint maxNodeOfOneCell;
	// @maxCellCount represents maximum number of cells in the system
	uint maxCellCount;
	uint maxTotalCellNodeCount;
	uint currentActiveCellCount;
	// if growthProgress[i] - lastCheckPoint[i] > growThreshold then isScheduledToGrow[i] = true;
	double growThreshold;

	double addNodeDistance;
	double minDistanceToOtherNode;
	double cellInitLength;
	double cellFinalLength;
	double elongationCoefficient;

	SceNodes* nodes;

	// values of these vectors corresponds to each cell.
	// which means these vectors have size of maxCellCount
	// progress == 0 means recently divided
	// progress == 1 means ready to divide
	thrust::device_vector<double> growthProgress;
	thrust::device_vector<double> expectedLength;
	thrust::device_vector<double> currentLength;
	thrust::device_vector<double> lengthDifference;
	// array to hold temp value computed in growth phase.
	// obtained by smallest value of vector product of (a cell node to the cell center)
	// and (growth direction). This will be a negative value
	thrust::device_vector<double> smallestDistance;
	// biggest value of vector product of (a cell node to the cell center)
	// and (growth direction). This will be a positive value
	thrust::device_vector<double> biggestDistance;
	thrust::device_vector<uint> activeNodeCountOfThisCell;
	thrust::device_vector<double> lastCheckPoint;
	thrust::device_vector<bool> isDivided;
	thrust::device_vector<bool> isScheduledToGrow;
	thrust::device_vector<double> centerCoordX;
	thrust::device_vector<double> centerCoordY;
	thrust::device_vector<double> centerCoordZ;

	thrust::device_vector<double> growthSpeed;
	thrust::device_vector<double> growthXDir;
	thrust::device_vector<double> growthYDir;

	// some memory for holding intermediate values instead of dynamically allocating.
	thrust::device_vector<uint> cellRanksTmpStorage;

	// these tmp coordinates will be temporary storage for division info
	// their size will be the same with maximum node count.
	thrust::device_vector<double> xCoordTmp;
	thrust::device_vector<double> yCoordTmp;
	thrust::device_vector<double> zCoordTmp;

	// some memory for holding intermediate values instead of dynamically allocating.
	thrust::device_vector<uint> cellRanks;
	thrust::device_vector<double> activeXPoss;
	thrust::device_vector<double> activeYPoss;
	thrust::device_vector<double> activeZPoss;

	// temp value for holding a node direction to its corresponding center
	thrust::device_vector<double> distToCenterAlongGrowDir;

	SceCells(SceNodes* nodesInput);

	void distributeIsActiveInfo();
	void growAndDivide(double dt);
	void grow2DSimplified(double dt,
			thrust::device_vector<double> &growthFactorMag,
			thrust::device_vector<double> &growthFactorDirXComp,
			thrust::device_vector<double> &growthFactorDirYComp,
			uint GridDimensionX, uint GridDimensionY, double GridSpacing);
	void computeCenterPos();
	void processDivisionInfoAndAddNewCells();
	void divide2DSimplified();
};

#endif /* SCECELLS_H_ */
