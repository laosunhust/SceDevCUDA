#ifndef SCENODES_H_
#define SCENODES_H_

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/tuple.h>
#include <thrust/random.h>

#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/unique.h>

#include <thrust/for_each.h>
#include <cuda_runtime.h>
#include "ConfigParser.h"
#include <assert.h>

/**
 * @mainpage chicken beak development project main documentation page
 *
 * @author Wenzhao Sun wsun2@nd.edu
 *
 *  This simulation package is highly dependent on Thrust library, which is an official
 *  fast - GPU development toolkit released by NVIDIA research.
 *  The code requires CMake to build. When making project, html style documentation will be
 *  automatically generated in "html" folder in project root directory.
 *
 *  1) The goal of the project is to simulate a developmental biology phenomenon
 *  that the shape of the beak is changed from normal triangular shape to
 *  a hooked shape after being treated by VPA, which will support our hypothesis
 *  that the final shape of the chicken beak is determined by the mechanical interaction between cells.
 *
 *  2) There are five essential parts in our simulation model -
 *  FNM bud, MX bud, ECM, epithelium, and external chemical, which is FGF signal.
 *
 *  3) The approximate time scale of our simulation is several days. (day 8 - day 11)?
 */

typedef unsigned int uint;
typedef thrust::tuple<double, double> CVec2;
typedef thrust::tuple<bool, double> BoolD;
typedef thrust::tuple<bool, uint, double> BoolUID;
typedef thrust::tuple<bool, uint, double, double, uint, double> BoolUIDDUID;
typedef thrust::tuple<double, double, double> CVec3;
typedef thrust::tuple<double, double, double, uint> CVec3Int;
typedef thrust::tuple<double, double, double, bool, uint> CVec3BoolInt;
typedef thrust::tuple<double, double, double, double> CVec4;
typedef thrust::tuple<double, double, double, double, double> CVec5;
typedef thrust::tuple<double, double, double, double, double, double> CVec6;
typedef thrust::tuple<double, double, double, double, double, double, bool> CVec6Bool;
typedef thrust::tuple<uint, uint> Tuint2;
typedef thrust::tuple<uint, uint, uint> Tuint3;
typedef thrust::tuple<uint, uint, uint, double, double, double> Tuuuddd;

//extern __constant__ double sceInterPara[4];
//extern __constant__ double sceIntraPara[4];
//extern double sceInterParaCPU[4];
//extern double sceIntraParaCPU[4];

/**
 * enum type cell type.
 */
enum CellType {
	Boundary, Profile, ECM, FNM, MX
};

/**
 * This structure is for visualization purpose only.
 */
struct Point {
	CellType type;
	int rank;
};

/**
 * Depreciated.
 */
struct InitFunctor: public thrust::unary_function<Tuint3, Tuint3> {
	uint maxCellCount;__host__ __device__
	InitFunctor(uint maxCell) :
			maxCellCount(maxCell) {
	}
	__host__ __device__
	Tuint3 operator()(const Tuint3 &v) {
		uint iter = thrust::get<2>(v);
		uint cellRank = iter / maxCellCount;
		uint nodeRank = iter % maxCellCount;
		return thrust::make_tuple(cellRank, nodeRank, iter);
	}
};

/**
 * functor to add two three dimensional vectors.
 */
struct AddFunctor: public thrust::binary_function<CVec3, CVec3, CVec3> {
	double _dt;__host__ __device__
	AddFunctor(double dt) :
			_dt(dt) {
	}

	__host__ __device__
	CVec3 operator()(const CVec3 &vel, const CVec3 &loc) {
		double xMoveDist = thrust::get<0>(vel) * _dt;
		double yMoveDist = thrust::get<1>(vel) * _dt;
		double zMoveDist = thrust::get<2>(vel) * _dt;
		double xPosAfterMove = xMoveDist + thrust::get<0>(loc);
		double yPosAfterMove = yMoveDist + thrust::get<1>(loc);
		double zPosAfterMove = zMoveDist + thrust::get<2>(loc);
		return thrust::make_tuple(xPosAfterMove, yPosAfterMove, zPosAfterMove);
	}
};

/**
 * random number generation engine.
 */
struct Prg {
	double a, b;

	__host__ __device__
	Prg(double _a = 0.f, double _b = 1.f) :
			a(_a), b(_b) {
	}
	__host__ __device__
	double operator()(const unsigned int n) const {
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<double> dist(a, b);
		rng.discard(n);
		return dist(rng);
	}
};

/**
 * map a node coordinate to its bucket index.
 *
 */
struct pointToBucketIndex2D: public thrust::unary_function<CVec3BoolInt, Tuint2> {
	double _minX;
	double _maxX;
	double _minY;
	double _maxY;

	double _bucketSize;
	unsigned int width;

	__host__ __device__
	pointToBucketIndex2D(double minX, double maxX, double minY, double maxY,
			double bucketSize) :
			_minX(minX), _maxX(maxX), _minY(minY), _maxY(maxY), _bucketSize(
					bucketSize), width((maxX - minX) / bucketSize + 1) {
	}

	__host__ __device__
	Tuint2 operator()(const CVec3BoolInt& v) const {
		// find the raster indices of p's bucket
		if (thrust::get<3>(v) == true) {
			unsigned int x = static_cast<unsigned int>((thrust::get<0>(v)
					- _minX) / _bucketSize);
			unsigned int y = static_cast<unsigned int>((thrust::get<1>(v)
					- _minY) / _bucketSize);

			// return the bucket's linear index and node's global index
			return thrust::make_tuple(y * width + x, thrust::get<4>(v));
		} else {
			// return UINT_MAX to indicate the node is inactive and its value should not be used
			return thrust::make_tuple(UINT_MAX, UINT_MAX);
		}
	}
};

/**
 * Functor to compute neighbor buckets(center bucket included) of a node.
 * @param input1 bucket index of node
 * @param input2 pick from the sequence, which is also global rank of the node
 *
 * @return output1 bucket indices of node ( all neighbors and the center bucket) of node
 * @return output2 global rank of node
 */
struct NeighborFunctor2D: public thrust::unary_function<Tuint2, Tuint2> {
	uint _numOfBucketsInXDim;
	uint _numOfBucketsInYDim;__host__ __device__
	NeighborFunctor2D(uint numOfBucketsInXDim, uint numOfBucketsInYDim) :
			_numOfBucketsInXDim(numOfBucketsInXDim), _numOfBucketsInYDim(
					numOfBucketsInYDim) {
	}
	__host__ __device__
	Tuint2 operator()(const Tuint2 &v) {
		uint relativeRank = thrust::get<1>(v) % 9;
		uint xPos = thrust::get<0>(v) % _numOfBucketsInXDim;
		uint yPos = thrust::get<0>(v) / _numOfBucketsInXDim;
		switch (relativeRank) {
		case 0:
			return thrust::make_tuple(thrust::get<0>(v), thrust::get<1>(v));
		case 1:
			if (xPos > 0 && yPos > 0) {
				uint topLeft = (xPos - 1) + (yPos - 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(topLeft, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 2:
			if (yPos > 0) {
				uint top = xPos + (yPos - 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(top, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 3:
			if (yPos > 0 && xPos < _numOfBucketsInXDim - 1) {
				uint topRight = xPos + 1 + (yPos - 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(topRight, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 4:
			if (xPos < _numOfBucketsInXDim - 1) {
				uint right = xPos + 1 + yPos * _numOfBucketsInXDim;
				return thrust::make_tuple(right, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 5:
			if (xPos < _numOfBucketsInXDim - 1
					&& yPos < _numOfBucketsInYDim - 1) {
				uint bottomRight = xPos + 1 + (yPos + 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(bottomRight, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 6:
			if (yPos < _numOfBucketsInYDim - 1) {
				uint bottom = xPos + (yPos + 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(bottom, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 7:
			if (xPos > 0 && yPos < _numOfBucketsInYDim - 1) {
				uint bottomLeft = (xPos - 1) + (yPos + 1) * _numOfBucketsInXDim;
				return thrust::make_tuple(bottomLeft, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		case 8:
			if (xPos > 0) {
				uint left = xPos - 1 + yPos * _numOfBucketsInXDim;
				return thrust::make_tuple(left, thrust::get<1>(v));
			} else {
				return thrust::make_tuple(UINT_MAX, thrust::get<1>(v));
			}
		default:
			return thrust::make_tuple(UINT_MAX, UINT_MAX);
		}
	}
};

/**
 * functor to check if both inputs are non-zero.
 */
struct bothNoneZero {
	__host__ __device__
	bool operator()(Tuint2 v) {
		if (thrust::get<0>(v) != 0 && thrust::get<1>(v) != 0) {
			return true;
		} else {
			return false;
		}
	}
};

/**
 * functor to check if first two of the three inputs are non-zero.
 */
struct bothNoneZero2 {
	__host__ __device__
	bool operator()(Tuint3 v) {
		if (thrust::get<0>(v) != 0 && thrust::get<1>(v) != 0) {
			return true;
		} else {
			return false;
		}
	}
};

/**
 * functor to check if an integer is one.
 */
struct isOne {
	__host__ __device__
	bool operator()(int i) {
		if (i == 1) {
			return true;
		} else {
			return false;
		}
	}
};

__device__
double computeDist(double &xPos, double &yPos, double &zPos, double &xPos2,
		double &yPos2, double &zPos2);

__device__
void calculateAndAddECMForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes);

__device__
void calculateAndAddInterForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes);

__device__
void calculateAndAddIntraForce(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes);

__device__
void calculateAndAddInterForceDiffType(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes);

__device__ bool bothNodesCellNode(uint nodeGlobalRank1, uint nodeGlobalRank2,
		uint cellNodesThreshold);

__device__ bool isSameCell(uint nodeGlobalRank1, uint nodeGlobalRank2,
		uint nodeCountPerCell);

__device__ bool ofSameType(uint cellType1, uint cellType2);

__device__
void handleForceBetweenNodes(uint &nodeRank1, CellType &type1, uint &nodeRank2,
		CellType &type2, double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes, double* _nodeLocXAddress, double* _nodeLocYAddress,
		double* _nodeLocZAddress, uint startPosOfCells);

/**
 * a complicated data structure for adding subcellular element force to cell nodes.
 * This data structure is designed in an unconventional way because of performance considerations.
 */
struct AddSceForce: public thrust::unary_function<Tuuuddd, CVec3> {
	uint* _extendedValuesAddress;
	double* _nodeLocXAddress;
	double* _nodeLocYAddress;
	double* _nodeLocZAddress;
	uint* _nodeRankAddress;
	CellType* _cellTypesAddress;
	uint _cellNodesThreshold;
	uint _cellNodesStartPos;
	uint _nodeCountPerCell;
	uint _cellNodesPerECM;__host__ __device__
	AddSceForce(uint* valueAddress, double* nodeLocXAddress,
			double* nodeLocYAddress, double* nodeLocZAddress,
			uint* nodeRankAddress, CellType* cellTypesAddress,
			uint cellNodesThreshold, uint nodeCountPerCell,
			uint cellNodesStartPos, uint celLNodesPerECM) :
			_extendedValuesAddress(valueAddress), _nodeLocXAddress(
					nodeLocXAddress), _nodeLocYAddress(nodeLocYAddress), _nodeLocZAddress(
					nodeLocZAddress), _nodeRankAddress(nodeRankAddress), _cellTypesAddress(
					cellTypesAddress), _cellNodesThreshold(cellNodesThreshold), _cellNodesStartPos(
					cellNodesStartPos), _nodeCountPerCell(nodeCountPerCell), _cellNodesPerECM(
					celLNodesPerECM) {
	}
	__device__
	CVec3 operator()(const Tuuuddd &u3d3) const {
		double xRes = 0.0;
		double yRes = 0.0;
		double zRes = 0.0;

		uint begin = thrust::get<0>(u3d3);
		uint end = thrust::get<1>(u3d3);
		uint myValue = thrust::get<2>(u3d3);
		double xPos = thrust::get<3>(u3d3);
		double yPos = thrust::get<4>(u3d3);
		double zPos = thrust::get<5>(u3d3);

		CellType myType = _cellTypesAddress[myValue];

		for (uint i = begin; i < end; i++) {
			//for (uint i = begin; i < begin+1; i++) {
			uint nodeRankOfOtherNode = _extendedValuesAddress[i];
			CellType cellTypeOfOtherNode =
					_cellTypesAddress[nodeRankOfOtherNode];
			if (nodeRankOfOtherNode == myValue) {
				continue;
			}

			handleForceBetweenNodes(myValue, myType, nodeRankOfOtherNode,
					cellTypeOfOtherNode, xPos, yPos, zPos,
					_nodeLocXAddress[nodeRankOfOtherNode],
					_nodeLocYAddress[nodeRankOfOtherNode],
					_nodeLocZAddress[nodeRankOfOtherNode], xRes, yRes, zRes,
					_nodeLocXAddress, _nodeLocYAddress, _nodeLocZAddress,
					_cellNodesStartPos);

		}

		return thrust::make_tuple(xRes, yRes, zRes);
	}
};

/**
 * return a tuple of three zero double numbers.
 */
struct GetZeroTupleThree: public thrust::unary_function<uint, CVec3> {
	__host__ __device__
	CVec3 operator()(const uint &value) {
		return thrust::make_tuple(0.0, 0.0, 0.0);
	}
};

/**
 * return true if this number is less than initial parameter.
 */
struct isLessThan: public thrust::unary_function<uint, bool> {
	uint initValue;__host__ __device__
	isLessThan(uint initV) :
			initValue(initV) {
	}

	__host__ __device__
	bool operator()(uint val) {
		if (val < initValue) {
			return true;
		} else {
			return false;
		}
	}
};

/**
 * Data structure for calculating nearby node interaction.
 * To maximize GPU performance, I choose to implement Structure of Arrays(SOA)
 * instead of Array Of Structure, which is commonly used in serial algorithms.
 * This data structure is a little bit counter-intuitive, but would expect to
 * give 2X - 3X performance gain.
 * To gain a better performance on GPU, we pre-allocate memory for everything that
 * will potentially
 * This data structure need to be initialized with following parameters:
 * 1) maximum number of cells to be simulated in the simulation domain
 * 2) maximum number of nodes inside a cell
 * 3) maximum number of uintra-cellular links per node
 * 4) maximum number of uinter-cellular links per node
 */
class SceNodes {
public:
	SceNodes() {
	}
	// in the new version, we need to initialize the nodes together with initial
	// boundary nodes and initial profile nodes.
	SceNodes(uint totalBdryNodeCount, uint maxProfileNodeCount,
			uint maxTotalECMCount, uint maxNodeInECM, uint maxTotalCellCount,
			uint maxNodeInCell);

	// parameters for discretizing domain into grids
	uint numOfBucketsInXDim;
	uint numOfBucketsInYDim;
	uint totalBucketCount;

	double minX;
	double maxX;
	double minY;
	double maxY;
	double bucketSize;

	thrust::device_vector<unsigned int> keyBegin;
	thrust::device_vector<unsigned int> keyEnd;

	// @maxNodeOfOneCell represents maximum number of nodes per cell
	uint maxNodeOfOneCell;
	// @maxCellCount represents maximum number of cells in the system
	uint maxCellCount;
	// @maxCellCount represents maximum number of cells (including pesudo-cells) in the system
	uint maxCellCountAll;
	// @maxTotalNodeCount represents maximum total number of nodes of all cells
	// maxTotalCellNodeCount = maxNodeOfOneCell * maxCellCount;
	uint maxTotalCellNodeCount;
	// @currentActiveCellCount represents number of cells that are currently active.
	uint currentActiveCellCount;

	// @maxNodePerECM represents maximum number of nodes per ECM
	uint maxNodePerECM;
	// @maxECMCount represents maximum number of ECM
	uint maxECMCount;
	// @maxTotalECMNodeCount represents maximum total number of node of ECM
	uint maxTotalECMNodeCount;
	// @currentActiveECM represents number of ECM that are currently active.
	uint currentActiveECM;

	// epithilum might grow or might not. Set maxProfileNodeCount as the maximum possible node count
	uint maxProfileNodeCount;
	// no matter whether epithilum grow or not we need to track the cucrent count.
	uint currentActiveProfileNodeCount;

	uint BdryNodeCount;

	// @cellSpaceForBdry First several spaces are reserved for boundary.
	// depreciated.
	// uint cellSpaceForBdry;
	// because several spaces are reserved for boundary, some nodes can't move
	// depreciated.
	// uint fixedNodeCount;

	uint startPosProfile;
	uint startPosECM;
	uint startPosCells;

	//thrust::device_vector<uint> cellRanks;
	//thrust::device_vector<uint> nodeRanks;

	// for all of these values that are allocated for all nodes, the actual value is meaningless
	// if the cell it belongs to is inactive

	// this vector is used to indicate whether a node is active or not.
	// E.g, max number of nodes of a cell is 100 maybe the first 75 nodes are active.
	// The value of this vector will be changed by external process.
	thrust::device_vector<bool> nodeIsActive;
	// X locations of nodes
	thrust::device_vector<double> nodeLocX;
	// Y locations of nodes
	thrust::device_vector<double> nodeLocY;
	// Z locations of nodes
	thrust::device_vector<double> nodeLocZ;
	// X velocities of nodes
	thrust::device_vector<double> nodeVelX;
	// Y velocities of nodes
	thrust::device_vector<double> nodeVelY;
	// Z velocities of nodes
	thrust::device_vector<double> nodeVelZ;

	// in order to represent differential adhesion, we also need an vector
	// for each cell node to identify the cell type.
	thrust::device_vector<CellType> nodeCellType;

	// for each node, we need to identify which cell it belongs to.
	thrust::device_vector<uint> nodeCellRank;

	// bucket key means which bucket ID does a certain point fit into
	thrust::device_vector<uint> bucketKeys;
	// bucket value means global rank of a certain point
	thrust::device_vector<uint> bucketValues;
	// bucket key expanded means what are the bucket IDs are the neighbors of a certain point
	thrust::device_vector<uint> bucketKeysExpanded;
	// bucket value expanded means each point ( represented by its global rank) will have multiple copies
	thrust::device_vector<uint> bucketValuesIncludingNeighbor;

	std::vector<std::pair<uint, uint> > obtainNeighborPairs();

	/**
	 * This function copies parameters to GPU constant memory.
	 */
	void copyParaToGPUConstMem();

	/**
	 * this method maps the points to their bucket ID.
	 * writes data to thrust::device_vector<uint> bucketValues and
	 * thrust::device_vector<uint> bucketValues;
	 * each data in bucketValues will represent
	 */
	void buildBuckets2D();
	/**
	 * this method extends the previously obtained vector of keys and values to a map
	 * that each point will map to all bucket IDs that are near the specific point.
	 */
	void extendBuckets2D();

	/**
	 * @brief This is the most important part of the parallel algorithm.
	 * For each particle in SCE model (represented by bucketValues here),
	 * the algorithm finds all particles that fits in the nearby grids and then
	 * apply forces with them.
	 * The algorithm is very scalable.
	 */
	void applySceForces();

	void initDimension(double minX, double maxX, double minY, double maxY,
			double bucketSize);
	void initValues(std::vector<double> &initBdryCellNodePosX,
			std::vector<double> &initBdryCellNodePosY,
			std::vector<double> &initProfileNodePosX,
			std::vector<double> &initProfileNodePosY,
			std::vector<double> &initECMNodePosX,
			std::vector<double> &initECMNodePosY,
			std::vector<double> &initFNMCellNodePosX,
			std::vector<double> &initFNMCellNodePosY,
			std::vector<double> &initMXCellNodePosX,
			std::vector<double> &initMXCellNodePosY);

	/**
	 * wrap three methods together.
	 */
	void calculateAndApplySceForces();

	/**
	 * add maxNodeOfOneCell
	 */
	void addNewlyDividedCells(thrust::device_vector<double> &nodeLocXNewCell,
			thrust::device_vector<double> &nodeLocYNewCell,
			thrust::device_vector<double> &nodeLocZNewCell,
			thrust::device_vector<bool> &nodeIsActive);
	void addNewlyDividedCells(thrust::device_vector<double> &nodeLocXNewCell,
			thrust::device_vector<double> &nodeLocYNewCell,
			thrust::device_vector<double> &nodeLocZNewCell,
			thrust::device_vector<bool> &nodeIsActiveNewCell,
			thrust::device_vector<CellType> &nodeCellTypeNewCell);

	uint getCurrentActiveCellCount() const {
		return currentActiveCellCount;
	}

	void setCurrentActiveCellCount(uint currentActiveCellCount) {
		this->currentActiveCellCount = currentActiveCellCount;
	}

	uint getCurrentActiveEcm() const {
		return currentActiveECM;
	}

	void setCurrentActiveEcm(uint currentActiveEcm) {
		currentActiveECM = currentActiveEcm;
	}

	uint getMaxCellCount() const {
		return maxCellCount;
	}

	void setMaxCellCount(uint maxCellCount) {
		this->maxCellCount = maxCellCount;
	}

	uint getMaxNodeOfOneCell() const {
		return maxNodeOfOneCell;
	}

	void setMaxNodeOfOneCell(uint maxNodeOfOneCell) {
		this->maxNodeOfOneCell = maxNodeOfOneCell;
	}

	uint getMaxTotalCellNodeCount() const {
		return maxTotalCellNodeCount;
	}

	void setMaxTotalCellNodeCount(uint maxTotalCellNodeCount) {
		this->maxTotalCellNodeCount = maxTotalCellNodeCount;
	}

	uint getCurrentActiveProfileNodeCount() const {
		return currentActiveProfileNodeCount;
	}

	void setCurrentActiveProfileNodeCount(uint currentActiveProfileNodeCount) {
		this->currentActiveProfileNodeCount = currentActiveProfileNodeCount;
	}

	uint getMaxEcmCount() const {
		return maxECMCount;
	}

	void setMaxEcmCount(uint maxEcmCount) {
		maxECMCount = maxEcmCount;
	}

	uint getMaxNodePerEcm() const {
		return maxNodePerECM;
	}

	void setMaxNodePerEcm(uint maxNodePerEcm) {
		maxNodePerECM = maxNodePerEcm;
	}

	uint getMaxProfileNodeCount() const {
		return maxProfileNodeCount;
	}

	void setMaxProfileNodeCount(uint maxProfileNodeCount) {
		this->maxProfileNodeCount = maxProfileNodeCount;
	}

	uint getMaxCellCountAll() const {
		return maxCellCountAll;
	}

	void setMaxCellCountAll(uint maxCellCountAll) {
		this->maxCellCountAll = maxCellCountAll;
	}
};

#endif /* SCENODES_H_ */
