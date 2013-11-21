#include <iostream>
#include "gtest/gtest.h"
#include "SceNodes.h"
#include <vector>
#include <cuda_runtime.h>
#include <algorithm>
using namespace std;

const int myDeviceId = 2;
const uint maxCellCount = 10;
const uint maxNodePerCell = 100;
const uint maxNodeCount = maxCellCount * maxNodePerCell;
const uint maxInterLink = 3;
const uint maxIntraLink = 5;
const double dt = 0.1;
const double errTol = 1.0e-12;

GlobalConfigVars globalConfigVars;

double computeDistInTest(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2) {
	return sqrt(
			(xPos - xPos2) * (xPos - xPos2) + (yPos - yPos2) * (yPos - yPos2)
					+ (zPos - zPos2) * (zPos - zPos2));
}

void calculateAndAddIntraForceInTest(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes, vector<double> &sceIntraPara) {
	double linkLength = computeDistInTest(xPos, yPos, zPos, xPos2, yPos2,
			zPos2);
	double forceValue = -sceIntraPara[0] / sceIntraPara[2]
			* exp(-linkLength / sceIntraPara[2])
			+ sceIntraPara[1] / sceIntraPara[3]
					* exp(-linkLength / sceIntraPara[3]);
	xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
	yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
	zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
}
void calculateAndAddInterForceInTest(double &xPos, double &yPos, double &zPos,
		double &xPos2, double &yPos2, double &zPos2, double &xRes, double &yRes,
		double &zRes, vector<double> &sceInterPara) {
	double linkLength = computeDistInTest(xPos, yPos, zPos, xPos2, yPos2,
			zPos2);
	double forceValue = 0;
	if (linkLength > sceInterPara[4]) {
		forceValue = 0;
	} else {
		forceValue = -sceInterPara[0] / sceInterPara[2]
				* exp(-linkLength / sceInterPara[2])
				+ sceInterPara[1] / sceInterPara[3]
						* exp(-linkLength / sceInterPara[3]);
	}
	//if (forceValue > 0) {
	//	forceValue = 0;
	//}
	xRes = xRes + forceValue * (xPos2 - xPos) / linkLength;
	yRes = yRes + forceValue * (yPos2 - yPos) / linkLength;
	zRes = zRes + forceValue * (zPos2 - zPos) / linkLength;
}

void computeResultFromCPUAllIntra2D(vector<double> &xPoss,
		vector<double> &yPoss, vector<double> &zPoss, vector<double> &xVels,
		vector<double> &yVels, vector<double> &zVels, vector<bool> &isActive,
		vector<double> &paraSet, double bucketSize, double minX, double maxX,
		double minY, double maxY) {
	unsigned int i, j;
	//double xMax = *std::max_element(xPoss.begin(), xPoss.end());
	//double xMin = *std::min_element(xPoss.begin(), xPoss.end());
	//double yMax = *std::max_element(yPoss.begin(), yPoss.end());
	//double yMin = *std::min_element(yPoss.begin(), yPoss.end());
	//uint xBucketCount = (xMax - xMin) / bucketSize + 1;
	//uint yBuckeyCount = (yMax - yMin) / bucketSize + 1;
	for (i = 0; i < xVels.size(); i++) {
		xVels[i] = 0.0;
		yVels[i] = 0.0;
		zVels[i] = 0.0;
	}
	for (i = 0; i < xPoss.size(); i++) {
		if (isActive[i] == true) {
			int xBucketPos = (xPoss[i] - minX) / bucketSize;
			int yBucketPos = (yPoss[i] - minY) / bucketSize;
			for (j = 0; j < xPoss.size(); j++) {
				if (j != i) {
					if (isActive[j] == true) {
						int xBucketPos2 = (xPoss[j] - minX) / bucketSize;
						int yBucketPos2 = (yPoss[j] - minY) / bucketSize;
						if (abs(xBucketPos - xBucketPos2) <= 1
								&& abs(yBucketPos - yBucketPos2) <= 1) {
							calculateAndAddIntraForceInTest(xPoss[i], yPoss[i],
									zPoss[i], xPoss[j], yPoss[j], zPoss[i],
									xVels[i], yVels[i], zVels[i], paraSet);
						}
					}
				}
			}
		}
	}
}

void computeResultFromCPUAllIntraAndInter2D(vector<double> &xPoss,
		vector<double> &yPoss, vector<double> &zPoss, vector<double> &xVels,
		vector<double> &yVels, vector<double> &zVels, vector<bool> &isActive,
		vector<double> &paraSet1, vector<double> &paraSet2, double bucketSize,
		uint activeCellCount, uint maxNodesPerCell, double minX, double maxX,
		double minY, double maxY) {
	unsigned int i, j;
	uint numberOfActiveNodes = activeCellCount * maxNodesPerCell;

	for (i = 0; i < numberOfActiveNodes; i++) {
		xVels[i] = 0.0;
		yVels[i] = 0.0;
		zVels[i] = 0.0;
	}
	for (i = 0; i < numberOfActiveNodes; i++) {
		if (isActive[i] == true) {
			int xBucketPos = (xPoss[i] - minX) / bucketSize;
			int yBucketPos = (yPoss[i] - minY) / bucketSize;
			for (j = 0; j < numberOfActiveNodes; j++) {
				if (j != i) {
					if (isActive[j] == true) {
						int xBucketPos2 = (xPoss[j] - minX) / bucketSize;
						int yBucketPos2 = (yPoss[j] - minY) / bucketSize;
						if (abs(xBucketPos - xBucketPos2) <= 1
								&& abs(yBucketPos - yBucketPos2) <= 1) {
							if (i / maxNodesPerCell == j / maxNodesPerCell) {
								calculateAndAddIntraForceInTest(xPoss[i],
										yPoss[i], zPoss[i], xPoss[j], yPoss[j],
										zPoss[j], xVels[i], yVels[i], zVels[i],
										paraSet1);

							} else {
								calculateAndAddInterForceInTest(xPoss[i],
										yPoss[i], zPoss[i], xPoss[j], yPoss[j],
										zPoss[j], xVels[i], yVels[i], zVels[i],
										paraSet2);
								//std::cout << "inter logic:" << std::endl;
								//cin >> j;
							}
						}
					}
				} else {
					continue;
				}
			}
		} else {
			continue;
		}
	}
}

class SceNodeTest: public ::testing::Test {
protected:
	double sceInterParaCPU[5];
	double sceIntraParaCPU[4];
	virtual void SetUp() {
		ConfigParser parser;
		std::string configFileName = "sceCell.config";
		globalConfigVars = parser.parseConfigFile(configFileName);
		static const double U0 =
				globalConfigVars.getConfigValue("InterCell_U0_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"InterCell_U0_DivFactor").toDouble();
		static const double V0 =
				globalConfigVars.getConfigValue("InterCell_V0_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"InterCell_V0_DivFactor").toDouble();
		static const double k1 =
				globalConfigVars.getConfigValue("InterCell_k1_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"InterCell_k1_DivFactor").toDouble();
		static const double k2 =
				globalConfigVars.getConfigValue("InterCell_k2_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"InterCell_k2_DivFactor").toDouble();
		static const double interLinkEffectiveRange =
				globalConfigVars.getConfigValue("InterCellLinkBreakRange").toDouble();

		sceInterParaCPU[0] = U0;
		sceInterParaCPU[1] = V0;
		sceInterParaCPU[2] = k1;
		sceInterParaCPU[3] = k2;
		sceInterParaCPU[4] = interLinkEffectiveRange;

		static const double U0_Intra =
				globalConfigVars.getConfigValue("IntraCell_U0_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"IntraCell_U0_DivFactor").toDouble();
		static const double V0_Intra =
				globalConfigVars.getConfigValue("IntraCell_V0_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"IntraCell_V0_DivFactor").toDouble();
		static const double k1_Intra =
				globalConfigVars.getConfigValue("IntraCell_k1_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"IntraCell_k1_DivFactor").toDouble();
		static const double k2_Intra =
				globalConfigVars.getConfigValue("IntraCell_k2_Original").toDouble()
						/ globalConfigVars.getConfigValue(
								"IntraCell_k2_DivFactor").toDouble();
		sceIntraParaCPU[0] = U0_Intra;
		sceIntraParaCPU[1] = V0_Intra;
		sceIntraParaCPU[2] = k1_Intra;
		sceIntraParaCPU[3] = k2_Intra;
	}
};

TEST(SceNodeConstructor, sizeTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	//EXPECT_EQ(nodes.cellRanks.size(), maxNodeCount);
	//EXPECT_EQ(nodes.nodeRanks.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeLocX.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeLocY.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeLocZ.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeVelX.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeVelY.size(), maxNodeCount);
	EXPECT_EQ(nodes.nodeVelZ.size(), maxNodeCount);
}

TEST(SceNodeConstructor, initTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	//thrust::host_vector<uint> cellRanksComputeFromDevice = nodes.cellRanks;
	//thrust::host_vector<uint> nodeRanksComputeFromDevice = nodes.nodeRanks;
	//vector<uint> cellRanksComputeFromHost;
	//vector<uint> nodeRanksComputeFromHost;
	//cellRanksComputeFromHost.resize(maxNodeCount);
	//nodeRanksComputeFromHost.resize(maxNodeCount);
	//for (uint i = 0; i < maxNodeCount; i++) {
	//	cellRanksComputeFromHost[i] = i / maxCellCount;
	//	nodeRanksComputeFromHost[i] = i % maxCellCount;
	//}
	//for (uint i = 0; i < maxNodeCount; i++) {
	//	EXPECT_EQ(cellRanksComputeFromHost[i], cellRanksComputeFromDevice[i]);
	//	EXPECT_EQ(nodeRanksComputeFromHost[i], nodeRanksComputeFromDevice[i]);
	//}
}

TEST(SceNodeFunction, moveTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	thrust::host_vector<double> nodeLocXHost(maxNodeCount);
	thrust::host_vector<double> nodeLocYHost(maxNodeCount);
	thrust::host_vector<double> nodeLocZHost(maxNodeCount);
	thrust::host_vector<double> nodeVelXHost(maxNodeCount);
	thrust::host_vector<double> nodeVelYHost(maxNodeCount);
	thrust::host_vector<double> nodeVelZHost(maxNodeCount);
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocXHost.begin(), Prg());
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocYHost.begin(), Prg());
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocZHost.begin(), Prg());
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeVelXHost.begin(), Prg());
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeVelYHost.begin(), Prg());
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeVelZHost.begin(), Prg());

	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeVelX = nodeVelXHost;
	nodes.nodeVelY = nodeVelYHost;
	nodes.nodeVelZ = nodeVelZHost;
	nodes.move(dt);

	thrust::host_vector<double> nodeLocXComputeFromHost(maxNodeCount);
	thrust::host_vector<double> nodeLocYComputeFromHost(maxNodeCount);
	thrust::host_vector<double> nodeLocZComputeFromHost(maxNodeCount);

	for (uint i = 0; i < maxNodeCount; i++) {
		nodeLocXComputeFromHost[i] = nodeLocXHost[i] + dt * nodeVelXHost[i];
		nodeLocYComputeFromHost[i] = nodeLocYHost[i] + dt * nodeVelYHost[i];
		nodeLocZComputeFromHost[i] = nodeLocZHost[i] + dt * nodeVelZHost[i];
	}
	thrust::host_vector<double> nodeLocXComputeFromDevice = nodes.nodeLocX;
	thrust::host_vector<double> nodeLocYComputeFromDevice = nodes.nodeLocY;
	thrust::host_vector<double> nodeLocZComputeFromDevice = nodes.nodeLocZ;

	for (uint i = 0; i < maxNodeCount; i++) {
		EXPECT_NEAR(nodeLocXComputeFromHost[i], nodeLocXComputeFromDevice[i],
				errTol);
		EXPECT_NEAR(nodeLocYComputeFromHost[i], nodeLocYComputeFromDevice[i],
				errTol);
		EXPECT_NEAR(nodeLocZComputeFromHost[i], nodeLocZComputeFromDevice[i],
				errTol);
	}
}

TEST(SceBuildBucket2D, putBucketfixedTest) {
	cudaSetDevice(myDeviceId);
	const uint testCellCount = 2;
	const uint testNodePerCell = 2;
	const uint testTotalNodeCount = testCellCount * testNodePerCell;
	SceNodes nodes(testCellCount, testNodePerCell);
	nodes.setCurrentActiveCellCount(testCellCount);
	thrust::host_vector<double> nodeLocXHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocYHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocZHost(testTotalNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(testTotalNodeCount);
	thrust::host_vector<uint> nodeExpectedBucket(testTotalNodeCount);
	const double minX = 0.0;
	const double maxX = 0.99999;
	const double minY = 0.0;
	const double maxY = 0.99999;
	const double bucketSize = 0.1;
	nodeLocXHost[0] = 0.0;
	nodeLocYHost[0] = 0.0;
	nodeIsActiveHost[0] = 1;
	nodeExpectedBucket[0] = 0;
	nodeLocXHost[1] = 0.199;
	nodeLocYHost[1] = 0.4;
	nodeIsActiveHost[1] = 1;
	nodeExpectedBucket[1] = 41;
	nodeLocXHost[2] = 0.2;
	nodeLocYHost[2] = 0.4;
	nodeIsActiveHost[2] = 1;
	nodeExpectedBucket[2] = 42;
	nodeLocXHost[3] = 0.5;
	nodeLocYHost[3] = 0.212;
	nodeIsActiveHost[3] = 0;
	nodeExpectedBucket[3] = 25;
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	thrust::host_vector<uint> keysFromGPU = nodes.bucketKeys;
	thrust::host_vector<uint> valuesFromGPU = nodes.bucketValues;
	uint activeNodeCount = 0;
	for (uint i = 0; i < nodeIsActiveHost.size(); i++) {
		if (nodeIsActiveHost[i] == true) {
			activeNodeCount++;
		}
	}
	EXPECT_EQ(keysFromGPU.size(), activeNodeCount);
	EXPECT_EQ(keysFromGPU.size(), valuesFromGPU.size());
	for (uint i = 0; i < keysFromGPU.size(); i++) {
		uint nodeRank = valuesFromGPU[i];
		uint expectedResultFromCPU = nodeExpectedBucket[nodeRank];
		EXPECT_EQ(expectedResultFromCPU, keysFromGPU[i]);
	}
}

TEST(SceBuildBucket2D, putBucketRandomTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	nodes.setCurrentActiveCellCount(maxCellCount);
	thrust::host_vector<double> nodeLocXHost(maxNodeCount);
	thrust::host_vector<double> nodeLocYHost(maxNodeCount);
	thrust::host_vector<double> nodeLocZHost(maxNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(maxNodeCount);
	const double minX = 0.5;
	const double maxX = 1.5;
	const double minY = 0.1;
	const double maxY = 2.2;
	const double minZ = 0.0;
	const double maxZ = 0.0;
	const double bucketSize = 0.1;
	const int width = (maxX - minX) / bucketSize + 1;
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocXHost.begin(), Prg(minX, maxX));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocYHost.begin(), Prg(minY, maxY));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocZHost.begin(), Prg(minZ, maxZ));
	for (uint i = 0; i < maxNodeCount; i++) {
		if (i % 2 == 0) {
			nodeIsActiveHost[i] = true;
		} else {
			nodeIsActiveHost[i] = false;
		}
	}
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	thrust::host_vector<uint> keysFromGPU = nodes.bucketKeys;
	thrust::host_vector<uint> valuesFromGPU = nodes.bucketValues;
	uint activeNodeCount = 0;
	for (uint i = 0; i < nodeIsActiveHost.size(); i++) {
		if (nodeIsActiveHost[i] == true) {
			activeNodeCount++;
		}
	}
	EXPECT_EQ(keysFromGPU.size(), activeNodeCount);
	EXPECT_EQ(valuesFromGPU.size(), activeNodeCount);
	for (uint i = 0; i < activeNodeCount; i++) {
		uint nodeRank = valuesFromGPU[i];
		uint resultFromCPU =
				(int) ((nodeLocXHost[nodeRank] - minX) / bucketSize)
						+ (int) ((nodeLocYHost[nodeRank] - minY) / bucketSize)
								* width;
		EXPECT_EQ(resultFromCPU, keysFromGPU[i]);
	}
}

TEST(SceExtendBucket2D, extendBucketfixedTest) {
	cudaSetDevice(myDeviceId);
	const uint testCellCount = 2;
	const uint testNodePerCell = 2;
	const uint testTotalNodeCount = testCellCount * testNodePerCell;
	SceNodes nodes(testCellCount, testNodePerCell);
	nodes.setCurrentActiveCellCount(testCellCount);
	thrust::host_vector<double> nodeLocXHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocYHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocZHost(testTotalNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(testTotalNodeCount);
	thrust::host_vector<uint> nodeExpectedBucket(testTotalNodeCount);
	const double minX = 0.0;
	const double maxX = 0.99999;
	const double minY = 0.0;
	const double maxY = 0.99999;
//const double minZ = 0.0;
//const double maxZ = 0.0;
	const double bucketSize = 0.1;
	nodeLocXHost[0] = 0.0;
	nodeLocYHost[0] = 0.0;
	nodeIsActiveHost[0] = true;
// 0
	nodeLocXHost[1] = 0.51;
	nodeLocYHost[1] = 0.212;
	nodeIsActiveHost[1] = true;
// 25
	nodeLocXHost[2] = 0.52;
	nodeLocYHost[2] = 0.211;
	nodeIsActiveHost[2] = true;
// 25
	nodeLocXHost[3] = 0.63;
	nodeLocYHost[3] = 0.207;
	nodeIsActiveHost[3] = false;
// 26
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
	const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
	nodes.extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);

	thrust::host_vector<uint> extendedKeysFromGPU = nodes.bucketKeysExpanded;
	thrust::host_vector<uint> extendValuesFromGPU =
			nodes.bucketValuesIncludingNeighbor;
	EXPECT_EQ(extendedKeysFromGPU.size(), (uint )22);
	EXPECT_EQ(extendValuesFromGPU.size(), (uint )22);
	int expectedKeys[] = { 0, 1, 10, 11, 14, 14, 15, 15, 16, 16, 24, 24, 25, 25,
			26, 26, 34, 34, 35, 35, 36, 36 };
	std::vector<uint> expectedResultsKeys(expectedKeys, expectedKeys + 22);
	for (int i = 0; i < 22; i++) {
		EXPECT_EQ(expectedResultsKeys[i], extendedKeysFromGPU[i]);
	}
}

/*
 * Expected size of the extended buckets equals to computed
 * all results fits the requirement
 * no duplicate */

TEST(SceExtendBucket2D, extendBucketRandomTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	nodes.setCurrentActiveCellCount(maxCellCount);
	thrust::host_vector<double> nodeLocXHost(maxNodeCount);
	thrust::host_vector<double> nodeLocYHost(maxNodeCount);
	thrust::host_vector<double> nodeLocZHost(maxNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(maxNodeCount);
	const double minX = 0.8;
	const double maxX = 1.9;
	const double minY = 0.6;
	const double maxY = 3.2;
	const double minZ = 0.0;
	const double maxZ = 0.0;
	const double bucketSize = 0.12;
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocXHost.begin(), Prg(minX, maxX));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocYHost.begin(), Prg(minY, maxY));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocZHost.begin(), Prg(minZ, maxZ));
	for (uint i = 0; i < maxNodeCount; i++) {
		if (i % 3 == 0) {
			nodeIsActiveHost[i] = false;
		} else {
			nodeIsActiveHost[i] = true;
		}
	}
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
	const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
	nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	nodes.extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);

	thrust::host_vector<uint> extendedKeysFromGPU = nodes.bucketKeysExpanded;
	thrust::host_vector<uint> extendValuesFromGPU =
			nodes.bucketValuesIncludingNeighbor;
	uint expectedResultKeyCount = 0;
	for (uint i = 0; i < maxNodeCount; i++) {
		if (nodeIsActiveHost[i] == true) {
			int xPos = (int) ((nodeLocXHost[i] - minX) / bucketSize);
			int yPos = (int) ((nodeLocYHost[i] - minY) / bucketSize);
			int xQuota = 3;
			int yQuota = 3;
			if (xPos == 0) {
				xQuota--;
			}
			if (xPos == numberOfBucketsInXDim - 1) {
				xQuota--;
			}
			if (yPos == 0) {
				yQuota--;
			}
			if (yPos == numberOfBucketsInYDim - 1) {
				yQuota--;
			}
			expectedResultKeyCount += xQuota * yQuota;
		}
	}
// verify if size is correct
	EXPECT_EQ(expectedResultKeyCount, extendedKeysFromGPU.size());
	EXPECT_EQ(expectedResultKeyCount, extendValuesFromGPU.size());
// verify if all key- value pairs fits our the requirement
	for (uint i = 0; i < extendValuesFromGPU.size(); i++) {
		int nodeRank = extendValuesFromGPU[i];
		int xPos = (int) ((nodeLocXHost[nodeRank] - minX) / bucketSize);
		int yPos = (int) ((nodeLocYHost[nodeRank] - minY) / bucketSize);
		int bucketXPos = extendedKeysFromGPU[i] % numberOfBucketsInXDim;
		int bucketYPos = extendedKeysFromGPU[i] / numberOfBucketsInXDim;
		EXPECT_TRUE(abs(xPos - bucketXPos) <= 1);
		EXPECT_TRUE(abs(yPos - bucketYPos) <= 1);
	}
//verify for each key, there is no duplicate for its values
	std::vector<uint> previousValues;
	bool startNewFlag = 1;
	for (uint i = 0; i < extendedKeysFromGPU.size(); i++) {
		if (startNewFlag == 1) {
			previousValues.clear();
			previousValues.push_back(extendValuesFromGPU[i]);
			startNewFlag = 0;
		} else {
			for (uint j = 0; j < previousValues.size(); j++) {
				EXPECT_TRUE(extendValuesFromGPU[i] != previousValues[j]);
			}
			previousValues.push_back(extendValuesFromGPU[i]);
		}

		if (i < extendedKeysFromGPU.size() - 1) {
			if (extendedKeysFromGPU[i] != extendedKeysFromGPU[i + 1]) {
				startNewFlag = 1;
			}
		}
	}

}
/*
 TEST(SceBuildPairs, buildPairFixedTest) {
 cudaSetDevice(myDeviceId);
 const uint testCellCount = 2;
 const uint testNodePerCell = 2;
 const uint testTotalNodeCount = testCellCount * testNodePerCell;
 SceNodes nodes(testCellCount, testNodePerCell);
 nodes.setCurrentActiveCellCount(testCellCount);
 thrust::host_vector<double> nodeLocXHost(testTotalNodeCount);
 thrust::host_vector<double> nodeLocYHost(testTotalNodeCount);
 thrust::host_vector<double> nodeLocZHost(testTotalNodeCount);
 const double minX = 0.0;
 const double maxX = 3.0 - 1.0e-10;
 const double minY = 0.0;
 const double maxY = 2.0 - 1.0e-10;
 //const double minZ = 0.0;
 //const double maxZ = 0.0;
 const double bucketSize = 1.0;
 nodeLocXHost[0] = 0.2;
 nodeLocYHost[0] = 0.5;
 // 0
 nodeLocXHost[1] = 1.2;
 nodeLocYHost[1] = 0.2;
 // 1
 nodeLocXHost[2] = 1.3;
 nodeLocYHost[2] = 0.5;
 // 1
 nodeLocXHost[3] = 2.7;
 nodeLocYHost[3] = 1.1;
 // 5
 nodes.nodeLocX = nodeLocXHost;
 nodes.nodeLocY = nodeLocYHost;
 nodes.nodeLocZ = nodeLocZHost;
 nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
 const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
 const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
 nodes.extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);
 nodes.buildPairsFromBucketsAndExtendedBuckets(numberOfBucketsInXDim,
 numberOfBucketsInYDim);
 }*/

TEST_F(SceNodeTest, addForceFixedNeighborTest) {
	cudaSetDevice(myDeviceId);
	const uint testCellCount = 1;
	const uint testNodePerCell = 4;
	const uint testTotalNodeCount = testCellCount * testNodePerCell;
	SceNodes nodes(testCellCount, testNodePerCell);
	nodes.setCurrentActiveCellCount(testCellCount);
	thrust::host_vector<double> nodeLocXHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocYHost(testTotalNodeCount);
	thrust::host_vector<double> nodeLocZHost(testTotalNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(testTotalNodeCount);
	const double minX = 0.0;
	const double maxX = 3.0 - 1.0e-10;
	const double minY = 0.0;
	const double maxY = 2.0 - 1.0e-10;
//const double minZ = 0.0;
//const double maxZ = 0.0;
	const double bucketSize = 1.0;
	nodeLocXHost[0] = 0.2;
	nodeLocYHost[0] = 0.5;
	nodeLocZHost[0] = 0.0;
	nodeIsActiveHost[0] = 1;
// 0
	nodeLocXHost[1] = 1.2;
	nodeLocYHost[1] = 0.2;
	nodeLocZHost[1] = 0.0;
	nodeIsActiveHost[1] = 1;
// 1
	nodeLocXHost[2] = 1.3;
	nodeLocYHost[2] = 0.5;
	nodeLocZHost[2] = 0.0;
	nodeIsActiveHost[2] = 1;
// 1
	nodeLocXHost[3] = 2.7;
	nodeLocYHost[3] = 1.1;
	nodeLocZHost[3] = 0.0;
	nodeIsActiveHost[3] = 1;
// 5
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	//nodes.buildBuckets2D(minX, maxX, minY, maxY, bucketSize);
	nodes.calculateAndApplySceForces(minX, maxX, minY, maxY, bucketSize);
	//const int numberOfBucketsInXDim = (maxX - minX) / bucketSize + 1;
	//const int numberOfBucketsInYDim = (maxY - minY) / bucketSize + 1;
	//nodes.extendBuckets2D(numberOfBucketsInXDim, numberOfBucketsInYDim);
	//std::cout << "before applying forces:" << std::endl;
	//thrust::host_vector<double> nodeVelXFromGPU_init = nodes.nodeVelX;
	//thrust::host_vector<double> nodeVelYFromGPU_init = nodes.nodeVelY;
	//thrust::host_vector<double> nodeVelZFromGPU_init = nodes.nodeVelZ;
	//for (uint i = 0; i < nodeVelXFromGPU_init.size(); i++) {
	//	std::cout << nodeVelXFromGPU_init[i] << ", " << nodeVelYFromGPU_init[i]
	//			<< ", " << nodeVelZFromGPU_init[i] << std::endl;
	//}
	//std::cout << std::endl;
	//nodes.applySceForces(numberOfBucketsInXDim, numberOfBucketsInYDim);
	//thrust::host_vector<uint> bucketsKeysFromGPU = nodes.bucketKeys;
	//thrust::host_vector<uint> bucketsValuesFromGPU = nodes.bucketValues;
	//std::cout << "printing key-value pairs:" << std::endl;
	//for (uint i = 0; i < bucketsKeysFromGPU.size(); i++) {
	//	std::cout << "Key :" << bucketsKeysFromGPU[i] << ", value: "
	//			<< bucketsValuesFromGPU[i] << std::endl;
	//}
	thrust::host_vector<double> nodeVelXFromGPU = nodes.nodeVelX;
	thrust::host_vector<double> nodeVelYFromGPU = nodes.nodeVelY;
	thrust::host_vector<double> nodeVelZFromGPU = nodes.nodeVelZ;
	for (uint i = 0; i < nodeVelXFromGPU.size(); i++) {
//std::cout << nodeVelXFromGPU[i] << ", " << nodeVelYFromGPU[i] << ", "
//		<< nodeVelZFromGPU[i] << std::endl;
	}

	vector<double> xPoss(testTotalNodeCount, 0.0);
	vector<double> yPoss(testTotalNodeCount, 0.0);
	vector<double> zPoss(testTotalNodeCount, 0.0);
	vector<double> xVels(testTotalNodeCount, 0.0);
	vector<double> yVels(testTotalNodeCount, 0.0);
	vector<double> zVels(testTotalNodeCount, 0.0);
	vector<bool> isActive(testTotalNodeCount, 0);
	for (uint i = 0; i < testTotalNodeCount; i++) {
		xPoss[i] = nodeLocXHost[i];
		yPoss[i] = nodeLocYHost[i];
		zPoss[i] = nodeLocZHost[i];
		isActive[i] = nodeIsActiveHost[i];
	}
	vector<double> paraSetIntra(4, 0.0);
	for (uint i = 0; i < 4; i++) {
		paraSetIntra[i] = sceIntraParaCPU[i];
	}
	computeResultFromCPUAllIntra2D(xPoss, yPoss, zPoss, xVels, yVels, zVels,
			isActive, paraSetIntra, bucketSize, minX, maxX, minY, maxY);
	for (uint i = 0; i < nodeVelXFromGPU.size(); i++) {
		EXPECT_NEAR(xVels[i], nodeVelXFromGPU[i], errTol);
		EXPECT_NEAR(yVels[i], nodeVelYFromGPU[i], errTol);
		EXPECT_NEAR(zVels[i], nodeVelZFromGPU[i], errTol);
	}

	//std::cout << std::endl;
}

TEST_F(SceNodeTest, addForceRandomTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	int currentActiveCellCount = maxCellCount-3;
	nodes.setCurrentActiveCellCount(currentActiveCellCount);
	thrust::host_vector<double> nodeLocXHost(maxNodeCount);
	thrust::host_vector<double> nodeLocYHost(maxNodeCount);
	thrust::host_vector<double> nodeLocZHost(maxNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(maxNodeCount);
	const double minX = 0.9;
	const double maxX = 2.4;
	const double minY = 0.8;
	const double maxY = 3.7;
	const double minZ = 0.0;
	const double maxZ = 0.0;
	const double bucketSize = 0.15;
	thrust::counting_iterator<unsigned int> index_sequence_begin(0);
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocXHost.begin(), Prg(minX, maxX));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocYHost.begin(), Prg(minY, maxY));
	thrust::transform(index_sequence_begin, index_sequence_begin + maxNodeCount,
			nodeLocZHost.begin(), Prg(minZ, maxZ));
	for (uint i = 0; i < maxNodeCount; i++) {
		if (i % maxNodePerCell < maxNodePerCell / 2) {
			nodeIsActiveHost[i] = true;
		} else {
			nodeIsActiveHost[i] = false;
		}
	}
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;
	nodes.calculateAndApplySceForces(minX, maxX, minY, maxY, bucketSize);

	thrust::host_vector<double> nodeVelXFromGPU = nodes.nodeVelX;
	thrust::host_vector<double> nodeVelYFromGPU = nodes.nodeVelY;
	thrust::host_vector<double> nodeVelZFromGPU = nodes.nodeVelZ;

	vector<double> xPoss(maxNodeCount, 0.0);
	vector<double> yPoss(maxNodeCount, 0.0);
	vector<double> zPoss(maxNodeCount, 0.0);
	vector<double> xVels(maxNodeCount, 0.0);
	vector<double> yVels(maxNodeCount, 0.0);
	vector<double> zVels(maxNodeCount, 0.0);
	vector<bool> isActive(maxNodeCount, 0.0);

	for (uint i = 0; i < maxNodeCount; i++) {
		xPoss[i] = nodeLocXHost[i];
		yPoss[i] = nodeLocYHost[i];
		zPoss[i] = nodeLocZHost[i];
		isActive[i] = nodeIsActiveHost[i];
	}

	vector<double> paraSetIntra(4, 0.0);
	for (uint i = 0; i < 4; i++) {
		paraSetIntra[i] = sceIntraParaCPU[i];
	}
	vector<double> paraSetInter(5, 0.0);
	for (uint i = 0; i < 5; i++) {
		paraSetInter[i] = sceInterParaCPU[i];
	}

	computeResultFromCPUAllIntraAndInter2D(xPoss, yPoss, zPoss, xVels, yVels,
			zVels, isActive, paraSetIntra, paraSetInter, bucketSize,
			currentActiveCellCount, maxNodePerCell, minX, maxX, minY, maxY);

	for (uint i = 0; i < currentActiveCellCount * maxNodePerCell; i++) {
		//std::cout << "xVel expected = " << xVels[i] << " xVel from GPU = "
		//		<< nodeVelXFromGPU[i] << std::endl;
		EXPECT_NEAR(xVels[i], nodeVelXFromGPU[i], errTol);
		EXPECT_NEAR(yVels[i], nodeVelYFromGPU[i], errTol);
		EXPECT_NEAR(zVels[i], nodeVelZFromGPU[i], errTol);
	}

}
