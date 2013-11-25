#include <iostream>
#include "gtest/gtest.h"
#include "SceCells.h"
#include <algorithm>
using namespace std;

const int myDeviceId = 2;
const uint maxCellCount = 10;
const uint maxNodePerCell = 100;
const uint maxNodeCount = maxCellCount * maxNodePerCell;
const double dt = 0.1;
const double errTol = 1.0e-12;

TEST(SceCellsInitTest, sizeTest) {
	cudaSetDevice(myDeviceId);
	SceNodes nodes(maxCellCount, maxNodePerCell);
	SceCells cells(&nodes);

	EXPECT_EQ(cells.growthProgress.size(), maxCellCount);
	EXPECT_EQ(cells.activeNodeCountOfThisCell.size(), maxCellCount);
	EXPECT_EQ(cells.lastCheckPoint.size(), maxCellCount);
	EXPECT_EQ(cells.isDivided.size(), maxCellCount);
	EXPECT_EQ(cells.centerCoordX.size(), maxCellCount);
	EXPECT_EQ(cells.centerCoordY.size(), maxCellCount);
	EXPECT_EQ(cells.centerCoordZ.size(), maxCellCount);

	EXPECT_EQ(cells.xCoordTmp.size(), maxNodeCount);
	EXPECT_EQ(cells.yCoordTmp.size(), maxNodeCount);
	EXPECT_EQ(cells.zCoordTmp.size(), maxNodeCount);
}

TEST(SceCellsDistriIsActiveInfoTest,fixedTest) {
	const uint maxCellCount = 2;
	const uint initCellCount = 2;
	const uint maxNodePerCell = 4;
	const uint maxECMCount = 2;
	const uint maxNodeInECM = 1;
	const uint maxTotalNodeCount = maxCellCount * maxNodePerCell
			+ maxECMCount * maxNodeInECM;
	SceNodes nodes(maxCellCount, maxNodePerCell, maxECMCount, maxNodeInECM);
	nodes.setCurrentActiveCellCount(initCellCount);
	nodes.setCurrentActiveEcm(maxNodeInECM);

	double nodeXInput[] = { 1.2, 3, 2, 1.5, 0.3, 1.1, 9.9, 0.0, 0.0, 0.0 };
	double nodeYInput[] = { 2.3, 1, 2, 5.6, 0.9, 8.6, 2.3, 0.0, 0.0, 0.0 };
	double nodeZInput[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	bool nodeIsActInput[] = { true, true, true, false, true, true, true, true,
			true, false };

	thrust::host_vector<double> nodeLocXHost(nodeXInput,
			nodeXInput + maxTotalNodeCount);
	thrust::host_vector<double> nodeLocYHost(nodeYInput,
			nodeYInput + maxTotalNodeCount);
	thrust::host_vector<double> nodeLocZHost(nodeZInput,
			nodeZInput + maxTotalNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(nodeIsActInput,
			nodeIsActInput + maxTotalNodeCount);
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;

	SceCells cells(&nodes);
	thrust::host_vector<uint> activeNodeCount(2);
	activeNodeCount[0] = 4;
	activeNodeCount[1] = 3;
	cells.activeNodeCountOfThisCell = activeNodeCount;
	cells.distributeIsActiveInfo();
	bool expectedNodeIsActiveOutput[] = { true, true, true, true, true, true,
			true, false, true, false };
	thrust::host_vector<bool> nodeIsActiveOutputFromGPU = nodes.nodeIsActive;
	for (uint i = 0; i < nodes.getCurrentActiveCellCount() * maxNodePerCell;
			i++) {
		EXPECT_EQ(expectedNodeIsActiveOutput[i], nodeIsActiveOutputFromGPU[i]);
	}
}

TEST(SceCellsCompCelLCenterTest,fixedTest) {
	const uint maxCellCount = 2;
	const uint initCellCount = 2;
	const uint maxNodePerCell = 4;
	const uint maxECMCount = 2;
	const uint maxNodeInECM = 1;
	const uint maxTotalNodeCount = maxCellCount * maxNodePerCell
			+ maxECMCount * maxNodeInECM;
	SceNodes nodes(maxCellCount, maxNodePerCell, maxECMCount, maxNodeInECM);
	nodes.setCurrentActiveCellCount(initCellCount);
	nodes.setCurrentActiveEcm(maxNodeInECM);

	double nodeXInput[] = { 1.2, 3, 2, 1.5, 0.3, 1.1, 9.9, 4.2, 0.0, 0.0 };
	double nodeYInput[] = { 2.3, 1, 2, 5.6, 0.9, 8.6, 2.3, 5.9, 0.0, 0.0 };
	double nodeZInput[] = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	bool nodeIsActInput[] = { true, true, true, false, true, true, true, true,
			true, false };

	thrust::host_vector<double> nodeLocXHost(nodeXInput,
			nodeXInput + maxTotalNodeCount);
	thrust::host_vector<double> nodeLocYHost(nodeYInput,
			nodeYInput + maxTotalNodeCount);
	thrust::host_vector<double> nodeLocZHost(nodeZInput,
			nodeZInput + maxTotalNodeCount);
	thrust::host_vector<bool> nodeIsActiveHost(nodeIsActInput,
			nodeIsActInput + maxTotalNodeCount);
	nodes.nodeLocX = nodeLocXHost;
	nodes.nodeLocY = nodeLocYHost;
	nodes.nodeLocZ = nodeLocZHost;
	nodes.nodeIsActive = nodeIsActiveHost;

	SceCells cells(&nodes);
	thrust::host_vector<uint> activeNodeCount(2);
	activeNodeCount[0] = 4;
	activeNodeCount[1] = 3;
	cells.activeNodeCountOfThisCell = activeNodeCount;
	cells.distributeIsActiveInfo();
	cells.computeCenterPos();
	thrust::host_vector<double> centerXFromGPU = cells.centerCoordX;
	thrust::host_vector<double> centerYFromGPU = cells.centerCoordY;
	double expectedCenterX[] = { 7.7/4, 11.3/3 };
	double expectedCenterY[] = { 10.9/4, 11.8/3 };
	EXPECT_NEAR(centerXFromGPU[0], expectedCenterX[0], errTol);
	EXPECT_NEAR(centerXFromGPU[1], expectedCenterX[1], errTol);
	EXPECT_NEAR(centerYFromGPU[0], expectedCenterY[0], errTol);
	EXPECT_NEAR(centerYFromGPU[1], expectedCenterY[1], errTol);
}
