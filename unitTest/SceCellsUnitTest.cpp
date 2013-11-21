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
