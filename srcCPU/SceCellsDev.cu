/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/copy.h>

/*
 static const int WORK_SIZE = 256;

 __host__ __device__ unsigned int bitreverse(unsigned int number)
 {
 number = ((0xf0f0f0f0 & number) >> 4) | ((0x0f0f0f0f & number) << 4);
 number = ((0xcccccccc & number) >> 2) | ((0x33333333 & number) << 2);
 number = ((0xaaaaaaaa & number) >> 1) | ((0x55555555 & number) << 1);
 return number;
 }

 struct bitreverse_functor
 {
 __host__ __device__ unsigned int operator()(const unsigned int &x)
 {
 return bitreverse(x);
 }
 };


 int main()
 {
 thrust::host_vector<unsigned int> idata(WORK_SIZE);
 thrust::host_vector<unsigned int> odata;
 thrust::device_vector<unsigned int> dv;
 int i;

 for (i = 0; i < WORK_SIZE; i++) {
 idata[i] = i;
 }
 dv = idata;

 thrust::transform(dv.begin(), dv.end(), dv.begin(), bitreverse_functor());

 odata = dv;
 for (int i = 0; i < WORK_SIZE; i++) {
 std::cout << "Input value: " << idata[i] << ", output value: "
 << odata[i] << std::endl;
 }

 return 0;
 }
 */

#include <iostream>
#include "ConfigParser.h"
#include "SceCellModified.h"
#include "SceNode.h"
#include "time.h"
#include "SimulationDomain.h"
using namespace std;

GlobalConfigVars globalConfigVars;

void generateStringInputs(std::string &loadMeshInput,
		std::string &animationInput, std::string &animationFolder,
		std::vector<std::string> &boundaryMeshFileNames) {
	std::string meshLocation =
			globalConfigVars.getConfigValue("MeshLocation").toString();
	std::string meshName =
			globalConfigVars.getConfigValue("MeshName").toString();
	std::string meshExtention =
			globalConfigVars.getConfigValue("MeshExtention").toString();
	loadMeshInput = meshLocation + meshName + meshExtention;

	animationFolder =
			globalConfigVars.getConfigValue("AnimationFolder").toString();
	animationInput = animationFolder
			+ globalConfigVars.getConfigValue("AnimationName").toString();

	std::string boundaryMeshLocation = globalConfigVars.getConfigValue(
			"BoundaryMeshLocation").toString();
	std::string boundaryMeshName = globalConfigVars.getConfigValue(
			"BoundaryMeshName").toString();
	std::string boundaryMeshExtention = globalConfigVars.getConfigValue(
			"BoundaryMeshExtention").toString();
	std::string boundaryMeshInput = boundaryMeshLocation + boundaryMeshName
			+ boundaryMeshExtention;
	boundaryMeshFileNames.push_back(boundaryMeshInput);
}

int main() {
	cudaSetDevice(2);
	srand(time(NULL));
	ConfigParser parser;
	std::string configFileName = "sceCell.config";
	globalConfigVars = parser.parseConfigFile(configFileName);
	std::string loadMeshInput;
	std::string animationInput;
	std::vector<std::string> boundaryMeshFileNames;
	std::string animationFolder;
	generateStringInputs(loadMeshInput, animationInput, animationFolder,
			boundaryMeshFileNames);

	double SimulationTotalTime = globalConfigVars.getConfigValue(
			"SimulationTotalTime").toDouble();
	double SimulationTimeStep = globalConfigVars.getConfigValue(
			"SimulationTimeStep").toDouble();
	int TotalNumOfOutputFrames = globalConfigVars.getConfigValue(
			"TotalNumOfOutputFrames").toInt();

	const double simulationTime = SimulationTotalTime;
	const double dt = SimulationTimeStep;
	const int numOfTimeSteps = simulationTime / dt;
	const int totalNumOfOutputFrame = TotalNumOfOutputFrames;
	const int outputAnimationAuxVarible = numOfTimeSteps
			/ totalNumOfOutputFrame;

	const double cellLifeSpan = simulationTime / 5.0;
	double initLength = 1.0;
	double finalLength = 1.8;
	BeakSce::SceCell sceCell(initLength, finalLength, cellLifeSpan);
	sceCell.loadMesh(loadMeshInput);
	const int NNum = sceCell.getNodeCount() * 2;
	const int growthAuxVar = numOfTimeSteps / NNum;
	vector<vector<CVector> > divideBuffer;

	//CVector startPoint = CVector(0, 0, 0);
	//double distance = 0.06;
	//double minDistanceToOtherNode = 0.05;
	//double bondCutoffDist = 0.40;

	CVector elongateDirection = CVector(1.0, 0.0, 0.0);
	/*
	 for (int i = 1; i <= numOfTimeSteps; i++) {
	 //if (i % (numOfTimeSteps / 2) == 0) {
	 //	sceCell.divide(elongateDirection);
	 //}
	 //if (i % growthAuxVar == 0) {
	 //	sceCell.addOneNode();
	 //}
	 sceCell.runGrowthLogic(dt);
	 sceCell.addNearbyPointsToLinkForAllPoints();

	 if (i % outputAnimationAuxVarible == 0) {
	 sceCell.outputVtkFiles(animationInput, i);
	 }
	 sceCell.clearVelocity();
	 sceCell.addAllVelFromLinks();
	 sceCell.addVelFromElongation(elongateDirection,
	 sceCell.getTargetLengthNow());
	 sceCell.move(dt);
	 sceCell.runDivideLogic(elongateDirection);

	 //sceCell.printMaxVel();
	 }
	 */
	vector<CVector> bdryLocations1;
	vector<CVector> bdryLocations2;
	vector<CVector> bdryLocations3;
	CVector startPt = CVector(-0.8, 0.8, 0);
	CVector endPt = CVector(-0.8, -0.8, 0);
	CVector pt1 = CVector(5, 0.8, 0);
	CVector pt2 = CVector(5, -0.8, 0);
	double increment = 0.1;
	CVector tmpPt = startPt;
	while (tmpPt.y > endPt.y) {
		bdryLocations1.push_back(tmpPt);
		tmpPt.y = tmpPt.y - increment;
	}
	tmpPt = startPt;
	while (tmpPt.x < pt1.x) {
		bdryLocations2.push_back(tmpPt);
		tmpPt.x = tmpPt.x + increment;
	}
	tmpPt = endPt;
	while (tmpPt.x < pt2.x) {
		bdryLocations3.push_back(tmpPt);
		tmpPt.x = tmpPt.x + increment;
	}

	vector<CVector> nodeLocations = sceCell.getNodeLoctions();
	BeakSce::SimulationDomain testDomain;
	testDomain.insertNewBoundary(bdryLocations1, 0);
	testDomain.insertNewBoundary(bdryLocations2, 1);
	testDomain.insertNewBoundary(bdryLocations3, 2);
	testDomain.insertNewCell(nodeLocations, 3);

	thrust::host_vector<SceNode> hostSceNodes(1000);
	thrust::generate(hostSceNodes.begin(),hostSceNodes.end(),rand);
	thrust::device_vector<SceNode> deviceSceNodes = hostSceNodes;

	/*
	for (int i = 1; i <= numOfTimeSteps; i++) {
		//if (i % (numOfTimeSteps / 2) == 0) {
		//	sceCell.divide(elongateDirection);
		//}
		//if (i % growthAuxVar == 0) {
		//	sceCell.addOneNode();
		//}
		//sceCell.moveAndExecuteAllLogic(dt, elongateDirection, divideBuffer);
		testDomain.inefficientlyBuildInterCellLinks();

		if (i % outputAnimationAuxVarible == 0) {
			//sceCell.outputVtkFiles(animationInput, i);
			//testDomain.outputVtkFiles(animationInput, i);
			testDomain.outputVtkFilesWithColor(animationInput, i);
		}
		testDomain.allCellsMoveAndRecordDivisionInfo(dt, elongateDirection);
		testDomain.processDivisionInfoAndAddNewCells();
		//sceCell.printMaxVel();
	}
	*/

//cout << "!!!Hello World!!!" << endl; // prints !!!Hello World!!!
	return 0;
}

