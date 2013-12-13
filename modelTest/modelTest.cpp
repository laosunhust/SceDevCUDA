#include <iostream>
#include "SimulationDomainGPU.h"
using namespace std;

const int myDeviceID = 2;
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

/**
 * This method is highly not robust and is for test purpose only.
 * This method assumes that the size of the computation domain
 * is 50 X 50 and left bottom point is (0,0). it initializes cells of a vertical line.
 * starting from position (5,5) with step of 3.0
 */
void generateCellInitInfo(std::string meshInput, uint numberOfInitCells,
		std::vector<double> &initCellNodePosX,
		std::vector<double> &initCellNodePosY, std::vector<double> &centerPosX,
		std::vector<double> &centerPosY) {
	centerPosX.resize(numberOfInitCells);
	centerPosY.resize(numberOfInitCells);

	fstream fs;
	uint NNum, LNum, ENum;
	uint i;
	double tmp;
	std::cout << "mesh input is :" << meshInput << std::endl;
	fs.open(meshInput.c_str(), ios::in);
	if (!fs.is_open()) {
		std::string errString =
				"Unable to load mesh in string input mode, meshname: "
						+ meshInput
						+ " ,possible reason is the file is not located in the project folder \n";
		std::cout << errString;
	}
	fs >> NNum >> LNum >> ENum;
	cout << "NNum = " << NNum << std::endl;

	initCellNodePosX.resize(NNum);
	initCellNodePosY.resize(NNum);

	for (i = 0; i < NNum; i++) {
		fs >> initCellNodePosX[i] >> initCellNodePosY[i] >> tmp;
		initCellNodePosX[i] = initCellNodePosX[i] / 20.0;
		initCellNodePosY[i] = initCellNodePosY[i] / 10.0;
	}
	fs.close();
	double startCenterPosX = 10.0;
	double startCenterPosY = 5.0;
	double currentCenterPosX = startCenterPosX;
	double currentCenterPosY = startCenterPosY;
	double stepIncrease = 0.99;
	for (i = 0; i < numberOfInitCells; i++) {
		centerPosX[i] = currentCenterPosX;
		centerPosY[i] = currentCenterPosY;
		currentCenterPosY = currentCenterPosY + stepIncrease;
	}
}

int main() {
	cudaSetDevice(myDeviceID);
	ConfigParser parser;
	std::string configFileName = "sceCell.config";
	globalConfigVars = parser.parseConfigFile(configFileName);

	double SimulationTotalTime = globalConfigVars.getConfigValue(
			"SimulationTotalTime").toDouble();
	double SimulationTimeStep = globalConfigVars.getConfigValue(
			"SimulationTimeStep").toDouble();
	int TotalNumOfOutputFrames = globalConfigVars.getConfigValue(
			"TotalNumOfOutputFrames").toInt();

	std::string loadMeshInput;
	std::string animationInput;
	std::vector < std::string > boundaryMeshFileNames;
	std::string animationFolder;
	generateStringInputs(loadMeshInput, animationInput, animationFolder,
			boundaryMeshFileNames);

	const double simulationTime = SimulationTotalTime;
	const double dt = SimulationTimeStep;
	const int numOfTimeSteps = simulationTime / dt;
	const int totalNumOfOutputFrame = TotalNumOfOutputFrames;
	const int outputAnimationAuxVarible = numOfTimeSteps
			/ totalNumOfOutputFrame;

	std::vector<double> initCellNodePosX;
	std::vector<double> initCellNodePosY;
	std::vector<double> centerPosX;
	std::vector<double> centerPosY;

	SimulationDomainGPU simuDomain;
	uint numberOfInitCells = globalConfigVars.getConfigValue(
			"NumberOfInitCells").toInt();
	generateCellInitInfo(loadMeshInput, numberOfInitCells, initCellNodePosX,
			initCellNodePosY, centerPosX, centerPosY);
	std::cout << "finished generate cell info" << std::endl;
	simuDomain.initializeCells(initCellNodePosX, initCellNodePosY, centerPosX,
			centerPosY);

	for (int i = 0; i <= numOfTimeSteps; i++) {
		if (i % outputAnimationAuxVarible == 0) {
			simuDomain.outputVtkFilesWithColor(animationInput, i);
			double lengthDiff = simuDomain.cells.lengthDifference[20];
			double expectedLen = simuDomain.cells.expectedLength[20];
			double growthProgress = simuDomain.cells.growthProgress[20];
			double isGoingToDivide = simuDomain.cells.isDivided[20];
			cout << "length difference: " << lengthDiff << endl;
			cout << "expected length: " << expectedLen << endl;
			cout << "growth progress: " << growthProgress << endl;
			cout << "schedule to divide? " << isGoingToDivide << endl;
		}
		simuDomain.runAllLogic(dt);
		double cellGrowProgress = simuDomain.cells.growthProgress[0];
		bool cellIsCheduledToGrow = simuDomain.cells.isScheduledToGrow[0];
		double lastCheckPoint = simuDomain.cells.lastCheckPoint[0];
		if (cellIsCheduledToGrow) {
			cout << "growth progress = " << cellGrowProgress << endl;
			cout << "Is scheduled to grow?" << cellIsCheduledToGrow << endl;
			cout << "last check point is " << lastCheckPoint << endl;
		}

	}

}
