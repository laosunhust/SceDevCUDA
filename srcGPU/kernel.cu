
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>


// sequence :   U0, V0, k1, k2
__constant__ double sceInterPara[4];
__constant__ double sceIntraPara[4];
double sceInterParaCPU[4];
double sceIntraParaCPU[4];


__device__ __host__ 
	double computeDist(double &xPos, double &yPos, double &zPos, double &xPos2, double &yPos2, double &zPos2){
		return sqrt((xPos-xPos2)*(xPos-xPos2)+(yPos-yPos2)*(yPos-yPos2)+(zPos-zPos2)*(zPos-zPos2));
}

__device__ __host__ 
	void calculateAndAddECMForce(double &xPos, double &yPos, double &zPos, double &xPos2, double &yPos2, double &zPos2, double &xRes,double &yRes,double &zRes){


}

__device__ __host__ 
	void calculateAndAddInterForce(double &xPos, double &yPos, double &zPos, double &xPos2, double &yPos2, double &zPos2, double &xRes,double &yRes,double &zRes){
#ifdef __CUDA_ARCH__
		double linkLength = computeDist(xPos,yPos,zPos,xPos2,yPos2,zPos2);
		double forceValue = -sceInterPara[0] / sceInterPara[2] * exp(-linkLength / sceInterPara[2])
			+ sceInterPara[1] / sceInterPara[3] * exp(-linkLength / sceInterPara[3]);
		if (forceValue > 0) {
			forceValue = 0;
		}
		xRes = xRes + forceValue * (xPos2 - xPos)/linkLength;
		yRes = yRes + forceValue * (yPos2 - yPos)/linkLength;
		zRes = zRes + forceValue * (zPos2 - zPos)/linkLength;
#else

#endif
}

__device__ __host__ 
	void calculateAndAddIntraForce(double &xPos, double &yPos, double &zPos, double &xPos2, double &yPos2, double &zPos2, double &xRes,double &yRes,double &zRes){
#ifdef __CUDA_ARCH__
		double linkLength = computeDist(xPos,yPos,zPos,xPos2,yPos2,zPos2);
		double forceValue = -sceIntraPara[0] / sceIntraPara[2] * exp(-linkLength / sceIntraPara[2])
			+ sceIntraPara[1] / sceIntraPara[3] * exp(-linkLength / sceIntraPara[3]);
		xRes = xRes + forceValue * (xPos2 - xPos)/linkLength;
		yRes = yRes + forceValue * (yPos2 - yPos)/linkLength;
		zRes = zRes + forceValue * (zPos2 - zPos)/linkLength;
#else

#endif
}


struct AddSceForce: public thrust::unary_function<Tuuuddd, CVec3> {
	uint* _extendedValuessAddress;
	double* _nodeLocXAddress;
	double* _nodeLocYAddress;
	double* _nodeLocZAddress;
	__host__ __device__ AddSceForce(uint* valueAddress, double* nodeLocXAddress,
		double* nodeLocYAddress, double* nodeLocZAddress) :
	_extendedKeysAddress(valueAddress), _nodeLocXAddress(
		nodeLocXAddress), _nodeLocYAddress(nodeLocYAddress), _nodeLocZAddress(
		nodeLocZAddress) {
	}
	__host__ __device__ CVec3 operator()(const Tuuuddd &u3d3) const {
		uint begin = thrust::get < 0 > (u3d3);
		uint end = thrust::get < 1 > (u3d3);
		uint myValue = thrust::get <2>(u3d3);
		double xPos = thrust::get<3>(u3d3);
		double yPos = thrust::get<4>(u3d3);
		double zPos = thrust::get<5>(u3d3);
		double xRes = 0.0;
		double yRes = 0.0;
		double zRes = 0.0;
		for(uint i=begin;i<end;i++){
			if(bothNodesCellNode(myValue,_extendedValuessAddress[i]){
				if(isSameCell(myValue, _extendedValuessAddress[i]){
					calculateAndAddIntraForce(xPos, yPos, zPos, _nodeLocXAddress[i], _nodeLocYAddress[i],_nodeLocYAddress[i],xRes,yRes,zRes);
				}else{
					calculateAndAddInterForce(xPos, yPos, zPos, _nodeLocXAddress[i], _nodeLocYAddress[i],_nodeLocYAddress[i],xRes,yRes,zRes);
				}
			}
			else{
				calculateAndAddECMForce(xPos, yPos, zPos, _nodeLocXAddress[i], _nodeLocYAddress[i],_nodeLocYAddress[i],xRes,yRes,zRes);
			}
		}
		return thrust::make_tuple(xRes,yRes,zRes);
	}
};

int main() {
	static const double U0 =
			globalConfigVars.getConfigValue("InterCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_U0_DivFactor").toDouble();
	static const double V0 =
			globalConfigVars.getConfigValue("InterCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_V0_DivFactor").toDouble();
	static const double k1 =
			globalConfigVars.getConfigValue("InterCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k1_DivFactor").toDouble();
	static const double k2 =
			globalConfigVars.getConfigValue("InterCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k2_DivFactor").toDouble();
	sceInterParaCPU[0] = U0;
	sceInterParaCPU[1] = V0;
	sceInterParaCPU[2] = k1;
	sceInterParaCPU[3] = k2;

	static const double U0_Intra =
			globalConfigVars.getConfigValue("IntraCell_U0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_U0_DivFactor").toDouble();
	static const double V0_Intra =
			globalConfigVars.getConfigValue("IntraCell_V0_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_V0_DivFactor").toDouble();
	static const double k1_Intra =
			globalConfigVars.getConfigValue("IntraCell_k1_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k1_DivFactor").toDouble();
	static const double k2_Intra =
			globalConfigVars.getConfigValue("IntraCell_k2_Original").toDouble()
					/ globalConfigVars.getConfigValue("InterCell_k2_DivFactor").toDouble();
	sceIntraParaCPU[0] = U0_Intra;
	sceIntraParaCPU[1] = V0_Intra;
	sceIntraParaCPU[2] = k1_Intra;
	sceIntraParaCPU[3] = k2_Intra;

	cudaMemcpyToSymbol(sceInterPara, sceInterParaCPU, 4 * sizeof(double), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(sceIntraPara, sceIntraParaCPU, 4 * sizeof(double), 0, cudaMemcpyHostToDevice);
}