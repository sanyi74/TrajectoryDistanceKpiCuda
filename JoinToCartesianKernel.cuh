#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Common.h"

// #define DOUBLE_PRECISION

#ifdef DOUBLE_PRECISION
	#define REAL double
#else
	#define REAL float
#endif

__device__ __host__ REAL reciprocalSqrt(REAL number) {
#if defined(__CUDA_ARCH__)
	#ifndef DOUBLE_PRECISION
		return __frsqrt_rn(number);
	#else
		return rsqrt(number);
	#endif
#else
	#ifndef DOUBLE_PRECISION
		float xhalf = 0.5f * number;
		int i = *(int*)&number;
		i = 0x5f3759df - (i >> 1);
		number = *(float*)&i;
		number = number * (1.5f - (xhalf * number * number));
		return number;
	#else
		return 1.0 / sqrt(number);
	#endif
#endif
}

__device__ __host__ void SinCos(REAL x, REAL* sptr, REAL* cptr) {
#if defined(__CUDA_ARCH__)
	#ifndef DOUBLE_PRECISION
		__sincosf(x, sptr, cptr);
	#else
		sincos(x, sptr, cptr);
	#endif
#else
		sincos(x, sptr, cptr);
#endif
}

__device__ __host__ REAL Sqrt(REAL x) {
#if defined(__CUDA_ARCH__)
#ifndef DOUBLE_PRECISION
	return __fsqrt_rn(x);
#else
	return sqrt(x);
#endif
#else
	return sqrt(x);
#endif
}

void inline forward(const float* q, REAL* T, float d1, float a2, float a3, float d4, float d5, float d6) {
	// Source: https://github.com/ros-industrial/universal_robot/blob/kinetic-devel/ur_kinematics/src/ur_kin.cpp
	REAL s1, c1; SinCos(*q, &s1, &c1); q++;
	REAL q23 = *q, q234 = *q; REAL s2, c2;  SinCos(*q, &s2, &c2); q++;
	REAL s3, c3; SinCos(*q, &s3, &c3); q23 += *q; q234 += *q; q++;
	REAL s4, c4;  SinCos(*q, &s4, &c4);	q234 += *q; q++;
	REAL s5, c5;  SinCos(*q, &s5, &c5);	q++;
	REAL s6, c6;  SinCos(*q, &s6, &c6);
	REAL s23, c23;  SinCos(q23, &s23, &c23);
	REAL s234, c234;  SinCos(q234, &s234, &c234);

	REAL A = (s1 * s5 + c234 * c1 * c5);
	REAL B = (c1 * s5 - c234 * c5 * s1);

	*T = c234 * c1 * s5 - c5 * s1; T++;
	*T = c6 * A - s234 * c1 * s6; T++;
	*T = -s6 * A - s234 * c1 * c6; T++;
	*T = d6 * c234 * c1 * s5 - a3 * c23 * c1 - a2 * c1 * c2 - d6 * c5 * s1 - d5 * s234 * c1 - d4 * s1; T++;
	*T = c1 * c5 + c234 * s1 * s5; T++;
	*T = -c6 * B - s234 * s1 * s6; T++;
	*T = s6 * B - s234 * c6 * s1; T++;
	*T = d6 * (c1 * c5 + c234 * s1 * s5) + d4 * c1 - a3 * c23 * s1 - a2 * c2 * s1 - d5 * s234 * s1; T++;
	*T = -s234 * s5; T++;
	*T = -c234 * s6 - s234 * c5 * c6; T++;
	*T = s234 * c5 * s6 - c234 * c6; T++;
	*T = d1 + a3 * s23 + a2 * s2 - d5 * (c23 * c4 - s23 * s4) - d6 * s5 * (c23 * s4 + s23 * c4);
}

__device__ __host__ void jointToCartesian(unsigned int index, DhParameter* dhParameters, Pose* poses, JointPoint* jointPoints, CartesianPoint* cartesianPoints) {

	unsigned int dhIndex = jointPoints[index].dhIndex;
	REAL d1 = dhParameters[dhIndex].d1;
	REAL a2 = dhParameters[dhIndex].a2;
	REAL a3 = dhParameters[dhIndex].a3;
	REAL d4 = dhParameters[dhIndex].d4;
	REAL d5 = dhParameters[dhIndex].d5;
	REAL d6 = dhParameters[dhIndex].d6;
	
	REAL T[12];
	forward(jointPoints[index].q, T, d1, a2, a3, d4, d5, d6);

	REAL TZ[8], sinRotZ, cosRotZ;
	SinCos(dhParameters[dhIndex].rotZ, &sinRotZ, &cosRotZ);
	TZ[0] = T[0] * cosRotZ - T[4] * sinRotZ;
	TZ[1] = T[1] * cosRotZ - T[5] * sinRotZ;
	TZ[2] = T[2] * cosRotZ - T[6] * sinRotZ;
	TZ[3] = T[3] * cosRotZ - T[7] * sinRotZ;
	TZ[4] = T[0] * sinRotZ + T[4] * cosRotZ;
	TZ[5] = T[1] * sinRotZ + T[5] * cosRotZ;
	TZ[6] = T[2] * sinRotZ + T[6] * cosRotZ;
	TZ[7] = T[3] * sinRotZ + T[7] * cosRotZ;
	T[0] = TZ[0]; T[1] = TZ[1]; T[2] = TZ[2]; T[3] = TZ[3]; T[4] = TZ[4]; T[5] = TZ[5]; T[6] = TZ[6]; T[7] = TZ[7];

	REAL quaternion[4], t;
	if (T[0] + T[4 + 1] + T[2 * 4 + 2] > (REAL)(0.0)) {
		t = T[0] + T[4 + 1] + T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[3] = t;
		quaternion[2] = T[1] - T[4];
		quaternion[1] = T[2 * 4] - T[2];
		quaternion[0] = T[4 + 2] - T[2 * 4 + 1];
	}
	else if (T[0] > T[4 + 1] && T[0] > T[2 * 4 + 2]) {
		t = T[0] - T[4 + 1] - T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[0] = t;
		quaternion[1] = T[1] + T[4];
		quaternion[2] = T[2 * 4] + T[2];
		quaternion[3] = T[4 + 2] - T[2 * 4 + 1];
	}
	else if (T[4 + 1] > T[2 * 4 + 2]) {
		t = -T[0] + T[4 + 1] - T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[1] = t;
		quaternion[0] = T[1] + T[4];
		quaternion[3] = T[2 * 4] - T[2];
		quaternion[2] = T[4 + 2] + T[2 * 4 + 1];
	}
	else {
		t = -T[0] - T[4 + 1] + T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[2] = t;
		quaternion[3] = T[1] - T[4];
		quaternion[0] = T[2 * 4] + T[2];
		quaternion[1] = T[4 + 2] + T[2 * 4 + 1];
	}
	t = reciprocalSqrt(t) * (REAL)(0.5);
	quaternion[0] *= t;	quaternion[1] *= t;	quaternion[2] *= t;	quaternion[3] *= t;

	unsigned int poseIndex = jointPoints[index].poseIndex;
	cartesianPoints[index].xyz[0] = (float)(T[3] + poses[poseIndex].xyz[0]); 
	cartesianPoints[index].xyz[1] = (float)(T[7] + poses[poseIndex].xyz[1]); 
	cartesianPoints[index].xyz[2] = (float)(T[11] + poses[poseIndex].xyz[2]);

	T[0] = poses[poseIndex].quaternion[0]; T[1] = poses[poseIndex].quaternion[1]; T[2] = poses[poseIndex].quaternion[2]; T[3] = poses[poseIndex].quaternion[3];
	T[4] = quaternion[0] * T[3] + quaternion[1] * T[2] - quaternion[2] * T[1] + quaternion[3] * T[0];
	T[5] = -quaternion[0] * T[2] + quaternion[1] * T[3] + quaternion[2] * T[0] + quaternion[3] * T[1];
	T[6] = quaternion[0] * T[1] - quaternion[1] * T[0] + quaternion[2] * T[3] + quaternion[3] * T[2];
	T[7] = -quaternion[0] * T[0] - quaternion[1] * T[1] - quaternion[2] * T[2] + quaternion[3] * T[3];

	T[8] = Sqrt(T[4] * T[4] + T[5] * T[5] + T[6] * T[6] + T[7] * T[7]);
	if (T[4] < (REAL)(0.0)) T[8] *= (REAL)(-1.0);
	T[4] /= T[8]; T[5] /= T[8];	T[6] /= T[8]; T[7] /= T[8];

	cartesianPoints[index].quaternion[0] = (float)(T[4]); 
	cartesianPoints[index].quaternion[1] = (float)(T[5]); 
	cartesianPoints[index].quaternion[2] = (float)(T[6]); 
	cartesianPoints[index].quaternion[3] = (float)(T[7]);
}

__global__ void jointToCartesianKernel(unsigned int nElements, DhParameter* dhParameters, Pose* poses, JointPoint* jointPoints, CartesianPoint* cartesianPoints) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < nElements) jointToCartesian(index, dhParameters, poses, jointPoints, cartesianPoints);
}

void jointToCartesianCpu(unsigned int index, DhParameter* dhParameters, Pose* poses, JointPoint* jointPoints, CartesianPoint* cartesianPoints) {
	jointToCartesian(index, dhParameters, poses, jointPoints, cartesianPoints);
}

void evaluateJointToCartesian(unsigned int nSample, unsigned int nPoints, unsigned int nKernel) {

	JointPoint* host_JointPoints;
	CartesianPoint* host_CartesianPoints;
	cudaHostAlloc((void**)&host_JointPoints, nPoints * sizeof(JointPoint), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_CartesianPoints, nPoints * sizeof(CartesianPoint), cudaHostAllocDefault);

	DhParameter* dev_DhParameters;
	Pose* dev_Poses;
	JointPoint* dev_JointPoints;
	CartesianPoint* dev_CartesianPoints;
	cudaMalloc((void**)&dev_DhParameters, 16 * sizeof(DhParameter));
	cudaMalloc((void**)&dev_Poses, 16 * sizeof(Pose));
	cudaMalloc((void**)&dev_JointPoints, nPoints * sizeof(JointPoint));
	cudaMalloc((void**)&dev_CartesianPoints, nPoints * sizeof(CartesianPoint));

	DhParameter host_DhParameters[16];
	Pose host_Poses[16];
	setUr5DhParameter(0, host_DhParameters);
	host_Poses[0].xyz[0] = host_Poses[0].xyz[1] = host_Poses[0].xyz[2] = 0;
	host_Poses[0].quaternion[0] = host_Poses[0].quaternion[1] = host_Poses[0].quaternion[2] = 0;
	host_Poses[0].quaternion[3] = 1.0;
	cudaMemcpy(dev_DhParameters, host_DhParameters, 16 * sizeof(DhParameter), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Poses, host_Poses, 16 * sizeof(Pose), cudaMemcpyHostToDevice);

	double cpuTime = 0;
	double gpuTime = 0;
	for (unsigned int iter = 0; iter < nSample; iter++) {
		std::cout << "*";
		auto start = std::chrono::system_clock::now();
		{ // CPU

			const unsigned int nThreads = 16;
			std::thread* t[nThreads];
			for (int i = 0; i < nThreads; i++) {
				t[i] = new std::thread([iter, i, host_JointPoints, nThreads, nPoints]() {
					uint32_t rseed = 2008 * (iter + 7) + 1974 * i;
					rseed = rseed * 214013 + 2531011;
					for (auto index = i * (nPoints / nThreads); index < (i + 1) * (nPoints / nThreads); index++) {
						for (auto j = 0; j < 6; j++) {
							host_JointPoints[index].q[j] = 2.0f * (float)M_PI * (float)rseed / (float)UINT_MAX - (float)M_PI;
							rseed = rseed * 214013 + 2531011;
						}
						host_JointPoints[index].dhIndex = host_JointPoints[index].poseIndex = 0;
					}
					});
			}
			for (int i = 0; i < nThreads; i++) {
				t[i]->join();
				delete t[i];
			}
		}

		start = std::chrono::system_clock::now();
		{ // GPU
			cudaMemcpy(dev_JointPoints, host_JointPoints, nPoints * sizeof(JointPoint), cudaMemcpyHostToDevice);
			unsigned int nBlocks = nPoints / 128;
			if (nPoints % 128 != 0) nBlocks++;
			for (int i = 0; i < nKernel; i++) jointToCartesianKernel << <nBlocks, 128 >> > (nPoints, dev_DhParameters, dev_Poses, dev_JointPoints, dev_CartesianPoints);
			cudaMemcpy(host_CartesianPoints, dev_CartesianPoints, nPoints * sizeof(CartesianPoint), cudaMemcpyDeviceToHost);
		}
		gpuTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

		start = std::chrono::system_clock::now();
		{ // CPU
			const unsigned int nThreads = 16;
			std::thread* t[nThreads];
			for (int i = 0; i < nThreads; i++) {
				t[i] = new std::thread([i, &host_DhParameters, &host_Poses, &host_JointPoints, &host_CartesianPoints, nThreads, nPoints]() {
					for (auto index = i * (nPoints / nThreads); index < (i + 1) * (nPoints / nThreads); index++)
						jointToCartesianCpu(index, host_DhParameters, host_Poses, host_JointPoints, host_CartesianPoints);
					});
			}
			for (int i = 0; i < nThreads; i++) {
				t[i]->join();
				delete t[i];
			}
		}
		cpuTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();
	}
	cpuTime = cpuTime / 1000000.0 / nSample;
	gpuTime = gpuTime / 1000000.0 / nSample;

	std::cout << "\r";
	std::cout << nPoints << " " << nKernel << " " << cpuTime << " sec " << gpuTime << " sec " << (gpuTime * 1000000.0 / nPoints) << " us/point " << (nPoints * (sizeof(JointPoint) + sizeof(CartesianPoint))) << " byte        " << std::endl;

	cudaFreeHost(host_JointPoints);
	cudaFreeHost(host_CartesianPoints);
	cudaFree(dev_JointPoints);
	cudaFree(dev_CartesianPoints);
}

int main() {

	if (cudaSetDevice(0) != cudaSuccess || cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) != cudaSuccess) {
		std::cout << "cudaSetDevice or cudaDeviceScheduleBlockingSync failed!" << std::endl;
		return -1;
	}

	// evaluateJointToCartesian(64, 32, 1);

	unsigned int nSamples = 4;
	for (double i = 64 * 1024 * 1024; i >= 1000; i /= 1.414213562373095) {
		evaluateJointToCartesian(nSamples, i, 0);
		for (unsigned int k = 1; k <= 256; k *= 2) {
			evaluateJointToCartesian(nSamples, i, k);
		}
	}

	while (true);

	return 1;
}