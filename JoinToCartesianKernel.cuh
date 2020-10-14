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

__device__ __host__ void jointToCartesian(unsigned int index, DhParameter* dhParameters, Pose* poses, JointPoint* jointPoints, CartesianPoint* cartesianPoints) {

	unsigned int dhIndex = jointPoints[index].dhIndex;
	REAL d1 = dhParameters[dhIndex].d1;
	REAL a2 = dhParameters[dhIndex].a2;
	REAL a3 = dhParameters[dhIndex].a3;
	REAL d4 = dhParameters[dhIndex].d4;
	REAL d5 = dhParameters[dhIndex].d5;
	REAL d6 = dhParameters[dhIndex].d6;
	REAL SIN_Z_ROTATION, COS_Z_ROTATION;
	SinCos(dhParameters[dhIndex].rotZ, &SIN_Z_ROTATION, &COS_Z_ROTATION);

	REAL j1, j2, s0, s1, s2, s4, s5, s123, c0, c1, c2, c4, c5, c123;
	j1 = jointPoints[index].q[1];
	j2 = jointPoints[index].q[2];
	SinCos(jointPoints[index].q[0], &s0, &c0);
	SinCos(j1, &s1, &c1);
	SinCos(j2, &s2, &c2);
	SinCos(jointPoints[index].q[4], &s4, &c4);
	SinCos(jointPoints[index].q[5], &s5, &c5);
	SinCos(j1 + j2 + jointPoints[index].q[3], &s123, &c123);

	REAL A1 = c0 * c123 + s0 * s123;
	REAL A2 = c0 * c123 - s0 * s123;
	REAL A3 = s0 * c123 + c0 * s123;
	REAL A4 = s0 * c123 - c0 * s123;
	REAL A5 = c123 * c4 + s123 * s4;
	REAL A6 = c123 * c4 - s123 * s4;

	REAL T[12];
	T[0] = (A2 * s4) / (REAL)(2.0) - c4 * s0 + (A1 * s4) / (REAL)(2.0);
	T[1] = (c5 * (s0 * s4 + (A2 * c4) / (REAL)(2.0) + (A1 * c4) / (REAL)(2.0)) - (s5 * (A3 - A4)) / (REAL)(2.0));
	T[2] = (-(c5 * (A3 - A4)) / (REAL)(2.0) - s5 * (s0 * s4 + (A2 * c4) / (REAL)(2.0) + (A1 * c4) / (REAL)(2.0)));
	T[3] = ((d5 * A4) / (REAL)(2.0) - (d5 * A3) / (REAL)(2.0) - d4 * s0 + (d6 * A2 * s4) / (REAL)(2.0) + (d6 * A1 * s4) / (REAL)(2.0) - a2 * c0 * c1 - d6 * c4 * s0 - a3 * c0 * c1 * c2 + a3 * c0 * s1 * s2);
	T[4] = c0 * c4 + (A3 * s4) / (REAL)(2.0) + (A4 * s4) / (REAL)(2.0);
	T[5] = (c5 * ((A3 * c4) / (REAL)(2.0) - c0 * s4 + (A4 * c4) / (REAL)(2.0)) + s5 * (A2 / (REAL)(2.0) - A1 / (REAL)(2.0)));
	T[6] = (c5 * (A2 / (REAL)(2.0) - A1 / (REAL)(2.0)) - s5 * ((A3 * c4) / (REAL)(2.0) - c0 * s4 + (A4 * c4) / (REAL)(2.0)));
	T[7] = ((d5 * A2) / (REAL)(2.0) - (d5 * A1) / (REAL)(2.0) + d4 * c0 + (d6 * A3 * s4) / (REAL)(2.0) + (d6 * A4 * s4) / (REAL)(2.0) + d6 * c0 * c4 - a2 * c1 * s0 - a3 * c1 * c2 * s0 + a3 * s0 * s1 * s2);
	T[8] = (A6 / (REAL)(2.0) - A5 / (REAL)(2.0));
	T[9] = ((s123 * c5 - c123 * s5) / (REAL)(2.0) - (s123 * c5 + c123 * s5) / (REAL)(2.0) - s123 * c4 * c5);
	T[10] = (s123 * c4 * s5 - (c123 * c5 + s123 * s5) / (REAL)(2.0) - (c123 * c5 - s123 * s5) / (REAL)(2.0));
	T[11] = (d1 + (d6 * A6) / (REAL)(2.0) + a3 * (s1 * c2 + c1 * s2) + a2 * s1 - (d6 * A5) / (REAL)(2.0) - d5 * c123);

	REAL Ttemp[8];
	Ttemp[0] = T[0] * COS_Z_ROTATION - T[4] * SIN_Z_ROTATION;
	Ttemp[1] = T[1] * COS_Z_ROTATION - T[5] * SIN_Z_ROTATION;
	Ttemp[2] = T[2] * COS_Z_ROTATION - T[6] * SIN_Z_ROTATION;
	Ttemp[3] = T[3] * COS_Z_ROTATION - T[7] * SIN_Z_ROTATION;
	Ttemp[4] = T[0] * SIN_Z_ROTATION + T[4] * COS_Z_ROTATION;
	Ttemp[5] = T[1] * SIN_Z_ROTATION + T[5] * COS_Z_ROTATION;
	Ttemp[6] = T[2] * SIN_Z_ROTATION + T[6] * COS_Z_ROTATION;
	Ttemp[7] = T[3] * SIN_Z_ROTATION + T[7] * COS_Z_ROTATION;
	for (unsigned int i = 0; i < 8; i++) T[i] = Ttemp[i];

	REAL quaternion[4], t;
	if (T[0] + T[4 + 1] + T[2 * 4 + 2] > (REAL)(0.0)) {
		t = T[0] + T[4 + 1] + T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[3] = t;
		quaternion[2] = (T[1] - T[4]);
		quaternion[1] = (T[2 * 4] - T[2]);
		quaternion[0] = (T[4 + 2] - T[2 * 4 + 1]);
	}
	else if (T[0] > T[4 + 1] && T[0] > T[2 * 4 + 2]) {
		t = T[0] - T[4 + 1] - T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[0] = t;
		quaternion[1] = (T[1] + T[4]);
		quaternion[2] = (T[2 * 4] + T[2]);
		quaternion[3] = (T[4 + 2] - T[2 * 4 + 1]);
	}
	else if (T[4 + 1] > T[2 * 4 + 2]) {
		t = -T[0] + T[4 + 1] - T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[1] = t;
		quaternion[0] = (T[1] + T[4]);
		quaternion[3] = (T[2 * 4] - T[2]);
		quaternion[2] = (T[4 + 2] + T[2 * 4 + 1]);
	}
	else {
		t = -T[0] - T[4 + 1] + T[2 * 4 + 2] + (REAL)(1.0);
		quaternion[2] = t;
		quaternion[3] = (T[1] - T[4]);
		quaternion[0] = (T[2 * 4] + T[2]);
		quaternion[1] = (T[4 + 2] + T[2 * 4 + 1]);
	}
	REAL s = reciprocalSqrt(t) * (REAL)(0.5);
	quaternion[0] *= s;	quaternion[1] *= s;	quaternion[2] *= s;	quaternion[3] *= s;

	unsigned int poseIndex = jointPoints[index].poseIndex;
	cartesianPoints[index].xyz[0] = (float)(T[3] + poses[poseIndex].xyz[0]);
	cartesianPoints[index].xyz[1] = (float)(T[7] + poses[poseIndex].xyz[1]);
	cartesianPoints[index].xyz[2] = (float)(T[11] + poses[poseIndex].xyz[2]);

	T[0] = poses[poseIndex].quaternion[0];
	T[1] = poses[poseIndex].quaternion[1];
	T[2] = poses[poseIndex].quaternion[2];
	T[3] = poses[poseIndex].quaternion[3];

	T[4] = quaternion[0] * T[3] + quaternion[1] * T[2] - quaternion[2] * T[1] + quaternion[3] * T[0];
	T[5] = -quaternion[0] * T[2] + quaternion[1] * T[3] + quaternion[2] * T[0] + quaternion[3] * T[1];
	T[6] = quaternion[0] * T[1] - quaternion[1] * T[0] + quaternion[2] * T[3] + quaternion[3] * T[2];
	T[7] = -quaternion[0] * T[0] - quaternion[1] * T[1] - quaternion[2] * T[2] + quaternion[3] * T[3];

	T[8] = Sqrt(T[4] * T[4] + T[5] * T[5] + T[6] * T[6] + T[7] * T[7]);
	if (T[8] < (REAL)(0.0)) T[8] *= (REAL)(-1.0);
	T[4] /= T[8];
	T[5] /= T[8];
	T[6] /= T[8];
	T[7] /= T[8];

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
		// std::cout << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

		start = std::chrono::system_clock::now();
		{ // GPU
			cudaMemcpy(dev_JointPoints, host_JointPoints, nPoints * sizeof(JointPoint), cudaMemcpyHostToDevice);
			unsigned int nBlocks = nPoints / 128;
			if (nPoints % 128 != 0) nBlocks++;
			for (int i = 0; i < nKernel; i++) jointToCartesianKernel << <nBlocks, 128 >> > (nPoints, dev_DhParameters, dev_Poses, dev_JointPoints, dev_CartesianPoints);
			cudaDeviceSynchronize();
			cudaMemcpy(host_CartesianPoints, dev_CartesianPoints, nPoints * sizeof(CartesianPoint), cudaMemcpyDeviceToHost);
			cudaDeviceSynchronize();
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

	evaluateJointToCartesian(16, 32, 1);

	unsigned int nSamples = 16;
	for (double i = 64 * 1024 * 1024; i >= 1000; i /= 1.414213562373095) {
		evaluateJointToCartesian(nSamples, i, 0);
		for (unsigned int k = 1; k <= 256; k *= 2) {
			evaluateJointToCartesian(nSamples, i, k);
		}
	}

	while (true);

	return 1;
}