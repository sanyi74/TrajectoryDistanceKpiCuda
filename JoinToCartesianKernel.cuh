#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "Common.h"

__device__ __host__ float reciprocalSqrt(float number) {
#if defined(__CUDA_ARCH__)
	return __frsqrt_rn(number);
#else
	float xhalf = 0.5f * number;
	int i = *(int*)&number;
	i = 0x5f3759df - (i >> 1);
	number = *(float*)&i;  number = number * (1.5f - (xhalf * number * number));
	return number;
#endif
}

__device__ __host__ unsigned int FloatToUint(float a) {
#if defined(__CUDA_ARCH__)
	return __float2uint_rn(a);
#else
	return (unsigned int)roundf(a);
#endif
}

template<unsigned int SIZE> __device__ __host__ void jointToCartesian(unsigned int index, DhParameters* dhParameters, Poses* poses, JointPoints<SIZE>* jointPoints, CartesianPoints<SIZE>* cartesianPoints) {

	unsigned int dhIndex = FloatToUint(jointPoints->jointPoint[index][6]);
	unsigned int poseIndex = FloatToUint(jointPoints->jointPoint[index][7]);

	float d1 = dhParameters->dh[dhIndex].d1;
	float a2 = dhParameters->dh[dhIndex].a2;
	float a3 = dhParameters->dh[dhIndex].a3;
	float d4 = dhParameters->dh[dhIndex].d4;
	float d5 = dhParameters->dh[dhIndex].d5;
	float d6 = dhParameters->dh[dhIndex].d6;
	float rotZ = dhParameters->dh[dhIndex].rotZ;

	float quaternionPose[4];
	quaternionPose[0] = poses->pose[poseIndex].quaternion[0];
	quaternionPose[1] = poses->pose[poseIndex].quaternion[1];
	quaternionPose[2] = poses->pose[poseIndex].quaternion[2];
	quaternionPose[3] = poses->pose[poseIndex].quaternion[3];
	float xyzPose[3];
	xyzPose[0] = poses->pose[poseIndex].xyz[0];
	xyzPose[1] = poses->pose[poseIndex].xyz[1];
	xyzPose[2] = poses->pose[poseIndex].xyz[2];

	float s0, s1, s2, s4, s5, s123, c0, c1, c2, c4, c5, c123, SIN_Z_ROTATION, COS_Z_ROTATION;
	sincos(jointPoints->jointPoint[index][0], &s0, &c0);
	sincos(jointPoints->jointPoint[index][0], &s0, &c0);
	sincos(jointPoints->jointPoint[index][1], &s1, &c1);
	sincos(jointPoints->jointPoint[index][2], &s2, &c2);
	sincos(jointPoints->jointPoint[index][4], &s4, &c4);
	sincos(jointPoints->jointPoint[index][5], &s5, &c5);
	sincos(jointPoints->jointPoint[index][1] + jointPoints->jointPoint[index][2] + jointPoints->jointPoint[index][3], &s123, &c123);
	sincos(rotZ, &SIN_Z_ROTATION, &COS_Z_ROTATION);

	float A1 = c0 * c123 + s0 * s123;
	float A2 = c0 * c123 - s0 * s123;
	float A3 = s0 * c123 + c0 * s123;
	float A4 = s0 * c123 - c0 * s123;
	float A5 = c123 * c4 + s123 * s4;
	float A6 = c123 * c4 - s123 * s4;

	float T[12];
	T[0] = (A2 * s4) / 2.0 - c4 * s0 + (A1 * s4) / 2.0;
	T[1] = (c5 * (s0 * s4 + (A2 * c4) / 2.0 + (A1 * c4) / 2.0) - (s5 * (A3 - A4)) / 2.0);
	T[2] = (-(c5 * (A3 - A4)) / 2.0 - s5 * (s0 * s4 + (A2 * c4) / 2.0 + (A1 * c4) / 2.0));
	T[3] = ((d5 * A4) / 2.0 - (d5 * A3) / 2.0 - d4 * s0 + (d6 * A2 * s4) / 2.0 + (d6 * A1 * s4) / 2.0 - a2 * c0 * c1 - d6 * c4 * s0 - a3 * c0 * c1 * c2 + a3 * c0 * s1 * s2);
	T[4] = c0 * c4 + (A3 * s4) / 2.0 + (A4 * s4) / 2.0;
	T[5] = (c5 * ((A3 * c4) / 2.0 - c0 * s4 + (A4 * c4) / 2.0) + s5 * (A2 / 2.0 - A1 / 2.0));
	T[6] = (c5 * (A2 / 2.0 - A1 / 2.0) - s5 * ((A3 * c4) / 2.0 - c0 * s4 + (A4 * c4) / 2.0));
	T[7] = ((d5 * A2) / 2.0 - (d5 * A1) / 2.0 + d4 * c0 + (d6 * A3 * s4) / 2.0 + (d6 * A4 * s4) / 2.0 + d6 * c0 * c4 - a2 * c1 * s0 - a3 * c1 * c2 * s0 + a3 * s0 * s1 * s2);
	T[8] = (A6 / 2.0 - A5 / 2.0);
	T[9] = ((s123 * c5 - c123 * s5) / 2.0 - (s123 * c5 + c123 * s5) / 2.0 - s123 * c4 * c5);
	T[10] = (s123 * c4 * s5 - (c123 * c5 + s123 * s5) / 2.0 - (c123 * c5 - s123 * s5) / 2.0);
	T[11] = (d1 + (d6 * A6) / 2.0 + a3 * (s1 * c2 + c1 * s2) + a2 * s1 - (d6 * A5) / 2.0 - d5 * c123);

	float Ttemp[8];
	Ttemp[0] = T[0] * COS_Z_ROTATION - T[4] * SIN_Z_ROTATION;
	Ttemp[1] = T[1] * COS_Z_ROTATION - T[5] * SIN_Z_ROTATION;
	Ttemp[2] = T[2] * COS_Z_ROTATION - T[6] * SIN_Z_ROTATION;
	Ttemp[3] = T[3] * COS_Z_ROTATION - T[7] * SIN_Z_ROTATION;
	Ttemp[4] = T[0] * SIN_Z_ROTATION + T[4] * COS_Z_ROTATION;
	Ttemp[5] = T[1] * SIN_Z_ROTATION + T[5] * COS_Z_ROTATION;
	Ttemp[6] = T[2] * SIN_Z_ROTATION + T[6] * COS_Z_ROTATION;
	Ttemp[7] = T[3] * SIN_Z_ROTATION + T[7] * COS_Z_ROTATION;
	for (auto i = 0; i < 8; i++) T[i] = Ttemp[i];

	float quaternion[4], quaternionTransformed[4];
	if (T[0] + T[4 + 1] + T[2 * 4 + 2] > 0.0) {
		float t = +T[0] + T[4 + 1] + T[2 * 4 + 2] + 1.0;
		float s = reciprocalSqrt(t) * 0.5;
		quaternion[3] = s * t;
		quaternion[2] = (T[1] - T[4]) * s;
		quaternion[1] = (T[2 * 4] - T[2]) * s;
		quaternion[0] = (T[4 + 2] - T[2 * 4 + 1]) * s;
	}
	else if (T[0] > T[4 + 1] && T[0] > T[2 * 4 + 2]) {
		float t = +T[0] - T[4 + 1] - T[2 * 4 + 2] + 1.0;
		float s = reciprocalSqrt(t) * 0.5;
		quaternion[0] = s * t;
		quaternion[1] = (T[1] + T[4]) * s;
		quaternion[2] = (T[2 * 4] + T[2]) * s;
		quaternion[3] = (T[4 + 2] - T[2 * 4 + 1]) * s;
	}
	else if (T[4 + 1] > T[2 * 4 + 2]) {
		float t = -T[0] + T[4 + 1] - T[2 * 4 + 2] + 1.0;
		float s = reciprocalSqrt(t) * 0.5;
		quaternion[1] = s * t;
		quaternion[0] = (T[1] + T[4]) * s;
		quaternion[3] = (T[2 * 4] - T[2]) * s;
		quaternion[2] = (T[4 + 2] + T[2 * 4 + 1]) * s;
	}
	else {
		float t = -T[0] - T[4 + 1] + T[2 * 4 + 2] + 1.0;
		float s = reciprocalSqrt(t) * 0.5;
		quaternion[2] = s * t;
		quaternion[3] = (T[1] - T[4]) * s;
		quaternion[0] = (T[2 * 4] + T[2]) * s;
		quaternion[1] = (T[4 + 2] + T[2 * 4 + 1]) * s;
	}

	quaternionTransformed[0] = quaternion[0] * quaternionPose[3] + quaternion[1] * quaternionPose[2] - quaternion[2] * quaternionPose[1] + quaternion[3] * quaternionPose[0];
	quaternionTransformed[1] = -quaternion[0] * quaternionPose[2] + quaternion[1] * quaternionPose[3] + quaternion[2] * quaternionPose[0] + quaternion[3] * quaternionPose[1];
	quaternionTransformed[2] = quaternion[0] * quaternionPose[1] - quaternion[1] * quaternionPose[0] + quaternion[2] * quaternionPose[3] + quaternion[3] * quaternionPose[2];
	quaternionTransformed[3] = -quaternion[0] * quaternionPose[0] - quaternion[1] * quaternionPose[1] - quaternion[2] * quaternionPose[2] + quaternion[3] * quaternionPose[3];

	float norm = sqrt(quaternionTransformed[0] * quaternionTransformed[0] + quaternionTransformed[1] * quaternionTransformed[1] + quaternionTransformed[2] * quaternionTransformed[2] + quaternionTransformed[3] * quaternionTransformed[3]);
	if (quaternionTransformed[0] < 0) norm *= -1.0;
	quaternionTransformed[0] /= norm;
	quaternionTransformed[1] /= norm;
	quaternionTransformed[2] /= norm;
	quaternionTransformed[3] /= norm;

	cartesianPoints->xyzQuaternionPoint[index][0] = T[3] + xyzPose[0];
	cartesianPoints->xyzQuaternionPoint[index][1] = T[7] + xyzPose[1];
	cartesianPoints->xyzQuaternionPoint[index][2] = T[11] + xyzPose[2];
	cartesianPoints->xyzQuaternionPoint[index][3] = quaternionTransformed[0];
	cartesianPoints->xyzQuaternionPoint[index][4] = quaternionTransformed[1];
	cartesianPoints->xyzQuaternionPoint[index][5] = quaternionTransformed[2];
	cartesianPoints->xyzQuaternionPoint[index][6] = quaternionTransformed[3];
}

template<unsigned int SIZE> __global__ void jointToCartesianKernel(unsigned int nElements, DhParameters* dhParameters, Poses* poses, JointPoints<SIZE>* jointPoints, CartesianPoints<SIZE>* cartesianPoints) {
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < nElements) jointToCartesian(index, dhParameters, poses, jointPoints, cartesianPoints);
}

template<unsigned int SIZE> void jointToCartesianCpu(unsigned int index, DhParameters* dhParameters, Poses* poses, JointPoints<SIZE>* jointPoints, CartesianPoints<SIZE>* cartesianPoints) {
	jointToCartesian(index, dhParameters, poses, jointPoints, cartesianPoints);
}

template<unsigned int SIZE> void evaluateJointToCartesian(unsigned int nSample) {

	cudaError_t cudaRes;

	JointPoints<SIZE>* host_JointPoints;
	CartesianPoints<SIZE>* host_CartesianPoints;
	cudaRes = cudaHostAlloc((void**)&host_JointPoints, sizeof(JointPoints<SIZE>), cudaHostAllocDefault);
	cudaRes = cudaHostAlloc((void**)&host_CartesianPoints, sizeof(CartesianPoints<SIZE>), cudaHostAllocDefault);

	DhParameters* dev_DhParameters;
	Poses* dev_Poses;
	JointPoints<SIZE>* dev_JointPoints;
	CartesianPoints<SIZE>* dev_CartesianPoints;
	cudaRes = cudaMalloc((void**)&dev_DhParameters, sizeof(DhParameters));
	cudaRes = cudaMalloc((void**)&dev_Poses, sizeof(Poses));
	cudaRes = cudaMalloc((void**)&dev_JointPoints, sizeof(JointPoints<SIZE>));
	cudaRes = cudaMalloc((void**)&dev_CartesianPoints, sizeof(CartesianPoints<SIZE>));

	DhParameters host_DhParameters;
	Poses host_Poses;
	setUr5DhParameter(0, &host_DhParameters);
	host_Poses.pose[0].xyz[0] = host_Poses.pose[0].xyz[1] = host_Poses.pose[0].xyz[2] = 0;
	host_Poses.pose[0].quaternion[0] = host_Poses.pose[0].quaternion[1] = host_Poses.pose[0].quaternion[2] = 0;
	host_Poses.pose[0].quaternion[3] = 1.0;
	cudaRes = cudaMemcpy(dev_DhParameters, &host_DhParameters, sizeof(DhParameters), cudaMemcpyHostToDevice);
	cudaRes = cudaMemcpy(dev_Poses, &host_Poses, sizeof(Poses), cudaMemcpyHostToDevice);

	double cpuTime = 0;
	double gpuTime = 0;
	for (auto iter = 0; iter < nSample; iter++) {

		std::cout << "*";
		auto start = std::chrono::system_clock::now();
		{ // CPU
			const unsigned int nThreads = 16;
			std::thread* t[nThreads];
			for (int i = 0; i < nThreads; i++) {
				t[i] = new std::thread([iter, i, host_JointPoints, nThreads]() {
					uint32_t rseed = 2008 * (iter + 7) + 1974 * i;
					rseed = rseed * 214013 + 2531011;
					for (auto index = i * (SIZE / nThreads); index < (i + 1) * (SIZE / nThreads); index++) {
						for (auto j = 0; j < 6; j++) {
							host_JointPoints->jointPoint[index][j] = 2.0 * M_PI * (double)rseed / (double)UINT_MAX - M_PI;
							rseed = rseed * 214013 + 2531011;
						}
						host_JointPoints->jointPoint[index][6] = host_JointPoints->jointPoint[index][7] = 0;
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
			cudaRes = cudaMemcpy(dev_JointPoints, host_JointPoints, sizeof(JointPoints<SIZE>), cudaMemcpyHostToDevice);
			unsigned int nBlocks = SIZE / 64;
			if (SIZE % 64 != 0) nBlocks++;
			jointToCartesianKernel<<<nBlocks,64>>>(SIZE, dev_DhParameters, dev_Poses, dev_JointPoints, dev_CartesianPoints);
			cudaRes = cudaMemcpy(host_CartesianPoints, dev_CartesianPoints, sizeof(CartesianPoints<SIZE>), cudaMemcpyDeviceToHost);
		}
		gpuTime += std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now() - start).count();

		start = std::chrono::system_clock::now();
		{ // CPU
			const unsigned int nThreads = 1;
			std::thread* t[nThreads];
			for (int i = 0; i < nThreads; i++) {
				t[i] = new std::thread([i, &host_DhParameters, &host_Poses, &host_JointPoints, &host_CartesianPoints, nThreads]() {
					for (auto index = i * (SIZE / nThreads); index < (i + 1) * (SIZE / nThreads); index++) 
						jointToCartesianCpu(index, &host_DhParameters, &host_Poses, host_JointPoints, host_CartesianPoints);
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
	std::cout << SIZE << " " << cpuTime << " sec " << gpuTime << " sec " << (gpuTime * 1000000.0 / SIZE) << " us " << (cpuTime / gpuTime) << " " << sizeof(JointPoints<SIZE>) << " " << sizeof(CartesianPoints<SIZE>) << "    " << std::endl;

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

	unsigned int nSamples = 16;
	evaluateJointToCartesian<64>(nSamples);

	return 1;
}