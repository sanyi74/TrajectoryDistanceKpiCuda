#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <thread>

struct DhParameter {
	float d1;
	float a2;
	float a3;
	float d4;
	float d5;
	float d6;
	float rotZ;
	uint32_t id;
};

struct Pose {
		float xyz[3];
		float quaternion[4];
};

struct TrajectoryEntry {
	uint32_t trajectoryStartEnd[2];
};

struct JointPoint {
	float q[6];
	uint16_t dhIndex;
	uint16_t poseIndex;
};

struct CartesianPoint {
	float xyz[3];
	float quaternion[4];
};

struct CartesianKpiPoint {
		float length;  // sum
		float lengthOrientation; // sum
		float distance;  // min,max,avg,rsd
		float distanceOrientation; // min,max,avg,rsd
		float distancePath; // min,max,avg,rsd
		float distancePathOrientation; // min,max,avg,rsd
		int32_t distancePathTime; // argMin,min,max,avg,rsd
		int32_t distancePathOrientationTime; // argMin,min,max,avg,rsd
		float distanceLocalPath;
		float distanceLocalPathOrientation;
		int32_t distanceLocalPathTime;
		int32_t distanceLocalPathOrientationTime;
};

struct TrajectoryKPI {
		// min,max,avg,rsd,argmin
		float length;
		float lengthOrientation;
		float distance[4];
		float distanceOrientation[4];
		float distancePath[4];
		float distancePathOrientation[4];
		float distancePathTime[5];
		float distancePathOrientationTime[5];
		float distanceLocalPath[4];
		float distanceLocalPathOrientation[4];
		float distanceLocalPathTime[5];
		float distanceLocalPathOrientationTime[5];
};

void setUr5DhParameter(unsigned int index, DhParameter* dhParameters);