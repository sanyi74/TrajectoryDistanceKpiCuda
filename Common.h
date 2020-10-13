#pragma once

#define _USE_MATH_DEFINES

#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>

#include <chrono>
#include <iostream>
#include <thread>

#define MAX_DH 128
#define MAX_POSE 128
#define MAX_TRAJECTORY 16384

struct DhParameters {
	struct DhParameter {
		float d1;
		float a2;
		float a3;
		float d4;
		float d5;
		float d6;
		float rotZ;
		uint32_t id;
	} dh[MAX_DH];
};

struct Poses {
	struct Pose {
		float xyz[3];
		float quaternion[4];
		uint32_t id;
	} pose[MAX_POSE];
};

struct TrajectoryTable {
	uint32_t trajectoryStartEnd[MAX_TRAJECTORY][2];
};

template<unsigned int TRAJCTORY_POINTS> struct JointPoints {
	float jointPoint[TRAJCTORY_POINTS][8]; // 6x joint pose + DH index + Pose index
};

template<unsigned int TRAJCTORY_POINTS> struct CartesianPoints {
	float xyzQuaternionPoint[TRAJCTORY_POINTS][7];
};

template<unsigned int TRAJCTORY_POINTS> struct CartesianKpiPoints {
	struct Kpi {
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
	}kpi[TRAJCTORY_POINTS];
};

struct TrajectoryKPIs {
	struct Kpi {
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
	}kpi[MAX_TRAJECTORY];
};

void setUr5DhParameter(unsigned int index, DhParameters* dhParameters);