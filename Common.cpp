#include "Common.h"

void setUr5DhParameter(unsigned int index, DhParameters* dhParameters) {
	dhParameters->dh[index].d1 = 0.089159f;
	dhParameters->dh[index].a2 = -0.42500f;
	dhParameters->dh[index].a3 = -0.39225f;
	dhParameters->dh[index].d4 = 0.10915f;
	dhParameters->dh[index].d5 = 0.09465f;
	dhParameters->dh[index].d6 = 0.0823f;
	dhParameters->dh[index].rotZ = (float)M_PI;
}