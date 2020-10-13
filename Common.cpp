#include "Common.h"

void setUr5DhParameter(unsigned int index, DhParameters* dhParameters) {
	dhParameters->dh[index].d1 = 0.089159;
	dhParameters->dh[index].a2 = -0.42500;
	dhParameters->dh[index].a3 = -0.39225;
	dhParameters->dh[index].d4 = 0.10915;
	dhParameters->dh[index].d5 = 0.09465;
	dhParameters->dh[index].d6 = 0.0823;
	dhParameters->dh[index].rotZ = M_PI;
}