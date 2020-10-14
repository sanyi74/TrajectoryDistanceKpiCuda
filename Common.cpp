#include "Common.h"

void setUr5DhParameter(unsigned int index, DhParameter* dhParameters) {
	dhParameters[index].d1 = 0.089159f;
	dhParameters[index].a2 = -0.42500f;
	dhParameters[index].a3 = -0.39225f;
	dhParameters[index].d4 = 0.10915f;
	dhParameters[index].d5 = 0.09465f;
	dhParameters[index].d6 = 0.0823f;
	dhParameters[index].rotZ = (float)M_PI;
}