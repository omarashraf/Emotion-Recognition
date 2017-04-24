
#ifndef _ML_H // must be unique name in the project
#define _ML_H

void trainClassifierEyebrows();
void trainClassifierMouth();
double loadClassifierAndPredictEyebrows(std::vector<double> eyebrows);
double loadClassifierAndPredictMouth(std::vector<double> mouth);

#endif 