#ifndef FORWARDDECS_H
#define FORWARDDECS_H

double sigmoid(double input);
void initialize(void);
void forwardProp(double x, double y);
double computeLoss(void);
void backProp(int i);

#endif
