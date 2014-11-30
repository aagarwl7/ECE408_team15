#ifndef _SUPPORT_H
#define _SUPPORT_H

#include<sys/time.h>

#define MIN_TEMP 0.05
#define MAX_TEMP 4.0
#define TEMP_DIFF 3.95
#define PI_2 6.283185307
#define rand_latt_elem() (PI_2*uniform())
#define rand_latt_ind() ((int)(latt_len*uniform()))
#define uniform() ((float)rand()/RAND_MAX)

typedef struct {
	struct timeval startTime;
	struct timeval endTime;
} Timer;

#define FATAL(msg, ...) \
	do { \
		fprintf(stderr, "[%s:%d] "msg"\n", __FILE__, __LINE__, ##__VA_ARGS__); \
		exit(-1); \
	} while(0)

void init_temp(float *Temp, int num_temps);
void init_latt(float *latt, int latt_len);
void write_data(float *Temp, float *Enrg, float *Magn, int data_len);

void startTime(Timer* timer);
void stopTime(Timer* timer);
float elapsedTime(Timer timer);

#endif //_SUPPORT_H

