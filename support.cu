#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>
#include"support.h"


void init_temp(float *Temp, int num_temps) {
	float curT = MIN_TEMP;
	float T_step = (MAX_TEMP-MIN_TEMP)/num_temps;
	for(int i = 0; i < num_temps; i++, curT += T_step)
		Temp[i] = curT;
}

void init_latt(float *latt, int latt_len) {
	static int seeded = 0;
	if(!seeded) {
		srand(time(0));
		seeded = 1;
	}
	for(int i = 0; i < latt_len; i++)
		latt[i] = rand_latt_elem();
}

void write_data(float *Temp, float *Enrg, float *Magn, int data_len) {
  for(int i = 0; i < data_len; i++)
    printf("%f,%f,%f\n", Temp[i], Enrg[i], Magn[i]);
}


void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}
