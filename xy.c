#include<stdlib.h>
#include<stdio.h>
#include<math.h>
#include<time.h>

#define rand_latt_elem() PI_2*((float)rand()/RAND_MAX);
#define INITIAL_AVG_IND num_steps/5
#define PI_2 6.283185307

#define norm_vect_len(m) (sqrt(pow((m).x, 2) + pow((m).y, 2))/(m).num_elem)

typedef struct {
  float x;
  float y;
  int num_elem;
} magn_t;

float calc_energy(float *latt, int latt_len);
void calc_mag(float *latt, int latt_len, magn_t *m);
void write_data(float *Temp, float *Enrg, float *Magn, int data_len);

int main(int argc, char **argv) {
  if(argc < 4) {
    printf("usage: %s num_temps latt_len num_steps\n", argv[0]);
    exit(0);
  }

  int num_temps = atoi(argv[1]);
  int latt_len = atoi(argv[2]);
  int num_steps = atoi(argv[3]);

  float *Temp = malloc(num_temps * sizeof(float));
  float *Enrg = malloc(num_temps * sizeof(float));
  float *Magn = malloc(num_temps * sizeof(float));
  float *latt = malloc(num_temps * sizeof(float));

  srand(time(0));

  // Initialize temperature array
  float curT = 0.05;
  float T_step = 2./num_temps;
  for(int i = 0; i < num_temps; i++, curT += T_step) 
    Temp[i] = curT;

  // Calculate energy and magnetization for each temperature in array
  for(int i = 0; i < num_temps; i++) {
    float avg_nrg = 0.0, avg_mag = 0.0;
    // Initialize random lattice and calculate inital energy
    for(int j = 0; j < latt_len; j++) {
      latt[j] = rand_latt_elem();
    }
    //float nrg = calc_energy(latt, latt_len);
    magn_t magn;
		//calc_mag(latt, latt_len, &magn);
    // Take q timesteps
    for(int j = 0; j < num_steps; j++) {
      // Randomly choose pertubation to lattice
      int rand_ind = (int)(((float)rand()/RAND_MAX)*latt_len);
      float new_val = rand_latt_elem(); 
      // Calculate effect of perturbation and randomly accept
      float delta_nrg = 2.*cos(new_val-latt[rand_ind])-2.; // -2. accounts for k == rand_ind
      for(int k = 0; k < latt_len; k++)
				delta_nrg -= 2*cos(new_val-latt[k])-2*cos(latt[rand_ind]-latt[k]);
      delta_nrg /= latt_len;
      // original code contained bug, exponent on e should be negative!!!
      // why are temps in the gold graphs and XY.py graphs different?
      if((((float)rand()/RAND_MAX) < exp(-delta_nrg/Temp[i])) || delta_nrg < 0) { 
				//nrg += delta_nrg;
				//magn.x += cos(new_val) - cos(latt[rand_ind]);
				//magn.y += sin(new_val) - cos(latt[rand_ind]);
				latt[rand_ind] = new_val;
      }
      // Save energy and magnetization if we are past base timestep
      /* Why are we not just taking final energy, magnetization??
				 if(j >= INITIAL_AVG_IND) {
				 avg_nrg += nrg;
				 avg_mag += norm_vect_len(magn);
				 }
      */
    }
    //avg_nrg /= latt_len-INITIAL_AVG_IND;
    //avg_mag /= latt_len-INITIAL_AVG_IND;
    Enrg[i] = calc_energy(latt, latt_len);
		calc_mag(latt, latt_len, &magn);
    Magn[i] = norm_vect_len(magn);
  }

  write_data(Temp, Enrg, Magn, num_temps);

  free(Temp);
  free(Enrg);
  free(Magn);
  free(latt);
}

float calc_energy(float *latt, int latt_len) {
  float nrg = 0.0;
  for(int i = 0; i < latt_len; i++)
    for(int j = i+1; j < latt_len; j++)
      nrg -= 2*cos(latt[i] - latt[j]);
  return nrg/latt_len;
}

void calc_mag(float *latt, int latt_len, magn_t *m) {
  m->x = 0.0;
  m->y = 0.0;
  m->num_elem = latt_len;
  for(int i = 0; i < latt_len; i++) {
    m->x += cos(latt[i]);
    m->y += sin(latt[i]);
  }
}

void write_data(float *Temp, float *Enrg, float *Magn, int data_len) {
  for(int i = 0; i < data_len; i++)
    printf("%f,%f,%f\n", Temp[i], Enrg[i], Magn[i]);
}
