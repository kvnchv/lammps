/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "min_cac_quickmin.h"
#include <mpi.h>
#include <cmath>
#include "universe.h"
#include "atom.h"
#include "force.h"
#include "update.h"
#include "output.h"
#include "timer.h"
#include "error.h"
#include "memory.h"

using namespace LAMMPS_NS;

// EPS_ENERGY = minimum normalization for energy tolerance

#define EPS_ENERGY 1.0e-8

#define DELAYSTEP 5

/* ---------------------------------------------------------------------- */

CACMinQuickMin::CACMinQuickMin(LAMMPS *lmp) : Min(lmp) {}

/* ---------------------------------------------------------------------- */

void CACMinQuickMin::init()
{
  Min::init();
  if (!atom->CAC_flag) error->all(FLERR,"CAC min styles require a CAC atom style");
  if (!atom->CAC_pair_flag) error->all(FLERR,"CAC min styles require a CAC pair style");
  dt = update->dt;
  last_negative = update->ntimestep;
}

/* ---------------------------------------------------------------------- */

void CACMinQuickMin::setup_style()
{
  double *min_v = atom->min_v;
  nvec=atom->dense_count;
  for (int i=0; i < nvec; i ++) min_v[i] = 0.0;
}

/* ----------------------------------------------------------------------
   set current vector lengths and pointers
   called after atoms have migrated
------------------------------------------------------------------------- */

void CACMinQuickMin::reset_vectors()
{
  int *npoly = atom->poly_count;
  int *nodes_per_element_list = atom->nodes_per_element_list;
  int *element_type = atom->element_type;

  atom->dense_count=0;
  for(int element_counter=0; element_counter < atom->nlocal; element_counter++){
     atom->dense_count+=3*npoly[element_counter]*nodes_per_element_list[element_type[element_counter]];
  }
  //copy nodal arrays to the continuous arrays for the min algorithm
  copy_force();
  nvec=atom->dense_count;
  if (nvec) xvec = atom->min_x;
  if (nvec) fvec = atom->min_f;
}

/* ----------------------------------------------------------------------
   minimization via QuickMin damped dynamics
------------------------------------------------------------------------- */

int CACMinQuickMin::iterate(int maxiter)
{
  bigint ntimestep;
  double vmax,vdotf,vdotfall,fdotf,fdotfall,scale;
  double dtvone,dtv,dtf,dtfm;
  int flag,flagall;

  int *element_type = atom->element_type;
  int **node_types = atom->node_types;
  int *npoly = atom->poly_count;
  int *nodes_per_element_list = atom->nodes_per_element_list;

  alpha_final = 0.0;

  for (int iter = 0; iter < maxiter; iter++) {

    if (timer->check_timeout(niter))
      return TIMEOUT;

    ntimestep = ++update->ntimestep;
    niter++;

    // zero velocity if anti-parallel to force
    // else project velocity in direction of force

    double *f = atom->min_f;
    double *v = atom->min_v;
    nvec=atom->dense_count;

    vdotf = 0.0;
    for (int i = 0; i < nvec; i+=3)
      vdotf += v[i]*f[i] + v[i]*f[i] + v[i]*f[i];
    MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,world);

    // sum vdotf over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      vdotf = vdotfall;
      MPI_Allreduce(&vdotf,&vdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
    }

    if (vdotfall < 0.0) {
      last_negative = ntimestep;
      for (int i = 0; i < nvec; i+=3)
        v[i] = v[i+1] = v[i+2] = 0.0;

    } else {
      fdotf = 0.0;
      for (int i = 0; i < nvec; i+=3)
        fdotf += f[i]*f[i] + f[i]*f[i] + f[i]*f[i];
      MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,world);

      // sum fdotf over replicas, if necessary
      // this communicator would be invalid for multiprocess replicas

      if (update->multireplica == 1) {
        fdotf = fdotfall;
        MPI_Allreduce(&fdotf,&fdotfall,1,MPI_DOUBLE,MPI_SUM,universe->uworld);
      }

      if (fdotfall == 0.0) scale = 0.0;
      else scale = vdotfall/fdotfall;
      for (int i = 0; i < nvec; i+=3) {
        v[i] = scale*f[i];
        v[i+1] = scale*f[i+1];
        v[i+2] = scale*f[i+2];
      }
    }

    // limit timestep so no particle moves further than dmax

    double *rmass = atom->rmass;
    double *mass = atom->mass;
    int *type = atom->type;

    dtvone = dt;

    for (int i = 0; i < nvec; i+=3) {
      vmax = MAX(fabs(v[i]),fabs(v[i+1]));
      vmax = MAX(vmax,fabs(v[i+3]));
      if (dtvone*vmax > dmax) dtvone = dmax/vmax;
    }
    MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,world);

    // min dtv over replicas, if necessary
    // this communicator would be invalid for multiprocess replicas

    if (update->multireplica == 1) {
      dtvone = dtv;
      MPI_Allreduce(&dtvone,&dtv,1,MPI_DOUBLE,MPI_MIN,universe->uworld);
    }

    dtf = dtv * force->ftm2v;

    // Euler integration step

    double *x = atom->min_x;

    if (rmass) {
      for (int i = 0; i < nvec; i++) {
        dtfm = dtf / rmass[i];
        x[i] += dtv * v[i];
        v[i] += dtfm * f[i];
      }
    } else {
        int dense = 0;
        // nodal loops required to get mass
        for(int element_counter=0; element_counter < atom->nlocal; element_counter++) {
          for (int poly_counter = 0; poly_counter < npoly[element_counter]; poly_counter++) {
            dtfm = dtf / mass[node_types[element_counter][poly_counter]];
            for(int node_counter=0; node_counter < nodes_per_element_list[element_type[element_counter]]; node_counter++){

              x[dense+0] += dtv * v[dense+0];
              x[dense+1] += dtv * v[dense+1];
              x[dense+2] += dtv * v[dense+2];
              v[dense+0] += dtfm * f[dense+0];
              v[dense+1] += dtfm * f[dense+1];
              v[dense+2] += dtfm * f[dense+2];

              dense+=3;
            }
          }
        }
    }

    eprevious = ecurrent;
    ecurrent = energy_force(0);
    neval++;

    // energy tolerance criterion
    // only check after DELAYSTEP elapsed since velocties reset to 0
    // sync across replicas if running multi-replica minimization

    if (update->etol > 0.0 && ntimestep-last_negative > DELAYSTEP) {
      if (update->multireplica == 0) {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          return ETOL;
      } else {
        if (fabs(ecurrent-eprevious) <
            update->etol * 0.5*(fabs(ecurrent) + fabs(eprevious) + EPS_ENERGY))
          flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return ETOL;
      }
    }

    // force tolerance criterion
    // sync across replicas if running multi-replica minimization

    if (update->ftol > 0.0) {
      fdotf = fnorm_sqr();
      if (update->multireplica == 0) {
        if (fdotf < update->ftol*update->ftol) return FTOL;
      } else {
        if (fdotf < update->ftol*update->ftol) flag = 0;
        else flag = 1;
        MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,universe->uworld);
        if (flagall == 0) return FTOL;
      }
    }

    // output for thermo, dump, restart files

    if (output->next == ntimestep) {
      timer->stamp();
      output->write(ntimestep);
      timer->stamp(Timer::OUTPUT);
    }
  }

  return MAXITER;
}

void CACMinQuickMin::copy_vectors(){
  int *npoly = atom->poly_count;
  int *nodes_per_element_list = atom->nodes_per_element_list;
  int *element_type = atom->element_type;
  double ****nodal_positions = atom->nodal_positions;
  double ****nodal_forces = atom->nodal_forces;
  double ****nodal_velocities = atom->nodal_velocities;
  double *min_x = atom->min_x;
  double *min_f = atom->min_f;
  double *min_v = atom->min_v;
  double **x = atom->x;
  int nodes_per_element;

  //copy contents to these vectors
  int dense_count_x=0;
  int dense_count_f=0;
  int dense_count_v=0;
  for(int element_counter=0; element_counter < atom->nlocal; element_counter++){
    for(int poly_counter=0; poly_counter < npoly[element_counter]; poly_counter++){
      for(int node_counter=0; node_counter < nodes_per_element_list[element_type[element_counter]]; node_counter++){
         nodal_positions[element_counter][poly_counter][node_counter][0] = min_x[dense_count_x++];
         nodal_positions[element_counter][poly_counter][node_counter][1] = min_x[dense_count_x++];
         nodal_positions[element_counter][poly_counter][node_counter][2] = min_x[dense_count_x++];
         nodal_forces[element_counter][poly_counter][node_counter][0] = min_f[dense_count_f++];
         nodal_forces[element_counter][poly_counter][node_counter][1] = min_f[dense_count_f++];
         nodal_forces[element_counter][poly_counter][node_counter][2] = min_f[dense_count_f++];
         nodal_velocities[element_counter][poly_counter][node_counter][0] = min_v[dense_count_v++];
         nodal_velocities[element_counter][poly_counter][node_counter][1] = min_v[dense_count_v++];
         nodal_velocities[element_counter][poly_counter][node_counter][2] = min_v[dense_count_v++];
       }
     }
  }

    // update x for elements and atoms using nodal variables
  for (int i = 0; i < atom->nlocal; i++){
    //determine element type

    nodes_per_element=nodes_per_element_list[element_type[i]];
    x[i][0] = 0;
    x[i][1] = 0;
    x[i][2] = 0;

    for (int poly_counter = 0; poly_counter < npoly[i];poly_counter++) {
      for(int k=0; k<nodes_per_element; k++){
        x[i][0] += nodal_positions[i][poly_counter][k][0];
        x[i][1] += nodal_positions[i][poly_counter][k][1];
        x[i][2] += nodal_positions[i][poly_counter][k][2];
      }
    }
  x[i][0] = x[i][0] / nodes_per_element / npoly[i];
  x[i][1] = x[i][1] / nodes_per_element / npoly[i];
  x[i][2] = x[i][2] / nodes_per_element / npoly[i];
  }

}


/* ----------------------------------------------------------------------
   copy atomvec arrays to continuous arrays after energy_force evaluation
------------------------------------------------------------------------- */

void CACMinQuickMin::copy_force(){
  int *npoly = atom->poly_count;
  int *nodes_per_element_list = atom->nodes_per_element_list;
  int *element_type = atom->element_type;
  double ****nodal_positions = atom->nodal_positions;
  double ****nodal_velocities = atom->nodal_velocities;
  double ****nodal_forces = atom->nodal_forces;
  double *min_x = atom->min_x;
  double *min_v = atom->min_v;
  double *min_f = atom->min_f;
  
  //copy contents of min vectors to the avec arrays and vice versa
  int dense_count_x=0;
  int dense_count_v=0;
  int dense_count_f=0;
  
  //grow the dense aligned vectors
  if(atom->dense_count>densemax){
  min_x = memory->grow(atom->min_x,atom->dense_count,"min_CAC_quickmin:min_x");
  min_v = memory->grow(atom->min_v,atom->dense_count,"min_CAC_quickmin:min_v");
  min_f = memory->grow(atom->min_f,atom->dense_count,"min_CAC_quickmin:min_f");
  densemax=atom->dense_count;
  }

  for(int element_counter=0; element_counter < atom->nlocal; element_counter++){
    for(int poly_counter=0; poly_counter < npoly[element_counter]; poly_counter++){
      for(int node_counter=0; node_counter < nodes_per_element_list[element_type[element_counter]]; node_counter++){
         min_x[dense_count_x++] = nodal_positions[element_counter][poly_counter][node_counter][0];
         min_x[dense_count_x++] = nodal_positions[element_counter][poly_counter][node_counter][1];
         min_x[dense_count_x++] = nodal_positions[element_counter][poly_counter][node_counter][2];
         min_v[dense_count_v++] = nodal_velocities[element_counter][poly_counter][node_counter][0];
         min_v[dense_count_v++] = nodal_velocities[element_counter][poly_counter][node_counter][1];
         min_v[dense_count_v++] = nodal_velocities[element_counter][poly_counter][node_counter][2];
         min_f[dense_count_f++] = nodal_forces[element_counter][poly_counter][node_counter][0];
         min_f[dense_count_f++] = nodal_forces[element_counter][poly_counter][node_counter][1];
         min_f[dense_count_f++] = nodal_forces[element_counter][poly_counter][node_counter][2];
       }
     }
  }
}

