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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "pair_cac_coul_dsf.h"
#include "atom.h"
#include "force.h"
#include "comm.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "update.h"
#include "neigh_list.h"
#include "integrate.h"
#include "respa.h"
#include "math_const.h"
#include "memory.h"
#include "error.h"
#include "domain.h"
#include "asa_user.h"

#define MAXNEIGHOUT  50
#define MAXNEIGHIN  10
#define EXPAND 10
#define EWALD_F   1.12837917
#define EWALD_P   0.3275911
#define A1        0.254829592
#define A2       -0.284496736
#define A3        1.421413741
#define A4       -1.453152027
#define A5        1.061405429
using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairCACCoulDSF::PairCACCoulDSF(LAMMPS *lmp) : PairCAC(lmp)
{
  restartinfo = 0;
  nmax = 0;
  outer_neighflag = 0;
  flux_enable = 1;
}

/* ---------------------------------------------------------------------- */

PairCACCoulDSF::~PairCACCoulDSF() {
  if (allocated) {
  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->destroy(inner_neighbor_coords);
  memory->destroy(inner_neighbor_types);
  memory->destroy(inner_neighbor_charges);
  }
}

/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
 allocate all arrays
 ------------------------------------------------------------------------- */

void PairCACCoulDSF::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  max_nodes_per_element = atom->nodes_per_element;
  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");
  memory->create(mass_matrix, max_nodes_per_element, max_nodes_per_element,"pairCAC:mass_matrix");
  memory->create(mass_copy, max_nodes_per_element, max_nodes_per_element,"pairCAC:copy_mass_matrix");
  memory->create(force_column, max_nodes_per_element,3,"pairCAC:force_residue");
  memory->create(current_force_column, max_nodes_per_element,"pairCAC:current_force_residue");
  memory->create(current_nodal_forces, max_nodes_per_element,"pairCAC:current_nodal_force");
  memory->create(pivot, max_nodes_per_element+1,"pairCAC:pivots");
  quadrature_init(2);
}

/* ----------------------------------------------------------------------
global settings
------------------------------------------------------------------------- */
void PairCACCoulDSF::settings(int narg, char **arg) {
  if (narg <2 || narg>3) error->all(FLERR, "Illegal pair_style command");

  force->newton_pair = 0;

  alf = force->numeric(FLERR, arg[0]);
  cut_global_s = force->numeric(FLERR, arg[1]);
  if (narg == 3) {
    if (strcmp(arg[2], "one") == 0) atom->one_layer_flag=one_layer_flag = 1;
    else error->all(FLERR, "Unexpected argument in cac/coul/dsf invocation");
  }
  cut_coul = cut_global_s;
}

/* ----------------------------------------------------------------------
 set coeffs for one or more type pairs
 ------------------------------------------------------------------------- */

void PairCACCoulDSF::coeff(int narg, char **arg) {
  if (narg != 2) error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo, ihi, jlo, jhi;
  force->bounds(FLERR, arg[0], atom->ntypes, ilo, ihi);
  force->bounds(FLERR, arg[1], atom->ntypes, jlo, jhi);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo, i); j <= jhi; j++) {
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
 init for one type pair i,j and corresponding j,i
 ------------------------------------------------------------------------- */

double PairCACCoulDSF::init_one(int i, int j) {
  return cut_global_s;
}

/* ---------------------------------------------------------------------- */


void PairCACCoulDSF::init_style()
{
  PairCAC::init_style();
  atom->max_neigh_inner_init = maxneigh_quad_inner = MAXNEIGHIN;
  atom->max_neigh_outer_init = maxneigh_quad_outer = MAXNEIGHOUT;
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style cac/coul/DSF requires atom IDs");
  atom->max_neigh_inner_init = maxneigh_quad_inner = MAXNEIGHIN;
  atom->max_neigh_outer_init = maxneigh_quad_outer = MAXNEIGHOUT;
  if (!atom->q_flag)
    error->all(FLERR, "Pair coul/DSF requires atom attribute q for charges");
  cut_coulsq = cut_coul*cut_coul;
  double erfcc = erfc(alf*cut_coul);
  double erfcd = exp(-alf*alf*cut_coul*cut_coul);
  f_shift = -(erfcc/cut_coulsq + 2.0/MY_PIS*alf*erfcd/cut_coul);
  e_shift = erfcc/cut_coul - f_shift*cut_coul;
}

//-----------------------------------------------------------------------


void PairCACCoulDSF::force_densities(int iii, double s, double t, double w, double coefficients,
  double &force_densityx, double &force_densityy, double &force_densityz) {

  double delx,dely,delz;

  double r2inv;
  double r6inv;
  double shape_func;
  double shape_func2;
  double fpair;
  double prefactor;
  double r;

  double e_self, qisq;
  double *special_coul = force->special_coul;
  double qqrd2e = force->qqrd2e;
  int *type = atom->type;
  double distancesq;
  double scan_position[3];
  double rcut;
  int current_type = poly_counter;
  int *element_type = atom->element_type;

  int nodes_per_element;
  int *nodes_count_list = atom->nodes_per_element_list;
  int neighbor_nodes_per_element;

  rcut = cut_global_s;
  int origin_type = type_array[poly_counter];

  int listtype;
  int listindex;
  int poly_index;
  int scan_type;
  int element_index;
  int *ilist, *jlist, *numneigh, **firstneigh;
  int neigh_max = inner_quad_lists_counts[pqi];
  
  double ****nodal_positions = atom->nodal_positions;
  int **node_types = atom->node_types;
  double **node_charges = atom->node_charges;
  double origin_element_charge= node_charges[iii][poly_counter];
  double neighbor_element_charge;
  int **inner_quad_indices = inner_quad_lists_index[pqi];
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;
  jlist = firstneigh[iii];
  qisq = origin_element_charge*origin_element_charge;
  e_self = -(e_shift/2.0 + alf/MY_PIS) * qisq*qqrd2e;
  quadrature_energy += e_self;

  //allocate arrays that store neighbor information around just this quadrature point
  allocate_quad_memory();
  //set virtual neighbor types, etc.
  init_quad_arrays();
  //interpolate virtual atom coordinates from shape functions corresponding to unit cells
  interpolation(iii,s,t,w);
  for (int l = 0; l < neigh_max; l++) {

    scan_type = inner_neighbor_types[l];
    scan_position[0] = inner_neighbor_coords[l][0];
    scan_position[1] = inner_neighbor_coords[l][1];
    scan_position[2] = inner_neighbor_coords[l][2];
    neighbor_element_charge = inner_neighbor_charges[l];

    delx = current_position[0] - scan_position[0];
    dely = current_position[1] - scan_position[1];
    delz = current_position[2] - scan_position[2];
    distancesq = delx*delx + dely*dely + delz*delz;
    if (distancesq < cut_coulsq) {
      factor_coul = special_coul[sbmask(inner_quad_indices[l][0])];
      fpair = pair_interaction_q(distancesq, origin_type, scan_type,
        origin_element_charge, neighbor_element_charge);

      force_densityx += delx*fpair;
      force_densityy += dely*fpair;
      force_densityz += delz*fpair;
      if(atom->CAC_virial){
      virial_density[0] += 0.5*delx*delx*fpair;
      virial_density[1] += 0.5*dely*dely*fpair;
      virial_density[2] += 0.5*delz*delz*fpair;
      virial_density[3] += 0.5*delx*dely*fpair;
      virial_density[4] += 0.5*delx*delz*fpair;
      virial_density[5] += 0.5*dely*delz*fpair;
      }
      if (quad_eflag)
        quadrature_energy += ecoul/2;
      //cac flux contribution due to current quadrature point and neighbor pair interactions
      if(quad_flux_flag){
        current_quad_flux(l,delx*fpair,dely*fpair,delz*fpair);
      }
    }
  }
  //end of force density loop
  
  //additional cac flux contributions due to neighbors interacting with neighbors
  //  in the vicinity of this quadrature point
  if (quad_flux_flag) {
    //compute_intersections();
    quad_neigh_flux();
  }
}

/* ---------------------------------------------------------------------- */

double PairCACCoulDSF::pair_interaction_q(double distancesq, int itype, int jtype
                                          , double qi, double qj)
{
  double fpair, forcecoul, prefactor, dvdrr;
  double r, t, erfcc, erfcd;
  double qqrd2e = force->qqrd2e;

  r = sqrt(distancesq);
  prefactor = qqrd2e*qi*qj / r;
  erfcd = exp(-alf*alf*distancesq);
  t = 1.0 / (1.0 + EWALD_P*alf*r);
  erfcc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * erfcd;
  forcecoul = prefactor * (erfcc/r + 2.0*alf/MY_PIS * erfcd +
                          r*f_shift) * r;
  if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;
  fpair = forcecoul / distancesq;
  if (quad_eflag){
    ecoul = prefactor * (erfcc - r*e_shift - distancesq*f_shift);
    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
  }
  return fpair;
}