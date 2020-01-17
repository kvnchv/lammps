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

// lmptype.h must be first b/c this file uses MAXBIGINT and includes mpi.h
// due to OpenMPI bug which sets INT64_MAX via its mpi.h
//   before lmptype.h can set flags to insure it is done correctly

#include "lmptype.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "neb_cac.h"
#include "universe.h"
#include "atom.h"
#include "update.h"
#include "domain.h"
#include "comm.h"
#include "comm_cac.h"
#include "min.h"
#include "modify.h"
#include "fix.h"
#include "fix_cac_neb.h"
#include "output.h"
#include "thermo.h"
#include "finish.h"
#include "timer.h"
#include "memory.h"
#include "error.h"
#include "force.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define MAXLINE 256
#define CHUNK 1024
#define ATTRIBUTE_PERLINE 6
#define MAXELEMENT 100      // max # of lines in one element


/* ---------------------------------------------------------------------- */

NEBCAC::NEBCAC(LAMMPS *lmp) : Pointers(lmp) {}

/* ----------------------------------------------------------------------
   internal NEB constructor, called from TAD
------------------------------------------------------------------------- */

NEBCAC::NEBCAC(LAMMPS *lmp, double etol_in, double ftol_in, int n1steps_in,
         int n2steps_in, int nevery_in, double *buf_init, double *buf_final)
  : Pointers(lmp)
{
  double delx,dely,delz;

  etol = etol_in;
  ftol = ftol_in;
  n1steps = n1steps_in;
  n2steps = n2steps_in;
  nevery = nevery_in;

  // replica info

  nreplica = universe->nworlds;
  ireplica = universe->iworld;
  me_universe = universe->me;
  uworld = universe->uworld;
  MPI_Comm_rank(world,&me);

  // generate linear interpolate replica

  double fraction = ireplica/(nreplica-1.0);

  double **x = atom->x;
  double **nodes = atom->nodal_positions[0][0];
  int nlocal = atom->maxpoly*atom->nodes_per_element * atom->nlocal;

//  int nlocal = atom->nlocal;

  int ii = 0;
  for (int i = 0; i < nlocal; i++) {
    delx = buf_final[ii] - buf_init[ii];
    dely = buf_final[ii+1] - buf_init[ii+1];
    delz = buf_final[ii+2] - buf_init[ii+2];
    domain->minimum_image(delx,dely,delz);
    nodes[i][0] = buf_init[ii] + fraction*delx;
    nodes[i][1] = buf_init[ii+1] + fraction*dely;
    nodes[i][2] = buf_init[ii+2] + fraction*delz;
    ii += 3;
  }
}

/* ---------------------------------------------------------------------- */

NEBCAC::~NEBCAC()
{
  MPI_Comm_free(&roots);
  memory->destroy(all);
  delete [] rdist;
}

/* ----------------------------------------------------------------------
   perform NEB on multiple replicas
------------------------------------------------------------------------- */

void NEBCAC::command(int narg, char **arg)
{
  if (domain->box_exist == 0)
    error->all(FLERR,"NEB command before simulation box is defined");

  if (narg < 6) error->universe_all(FLERR,"Illegal NEB command");

  etol = force->numeric(FLERR,arg[0]);
  ftol = force->numeric(FLERR,arg[1]);
  n1steps = force->inumeric(FLERR,arg[2]);
  n2steps = force->inumeric(FLERR,arg[3]);
  nevery = force->inumeric(FLERR,arg[4]);

  // error checks

  if (etol < 0.0) error->all(FLERR,"Illegal NEB command");
  if (ftol < 0.0) error->all(FLERR,"Illegal NEB command");
  if (nevery <= 0) error->universe_all(FLERR,"Illegal NEB command");
  if (n1steps % nevery || n2steps % nevery)
    error->universe_all(FLERR,"Illegal NEB command");

  // replica info

  nreplica = universe->nworlds;
  ireplica = universe->iworld;
  me_universe = universe->me;
  uworld = universe->uworld;
  MPI_Comm_rank(world,&me);

  // error checks

  if (nreplica == 1) error->all(FLERR,"Cannot use NEB with a single replica");
  if (atom->map_style == 0)
    error->all(FLERR,"Cannot use NEB unless atom map exists");

  // process file-style setting to setup initial configs for all replicas

  if (strcmp(arg[5],"final") == 0) {
    if (narg != 7 && narg !=8) error->universe_all(FLERR,"Illegal NEB command");
    infile = arg[6];
    readfile(infile,0);
  } else if (strcmp(arg[5],"each") == 0) {
    if (narg != 7 && narg !=8) error->universe_all(FLERR,"Illegal NEB command");
    infile = arg[6];
    readfile(infile,1);
  } else if (strcmp(arg[5],"none") == 0) {
    if (narg != 6 && narg !=7) error->universe_all(FLERR,"Illegal NEB command");
  } else error->universe_all(FLERR,"Illegal NEB command");

  verbose=false;
  if (strcmp(arg[narg-1],"verbose") == 0) verbose=true;
  // run the NEB calculation
  

  run();
}

/* ----------------------------------------------------------------------
   run NEB on multiple replicas
------------------------------------------------------------------------- */

void NEBCAC::run()
{
  // create MPI communicator for root proc from each world

  int color;
  if (me == 0) color = 0;
  else color = 1;
  MPI_Comm_split(uworld,color,0,&roots);

  int ineb;
  for (ineb = 0; ineb < modify->nfix; ineb++)
    if (strcmp(modify->fix[ineb]->style,"cac/neb") == 0) break;
  if (ineb == modify->nfix) error->all(FLERR,"cac/neb requires use of fix cac/neb");

  fneb = (FixNEBCAC *) modify->fix[ineb];
  if (verbose) numall =7;
  else  numall = 4;
  memory->create(all,nreplica,numall,"neb:all");
  rdist = new double[nreplica];

  // initialize LAMMPS

  update->whichflag = 2;
  update->etol = etol;
  update->ftol = ftol;
  update->multireplica = 1;

  lmp->init();

  if (update->minimize->searchflag)
    error->all(FLERR,"NEB requires damped dynamics minimizer");

  // setup regular NEB minimization
  FILE *uscreen = universe->uscreen;
  FILE *ulogfile = universe->ulogfile;

  if (me_universe == 0 && uscreen)
    fprintf(uscreen,"Setting up regular NEB ...\n");

  update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + n1steps;
  update->nsteps = n1steps;
  update->max_eval = n1steps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps for NEB");

  update->minimize->setup();

  if (me_universe == 0) {
    if (uscreen) {
      if (verbose) {
        fprintf(uscreen,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN pathangle1 angletangrad1 anglegrad1 gradV1 "
                "ReplicaForce1 MaxAtomForce1 pathangle2 angletangrad2 "
                "... ReplicaForceN MaxAtomForceN\n");
      } else {
        fprintf(uscreen,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN\n");
      }
    }

    if (ulogfile) {
      if (verbose) {
        fprintf(ulogfile,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN pathangle1 angletangrad1 anglegrad1 gradV1 "
                "ReplicaForce1 MaxAtomForce1 pathangle2 angletangrad2 "
                "... ReplicaForceN MaxAtomForceN\n");
      } else {
        fprintf(ulogfile,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT RD1 PE1 RD2 PE2 ... "
                "RDN PEN\n");
      }
    }
  }
  print_status();

  // perform regular NEB for n1steps or until replicas converge
  // retrieve PE values from fix NEB and print every nevery iterations
  // break out of while loop early if converged
  // damped dynamic min styles insure all replicas converge together

  timer->init();
  timer->barrier_start();

  while (update->minimize->niter < n1steps) {
    update->minimize->run(nevery);
    print_status();
    if (update->minimize->stop_condition) break;
  }
  // printf("Energy (%i): %f", me_universe, update->minimize->einitial);

  timer->barrier_stop();

  // update->minimize->cleanup();

  // Finish finish(lmp);
  // finish.end(1);

  // switch fix NEB to climbing mode
  // top = replica that becomes hill climber

  double vmax = all[0][0];
  int top = 0;
  for (int m = 1; m < nreplica; m++)
    if (vmax < all[m][0]) {
      vmax = all[m][0];
      top = m;
    }

  // setup climbing NEB minimization
  // must reinitialize minimizer so it re-creates its fix MINIMIZE

  if (me_universe == 0 && uscreen)
    fprintf(uscreen,"Setting up climbing ...\n");

  if (me_universe == 0) {
    if (uscreen)
      fprintf(uscreen,"Climbing replica = %d\n",top+1);
    if (ulogfile)
      fprintf(ulogfile,"Climbing replica = %d\n",top+1);
  }

  // update->beginstep = update->firststep = update->ntimestep;
  update->endstep = update->laststep = update->firststep + n2steps;
  update->nsteps = n2steps;
  update->max_eval = n2steps;
  if (update->laststep < 0)
    error->all(FLERR,"Too many timesteps");

  // update->minimize->init();
  fneb->rclimber = top;
  // update->minimize->setup();

  if (me_universe == 0) {
    if (uscreen) {
      if (verbose) {
        fprintf(uscreen,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN "
                "pathangle1 angletangrad1 anglegrad1 gradV1 "
                "ReplicaForce1 MaxAtomForce1 pathangle2 angletangrad2 "
                "... ReplicaForceN MaxAtomForceN\n");
      } else {
        fprintf(uscreen,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc "
                "EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN\n");
      }
    }
    if (ulogfile) {
      if (verbose) {
        fprintf(ulogfile,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN "
                "pathangle1 angletangrad1 anglegrad1 gradV1 "
                "ReplicaForce1 MaxAtomForce1 pathangle2 angletangrad2 "
                "... ReplicaForceN MaxAtomForceN\n");
      } else {
        fprintf(ulogfile,"Step MaxReplicaForce MaxAtomForce "
                "GradV0 GradV1 GradVc "
                "EBF EBR RDT "
                "RD1 PE1 RD2 PE2 ... RDN PEN\n");
      }
    }
  }
  print_status();

  // perform climbing NEB for n2steps or until replicas converge
  // retrieve PE values from fix NEB and print every nevery iterations
  // break induced if converged
  // damped dynamic min styles insure all replicas converge together

  timer->init();
  timer->barrier_start();

  while (update->minimize->niter < n2steps) {
    update->minimize->run(nevery);
    print_status();
    if (update->minimize->stop_condition) break;
  }

  timer->barrier_stop();

  update->minimize->cleanup();
  Finish finish(lmp);
  finish.end(1);

  update->whichflag = 0;
  update->multireplica = 0;
  update->firststep = update->laststep = 0;
  update->beginstep = update->endstep = 0;
}

/* ----------------------------------------------------------------------
   read initial config atom coords from file
   flag = 0
   only first replica opens file and reads it
   first replica bcasts lines to all replicas
   final replica stores coords
   intermediate replicas interpolate from coords
   new coord = replica fraction between current and final state
   initial replica does nothing
   flag = 1
   each replica (except first) opens file and reads it
   each replica stores coords
   initial replica does nothing
------------------------------------------------------------------------- */

void NEBCAC::readfile(char *file, int flag)
{
  int i,j,m,nchunk,eofflag,nlines, nodes_per_element;
  tagint tag;
  char *eof,*start,*next,*buf;
  char line[MAXLINE];
  double xx,yy,zz,delx,dely,delz;

  if (me_universe == 0 && screen)
    fprintf(screen,"Reading NEB coordinate file(s) ...\n");

  // flag = 0, universe root reads header of file, bcast to universe
  // flag = 1, each replica's root reads header of file, bcast to world
  //   but explicitly skip first replica
  // nlines in the CAC case is the number of full CAC elements to read
  if (flag == 0) {
    if (me_universe == 0) {
      open(file);
      while (1) {
        eof = fgets(line,MAXLINE,fp);
        if (eof == NULL) error->one(FLERR,"Unexpected end of neb file");
        start = &line[strspn(line," \t\n\v\f\r")];
        if (*start != '\0' && *start != '#') break;
      }
      sscanf(line,"%d",&nlines);
    }
    MPI_Bcast(&nlines,1,MPI_INT,0,uworld);

  } else {
    if (me == 0) {
      if (ireplica) {
        open(file);
        while (1) {
          eof = fgets(line,MAXLINE,fp);
          if (eof == NULL) error->one(FLERR,"Unexpected end of neb file");
          start = &line[strspn(line," \t\n\v\f\r")];
          if (*start != '\0' && *start != '#') break;
        }
        sscanf(line,"%d",&nlines);
      } else nlines = 0;
    }
    MPI_Bcast(&nlines,1,MPI_INT,0,world);
  }


  int npoly, nodecount;
  int decline = 6;
  char *buffer = new char[CHUNK*MAXLINE*MAXELEMENT];
  char **values = new char*[2 * MAXELEMENT*atom->words_per_node];
  char *read_element_type;

  double fraction = ireplica/(nreplica-1.0);
  double **x = atom->x;
  double ****nodal_positions=atom->nodal_positions;
  int nlocal = atom->maxpoly * atom->nodes_per_element * atom->nlocal;

  // loop over chunks of lines read from file
  // two versions of read_lines_from_file() for world vs universe bcast
  // count # of atom coords changed so can check for invalid atom IDs in file

  int nread = 0;
  int index = 0;
  while (nread < nlines) {
    int ncount = 0;
    nchunk = MIN(nlines-nread,CHUNK);
 

    if (flag == 0)
      eofflag = read_lines_from_CAC_universe(fp,nchunk,MAXLINE,MAXELEMENT,buffer);
    else
      eofflag = read_lines_from_CAC(fp,nchunk,MAXLINE,MAXELEMENT,buffer);
    if (eofflag) error->all(FLERR,"Unexpected end of neb file");
    buf = buffer;
    


    // loop over lines of element/nodal coords
    // tokenize the line into values
    for (i = 0; i < nchunk; i++) {
      if (i == 0) {
        values[0] = strtok(buf, " \t\n\r\f");
      }
      else {
        values[0] = strtok(NULL, " \t\n\r\f");
      }
      if (values[0] == NULL)
        error->one(FLERR, "Incorrect atom format in data file");
      values[1] = strtok(NULL, " \t\n\r\f");

      if (values[1] == NULL)
        error->one(FLERR, "Incorrect atom format in data file");
      read_element_type = values[1];

      values[2] = strtok(NULL, " \t\n\r\f");
      if (values[2] == NULL)
        error->one(FLERR, "Incorrect atom format in data file");
      npoly = atoi(values[2]);
      if (strcmp(read_element_type, "Eight_Node") == 0) {
        nodecount = 8;  
      }
      else if (strcmp(read_element_type, "Atom") == 0) {
        nodecount = 1;
        npoly = 1;
      }

      
      for (j = decline - 3; j < nodecount*npoly*atom->words_per_node + decline; j++) {
        values[j] = strtok(NULL, " \t\n\r\f");
        if (values[j] == NULL)
          error->one(FLERR, "Incorrect atom format in data file");
      }

      // adjust nodal coord based on replica fraction
      // for flag = 0, interpolate for intermediate and final replicas
      // for flag = 1, replace existing coord with new coord
      // ignore image flags of final x
      // for interpolation:
      //   new x is displacement from old x via minimum image convention
      //   if final x is across periodic boundary:
      //     new x may be outside box
      //     will be remapped back into box when simulation starts
      //     its image flags will then be adjusted
      tag = ATOTAGINT(values[0]);
      m = atom->map(tag);
      if (m >= 0 && m < atom->nlocal) {
        x[m][0] = x[m][1] = x[m][2] = 0;
        for (int p = 0; p < npoly; p++){  
          for (int k = 0; k < nodecount; k++) {          
            index = 9 + p*nodecount*decline + k*decline;
            xx = atof(values[index]);
            yy = atof(values[index+1]);
            zz = atof(values[index+2]);
            if (flag == 0) {
              delx = xx - nodal_positions[m][p][k][0];
              dely = yy - nodal_positions[m][p][k][1];
              delz = zz - nodal_positions[m][p][k][2];
              domain->minimum_image(delx,dely,delz);
              nodal_positions[m][p][k][0] += fraction*delx;
              nodal_positions[m][p][k][1] += fraction*dely;
              nodal_positions[m][p][k][2] += fraction*delz;
            }
            else {
              nodal_positions[m][p][k][0] = xx;
              nodal_positions[m][p][k][1] = yy;
              nodal_positions[m][p][k][2] = zz;
            }
            x[m][0] += nodal_positions[m][p][k][0];
            x[m][1] += nodal_positions[m][p][k][1];
            x[m][2] += nodal_positions[m][p][k][2];
            ncount++;
          }
        }
        x[m][0] = x[m][0] / nodecount / npoly;
        x[m][1] = x[m][1] / nodecount / npoly;
        x[m][2] = x[m][2] / nodecount / npoly;
      }
    }
    nread += ncount;

  }

  // check that all IDs in file were found by a proc

  if (flag == 0) {
    int ntotal;
    MPI_Allreduce(&nread,&ntotal,1,MPI_INT,MPI_SUM,uworld);
    if (ntotal != nreplica*nlines)
      error->universe_all(FLERR,"Invalid atom IDs in neb file");
  } else {
    int ntotal;
    MPI_Allreduce(&nread,&ntotal,1,MPI_INT,MPI_SUM,world);
    if (ntotal != nlines)
      error->all(FLERR,"Invalid atom IDs in neb file");
  }

  // clean up

  delete [] buffer;
  delete [] values;

  if (flag == 0) {
    if (me_universe == 0) {
      if (compressed) pclose(fp);
      else fclose(fp);
    }
  } else {
    if (me == 0 && ireplica) {
      if (compressed) pclose(fp);
      else fclose(fp);
    }
  }
}

/* ----------------------------------------------------------------------
   universe proc 0 opens NEB data file
   test if gzipped
------------------------------------------------------------------------- */

void NEBCAC::open(char *file)
{
  compressed = 0;
  char *suffix = file + strlen(file) - 3;
  if (suffix > file && strcmp(suffix,".gz") == 0) compressed = 1;
  if (!compressed) fp = fopen(file,"r");
  else {
#ifdef LAMMPS_GZIP
    char gunzip[128];
    sprintf(gunzip,"gzip -c -d %s",file);

#ifdef _WIN32
    fp = _popen(gunzip,"rb");
#else
    fp = popen(gunzip,"r");
#endif

#else
    error->one(FLERR,"Cannot open gzipped file");
#endif
  }

  if (fp == NULL) {
    char str[128];
    sprintf(str,"Cannot open file %s",file);
    error->one(FLERR,str);
  }
}

/* ----------------------------------------------------------------------
   query fix NEB for info on each replica
   universe proc 0 prints current NEB status
------------------------------------------------------------------------- */

void NEBCAC::print_status()
{
  double fnorm2 = sqrt(update->minimize->fnorm_sqr());
  double fmaxreplica;
  MPI_Allreduce(&fnorm2,&fmaxreplica,1,MPI_DOUBLE,MPI_MAX,roots);
  double fnorminf = update->minimize->fnorm_inf();
  double fmaxatom;
  MPI_Allreduce(&fnorminf,&fmaxatom,1,MPI_DOUBLE,MPI_MAX,roots);

  if (verbose) {
    freplica = new double[nreplica];
    MPI_Allgather(&fnorm2,1,MPI_DOUBLE,&freplica[0],1,MPI_DOUBLE,roots);
    fmaxatomInRepl = new double[nreplica];
    MPI_Allgather(&fnorminf,1,MPI_DOUBLE,&fmaxatomInRepl[0],1,MPI_DOUBLE,roots);
  }


  double one[numall];
  one[0] = fneb->veng;
  one[1] = fneb->plennode;
  one[2] = fneb->nlennode;
  one[3] = fneb->gradlennode;

  if (verbose) {
    one[4] = fneb->dotpathnode;
    one[5] = fneb->dottangradnode;
    one[6] = fneb->dotgradnode;
  }

  if (output->thermo->normflag) one[0] /= atom->natoms;
  if (me == 0)
    MPI_Allgather(one,numall,MPI_DOUBLE,&all[0][0],numall,MPI_DOUBLE,roots);
  MPI_Bcast(&all[0][0],numall*nreplica,MPI_DOUBLE,0,world);

  rdist[0] = 0.0;
  for (int i = 1; i < nreplica; i++)
    rdist[i] = rdist[i-1] + all[i][1];
  double endpt = rdist[nreplica-1] = rdist[nreplica-2] + all[nreplica-2][2];
  for (int i = 1; i < nreplica; i++)
    rdist[i] /= endpt;

  // look up GradV for the initial, final, and climbing replicas
  // these are identical to fnorm2, but to be safe we
  // take them straight from fix_neb

  double gradvnorm0, gradvnorm1, gradvnormc;

  int irep;
  irep = 0;
  gradvnorm0 = all[irep][3];
  irep = nreplica-1;
  gradvnorm1 = all[irep][3];
  irep = fneb->rclimber;
  if (irep > -1) {
    gradvnormc = all[irep][3];
    ebf = all[irep][0]-all[0][0];
    ebr = all[irep][0]-all[nreplica-1][0];
  } else {
    double vmax = all[0][0];
    int top = 0;
    for (int m = 1; m < nreplica; m++)
      if (vmax < all[m][0]) {
        vmax = all[m][0];
        top = m;
      }
    irep = top;
    gradvnormc = all[irep][3];
    ebf = all[irep][0]-all[0][0];
    ebr = all[irep][0]-all[nreplica-1][0];
  }


  if (me_universe == 0) {
    const double todeg=180.0/MY_PI;
    FILE *uscreen = universe->uscreen;
    FILE *ulogfile = universe->ulogfile;
    if (uscreen) {
      fprintf(uscreen,BIGINT_FORMAT " %12.8g %12.8g ",
              update->ntimestep,fmaxreplica,fmaxatom);
      fprintf(uscreen,"%12.8g %12.8g %12.8g ",
              gradvnorm0,gradvnorm1,gradvnormc);
      fprintf(uscreen,"%12.8g %12.8g %12.8g ",ebf,ebr,endpt);
      for (int i = 0; i < nreplica; i++)
        fprintf(uscreen,"%12.8g %12.8g ",rdist[i],all[i][0]);
      if (verbose) {
        fprintf(uscreen,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                NAN,180-acos(all[0][5])*todeg,180-acos(all[0][6])*todeg,
                all[0][3],freplica[0],fmaxatomInRepl[0]);
        for (int i = 1; i < nreplica-1; i++)
          fprintf(uscreen,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                  180-acos(all[i][4])*todeg,180-acos(all[i][5])*todeg,
                  180-acos(all[i][6])*todeg,all[i][3],freplica[i],
                  fmaxatomInRepl[i]);
        fprintf(uscreen,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                NAN,180-acos(all[nreplica-1][5])*todeg,NAN,all[nreplica-1][3],
                freplica[nreplica-1],fmaxatomInRepl[nreplica-1]);
      }
      fprintf(uscreen,"\n");
    }

    if (ulogfile) {
      fprintf(ulogfile,BIGINT_FORMAT " %12.8g %12.8g ",
              update->ntimestep,fmaxreplica,fmaxatom);
      fprintf(ulogfile,"%12.8g %12.8g %12.8g ",
              gradvnorm0,gradvnorm1,gradvnormc);
      fprintf(ulogfile,"%12.8g %12.8g %12.8g ",ebf,ebr,endpt);
      for (int i = 0; i < nreplica; i++)
        fprintf(ulogfile,"%12.8g %12.8g ",rdist[i],all[i][0]);
      if (verbose) {
        fprintf(ulogfile,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                NAN,180-acos(all[0][5])*todeg,180-acos(all[0][6])*todeg,
                all[0][3],freplica[0],fmaxatomInRepl[0]);
        for (int i = 1; i < nreplica-1; i++)
          fprintf(ulogfile,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                  180-acos(all[i][4])*todeg,180-acos(all[i][5])*todeg,
                  180-acos(all[i][6])*todeg,all[i][3],freplica[i],
                  fmaxatomInRepl[i]);
        fprintf(ulogfile,"%12.5g %12.5g %12.5g %12.5g %12.5g %12.5g",
                NAN,180-acos(all[nreplica-1][5])*todeg,NAN,all[nreplica-1][3],
                freplica[nreplica-1],fmaxatomInRepl[nreplica-1]);
      }
      fprintf(ulogfile,"\n");
      fflush(ulogfile);
    }
  }
}


/* ----------------------------------------------------------------------
    readline methods for CAC/neb
------------------------------------------------------------------------- */
int NEBCAC::read_lines_from_CAC(FILE *fp, int nlines, int maxline, 
                                  int maxelement,   char *buf)
{
  int m, nlineinner, nodecount, npoly, tmp;
  char *element_type = (char*)memory->smalloc(sizeof(char) * 20, "read_lines_CAC: element type string");
  char **element_names = atom->element_names;
  int element_type_count = atom->element_type_count;
  int *nodes_per_element_list = atom->nodes_per_element_list;


  if (me == 0) {
    m = 0;
    for (int i = 0; i < nlines; i++) {
      if (!fgets(&buf[m], maxline, fp)) {
        m = 0;
        break;
      }

      sscanf(&buf[m], "%d %s %d", &tmp, element_type, &npoly);
      m += strlen(&buf[m]);
      element_type = strtok(element_type, " \t\n\r\f");

      int type_found = 0;
      for(int string_check=1; string_check < element_type_count; string_check++){
        if (strcmp(element_type, element_names[string_check]) == 0){
          type_found=1; 
          nodecount = nodes_per_element_list[string_check];
        }
      }
      if (strcmp(element_type, "Atom") == 0) {
        type_found=1;
        nodecount = 1;
        npoly = 1;
      }
      if(!type_found) {
        error->one(FLERR, "element type not yet defined, add definition in process_args function of atom_vec_CAC.cpp style");
      }
      if(npoly<1)
        error->one(FLERR, "poly_count less than one in data file");
      // read lines one at a time into buffer and count words
      // count to ninteger and ndouble until have enough lines
      //comm->size_forward = 9 * nodecount*npoly + 8 + npoly;

      for (nlineinner = 0; nlineinner < nodecount*npoly; nlineinner++) {
        if (!fgets(&buf[m], maxline, fp)){
          m = 0;
          break;
        }
        m += strlen(&buf[m]);
        if (nlineinner + 1 >= maxelement) {
          error->one(FLERR,
            "Too many lines in one element in data file - increase maxpoly or max nodes per element for atom style CAC");
        }
      }
    }
  }

  MPI_Bcast(&m, 1, MPI_INT, 0, world);
  if (m == 0) return 1;
  MPI_Bcast(buf, m, MPI_CHAR, 0, world);
  return 0;
}

int NEBCAC::read_lines_from_CAC_universe(FILE *fp, int nlines, int maxline, 
                                           int maxelement, char *buf)
{
  int m, nlineinner, nodecount, npoly, tmp;
  char* element_type = (char*)malloc(sizeof(char)*20);

  int me_universe = universe->me;
  MPI_Comm uworld = universe->uworld;

  if (me_universe == 0) {
    m = 0;
    for (int i = 0; i < nlines; i++) {
      if (!fgets(&buf[m], maxline, fp)) {
        m = 0;
        break;
      }
      sscanf(&buf[m], "%d %s %d", &tmp, element_type, &npoly);
      m += strlen(&buf[m]);
      element_type = strtok(element_type, " \t\n\r\f");
      if (strcmp(element_type, "Eight_Node") == 0) nodecount = 8;
      else if (strcmp(element_type, "Atom") == 0) {
        nodecount = 1;
        npoly = 1;
      }
      else {
        error->one(FLERR, "Unexpected element type in data file");
      }
      // read lines one at a time into buffer and count words
      // count to ninteger and ndouble until have enough lines
      //comm->size_forward = 9 * nodecount*npoly + 8 + npoly;

      for (nlineinner = 0; nlineinner < nodecount*npoly; nlineinner++) {
        if (!fgets(&buf[m], maxline, fp)){
          m = 0;
          break;
        }
        m += strlen(&buf[m]);
        if (nlineinner + 1 > maxelement) {
          error->one(FLERR,
            "Too many lines in one element in data file - increase MAXELEMENT in read_data.cpp");
        }
      }
    }
  }
  
  MPI_Bcast(&m, 1, MPI_INT, 0, uworld);
  if (m == 0) return 1;
  MPI_Bcast(buf, m, MPI_CHAR, 0, uworld);
  return 0;
}

