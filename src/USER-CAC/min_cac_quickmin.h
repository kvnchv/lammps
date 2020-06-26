/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef MINIMIZE_CLASS

MinimizeStyle(cac/quickmin,CACMinQuickMin)

#else

#ifndef LMP_CAC_MIN_QUICKMIN_H
#define LMP_CAC_MIN_QUICKMIN_H

#include "min_cac.h"

namespace LAMMPS_NS {

class CACMinQuickMin : public CACMin {
 public:
  CACMinQuickMin(class LAMMPS *);
  ~CACMinQuickMin() {}
  void init();
  void setup_style();
  void reset_vectors();
  int iterate(int);
 protected:
  int densemax;               // bounds arrays size for continuous x,v,f nodal arrays

  virtual void copy_vectors();
  virtual void copy_force();
 private:
  double dt;
  bigint last_negative;
};

}

#endif
#endif