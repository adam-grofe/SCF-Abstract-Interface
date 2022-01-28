/*
 *  This file is part of the Chronus Quantum (ChronusQ) software package
 *
 *  Copyright (C) 2014-2020 Li Research Group (University of Washington)
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 *  Contact the Developers:
 *    E-Mail: xsli@uw.edu
 *
 */
#pragma once

#include <modifyorbitals/optOrbitals.hpp>

//#define _NRSCF_DEBUG
//#define _NRSCF_DEBUG_UNITARY
//#define _NRSCF_DEBUG_BFGS
//#define _NRSCF_DEBUG_SR1
//#define _NRSCF_PRINT_MOS

namespace ChronusQ {

template<typename MatsT>
class NewtonRaphsonSCF : public OptimizeOrbitals<MatsT> {
protected:
  size_t nParam;                  ///< Total number of independent parameters
  bool refMOsAllocated = false;   ///< Whether the reference MOs were allocated yet
  bool storedRef       = false;   ///< Save the MO's on the first iteration
  MatsT* orbRot;                  ///< Orbital Rotation Parameters
  MatsT* orbGrad;                 ///< Orbital Gradient
  MatsT* orbDiagHess;             ///< Diagonal Hessian Approximation
  std::vector<SquareMatrix<MatsT>> refMO;   ///< The reference set of MO's that are being rotated

  // Quasi-Newton Data Structures
  std::vector<MatsT*> qnOrbRot;             ///< The Set of variable iterates for Quasi-Newton
  std::vector<MatsT*> qnOrbGrad;            ///< The set of gradients for Quasi-Newton

public:
  // Constructor
  NewtonRaphsonSCF() = delete;
  NewtonRaphsonSCF(SCFControls sC, MPI_Comm comm, ModifyOrbitalsOptions<MatsT> modOpt, CQMemManager& mem):
    OptimizeOrbitals<MatsT>(sC, comm, modOpt, mem) {

    // Sanity Check on size of vectors
    std::vector<std::shared_ptr<SquareMatrix<MatsT>>> fock = this->modOrbOpt.getFock();
    if( fock.size() != this->modOrbOpt.nOcc.size() ) CErr("The size of ModifyOrbitalsOptions is not the same as the Fock Matrix");

    // Compute the total number of nonredundant parameters
    nParam = 0;
    for( size_t i = 0; i < this->modOrbOpt.nOcc.size(); i++ ) {
      size_t NBC = fock[i]->dimension();
      size_t NB  = this->modOrbOpt.nC[i] == 4 ? NBC / 2 : NBC;

      size_t nVirt = NB - this->modOrbOpt.nOcc[i];
      nParam += this->modOrbOpt.nOcc[i] * nVirt;
    }

    // Alloc Quasi-Newton matrices
    alloc();
  }

  // Destructor
  ~NewtonRaphsonSCF() {
    this->memManager.free(orbRot, orbGrad, orbDiagHess);

    if( this->scfControls.nrAlg == QUASI_BFGS or this->scfControls.nrAlg == QUASI_SR1 ) {
      for( auto* p : qnOrbRot )
        this->memManager.free(p);
      for( auto* p : qnOrbGrad )
        this->memManager.free(p);
    }
  }

  // Functions
  void alloc();
  void getNewOrbitals(EMPerturbation&, std::vector<std::reference_wrapper<SquareMatrix<MatsT>>>, std::vector<double*>);
  void NewtonRaphsonIteration();
  void printRunHeader(std::ostream&, EMPerturbation&);

  // Newton-Raphson Approximation Steps
  void fullNRStep();
  void qnBFGSStep();
  void qnSR1Step();

  // Common Functions
  void computeEigenvalues(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>>, std::vector<double*>);
  void computeGradient(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>>, std::vector<double*>);
  void rotateMOs(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>>);
  void saveRefMOs(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo);
  SquareMatrix<MatsT> MatExpT(MatsT*, size_t, size_t, size_t);

  // Line Search and step functions
  void takeStep(MatsT*);

  // Quasi-Newton Functions
  void qnSetup();
  void computeBFGS(size_t N, std::vector<MatsT*> x, std::vector<MatsT*> g, MatsT* dx);
  void computeSR1(size_t N, std::vector<MatsT*> x, std::vector<MatsT*> g, MatsT* dx);
  void computeMatrixInverse(const size_t, MatsT*, const size_t, const double);

  template<typename MatsU>
  void printParamVec(std::string, MatsU*);

  // Convergence Testing functions
  double computeFDCConv();
};
};   // namespace ChronusQ
