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

#include <modifyorbitals/newtonRaphsonSCF/quasi-newton.hpp>

namespace ChronusQ {

/**
 *  \Brief: Computes the fock Matrix and then computes a new set of orbitals
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::getNewOrbitals(EMPerturbation& pert, std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo,
                                             std::vector<double*> eps) {

  // Form the Fock matrix D(k) -> F(k)
  ProgramTimer::timeOp("Form Fock", [&]() { this->modOrbOpt.formFock(pert); });

  if( not storedRef ) saveRefMOs(mo);

  computeGradient(mo, eps);

  // Compute NR step and update orbRot
  NewtonRaphsonIteration();

  // Rotate reference MO's to new MO's
  rotateMOs(mo);

  // Compute new Eigenvalues
  computeEigenvalues(mo, eps);

  this->modOrbOpt.formDensity();
}

/*
 * Brief: Saves the initial set of orbitals to be used as the
 *        reference for the unitary transformation
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::saveRefMOs(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo) {

  // Store Reference MO's from which to compute rotations
  size_t nMO = mo.size();

  // Allocate Matrices
  if( not refMOsAllocated ) {
    refMO.reserve(nMO);
    for( size_t i = 0; i < nMO; i++ ) {
      size_t NBC = mo[i].get().dimension();
      refMO.emplace_back(this->memManager, NBC);
    }
  }

  // Copy MO's
  for( size_t i = 0; i < nMO; i++ ) {
    size_t NBC = mo[i].get().dimension();
    std::copy_n(mo[i].get().pointer(), NBC * NBC, refMO[i].pointer());
  }

  storedRef = true;
}

/*
 *   Brief: Function to choose which algorithm/approximation is used
 *          for the Newton step.
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::NewtonRaphsonIteration() {
  if( this->scfControls.nrAlg == FULL_NR ) {

    fullNRStep();

  } else if( this->scfControls.nrAlg == QUASI_BFGS ) {

    qnBFGSStep();

  } else if( this->scfControls.nrAlg == QUASI_SR1 ) {

    qnSR1Step();

  } else if( this->scfControls.nrAlg == GRAD_DESCENT ) {

    MatsT* dx = this->memManager.template malloc<MatsT>(nParam);
    for( size_t i=0; i<nParam; ++i )
        dx[i] = orbGrad[i] / orbDiagHess[i];
    takeStep(dx);
    this->memManager.free(dx);

  } else {

    CErr("Requested Newton-Raphson Step not yet implemented");

  }
}

/*
 *     Brief: Rotate the vector of MO's using the computed OrbRot parameters
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::rotateMOs(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo) {

  size_t disp = 0;
  for( size_t i = 0; i < mo.size(); i++ ) {
    size_t nOcc  = this->modOrbOpt.nOcc[i];
    size_t NBC   = mo[i].get().dimension();
    size_t NB    = this->modOrbOpt.nC[i] == 4 ? NBC / 2 : NBC;
    size_t nVirt = NB - nOcc;
    size_t nP    = nOcc * nVirt;

    // FIXME: Use Implemented Matrix Exponential
    SquareMatrix<MatsT> U = MatExpT(orbRot + disp, nP, nOcc, NB);
    SquareMatrix<MatsT> SCR(this->memManager, NBC);

#ifdef _NRSCF_DEBUG_UNITARY
    SCR.clear();
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, NB, NB, NB, MatsT(1.), U.pointer(), NB, U.pointer(), NB, MatsT(0.), SCR.pointer(), NBC);
    for( size_t p = 0; p < NB; p++ )
      SCR(p, p) -= MatsT(1.);
    double sum = 0.;
    for( size_t p = 0; p < NBC * NBC; p++ )
      sum += std::abs(SCR.pointer()[p]);
    std::cerr << "Unitary Test:  U^H * U = 1 -> Error =  " << sum << std::endl;
#endif

    // Rotate MO's
    size_t shift = this->modOrbOpt.nC[i] == 4 ? NB * NBC : 0;   // Shift to positive energy states
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, NBC, NB, NB, MatsT(1.), refMO[i].pointer() + shift, NBC, U.pointer(), NB, MatsT(0.), SCR.pointer(), NBC);

    // Copy Rotated MO's
    std::copy_n(SCR.pointer(), NBC * NB, mo[i].get().pointer() + shift);

#ifdef _NRSCF_PRINT_MOS
    prettyPrintSmart(std::cout, "MO " + std::to_string(i), mo[i].get().pointer(), NBC, NBC, NBC);
#endif

    disp += nP;
  }
}

/*
 * Matrix Exponential Using Taylor Expansion
 */
template<typename MatsT>
SquareMatrix<MatsT> NewtonRaphsonSCF<MatsT>::MatExpT(MatsT* xRot, size_t nRot, size_t nOcc, size_t NB) {

  SquareMatrix<MatsT> ExpA(this->memManager, NB);
  SquareMatrix<MatsT> A(this->memManager, NB);
  A.clear();

  // Assemble the Anti-symmetric Matrix
  size_t nVirt = NB - nOcc;
  for( size_t iOcc = 0; iOcc < nOcc; iOcc++ )
    for( size_t iVirt = 0; iVirt < nVirt; iVirt++ ) {
      A(iOcc, iVirt + nOcc) = -xRot[iVirt + iOcc*nVirt];
      A(iVirt + nOcc, iOcc) = SmartConj(xRot[iVirt + iOcc*nVirt]);
    }

  // allocate memory
  size_t N2        = NB * NB;
  MatsT* OddTerms  = this->memManager.template malloc<MatsT>(N2);
  MatsT* EvenTerms = this->memManager.template malloc<MatsT>(N2);

  std::fill_n(ExpA.pointer(), N2, MatsT(0.));
  // form zero order term
  for( auto i = 0ul; i < NB; i++ )
    ExpA(i, i) = MatsT(1.);

  // form 1st order term
  // TODO: add Alpha scaling
  std::copy_n(A.pointer(), N2, OddTerms);

  double residue;
  double small_number = std::numeric_limits<double>::epsilon();
  bool converged      = false;
  size_t maxIter      = 200;
  for( auto iter = 0ul; iter < maxIter; iter += 2 ) {
    // add the odd term
    MatAdd('N', 'N', NB, NB, MatsT(1.), OddTerms, NB, MatsT(1.), ExpA.pointer(), NB, ExpA.pointer(), NB);

    // residue = InnerProd<double>(N2, OddTerms, 1, OddTerms, 1);
    //residue = MatNorm<double>('F', NB, NB, OddTerms, NB);
    residue = blas::nrm2(NB*NB, OddTerms, 1);
    if( residue <= small_number ) {
      converged = true;
      break;
    }

    // form and add next even term
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, NB, NB, NB, MatsT(1. / (iter + 2)), A.pointer(), NB, OddTerms, NB, MatsT(0.), EvenTerms, NB);
    MatAdd('N', 'N', NB, NB, MatsT(1.), EvenTerms, NB, MatsT(1.), ExpA.pointer(), NB, ExpA.pointer(), NB);

    // residue = InnerProd<double>(N2, EvenTerms, 1, EvenTerms, 1);
    //residue = MatNorm<double>('F', NB, NB, EvenTerms, NB);
    residue = blas::nrm2(NB*NB,EvenTerms,1);
    if( residue <= small_number ) {
      converged = true;
      break;
    }

    // form next odd term
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, NB, NB, NB, MatsT(1. / (iter + 3)), A.pointer(), NB, EvenTerms, NB, MatsT(0.), OddTerms, NB);
  }

  this->memManager.free(OddTerms, EvenTerms);

  if( not converged ) CErr("MatExpT failed to converge.");

  return ExpA;
};
// NewtonRaphsonSCF::MatExpT


/*
 *  Brief: Computes the orbital energies from a set of MO's
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::computeEigenvalues(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo, std::vector<double*> eps) {

  std::vector<std::shared_ptr<SquareMatrix<MatsT>>> fock = this->modOrbOpt.getFock();

  for( size_t i = 0; i < mo.size(); i++ ) {
    size_t NB = fock[i]->dimension();

    SquareMatrix<MatsT> moFock = fock[i]->transform('N', mo[i].get().pointer(), NB, NB);
    for( size_t a = 0; a < NB; a++ )
      eps[i][a] = std::real(moFock(a, a));
  }
};

/*
 * Brief: This computes the new set of parameters from 
 *        a given search direction (dx)
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::takeStep(MatsT* dx) {

  // Compute norm of step
  double norm = blas::nrm2(nParam, dx, 1);
#ifdef _NRSCF_DEBUG
  std::cout << "Search Direction Norm = " << norm << std::endl;
  printParamVec("Search Direction", dx);
#endif

  double scale = norm > this->scfControls.nrTrust ? this->scfControls.nrTrust / norm : 1.;

  // Perform Damping
  if( this->scfControls.doDamp )
    scale *= double(1. - this->scfControls.dampParam);

  // take Step
  blas::axpy(nParam, MatsT(-scale), dx, 1, orbRot, 1);
};


/*
 *  Brief: Computes the orbital gradients and the diagonal Hessian and stores
 *         them in their vectors. The total gradient is the concatonation of
 *         orbital gradients for each Fock matrix. e.g. UHF->[Grad_alpha,Grad_beta]
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::computeGradient(std::vector<std::reference_wrapper<SquareMatrix<MatsT>>> mo, std::vector<double*> eps) {

  // Compute the Gradient of orbital rotations
  if( this->modOrbOpt.computeNROrbGrad ) {
    // Compute user-defined gradient
    this->modOrbOpt.computeNROrbGrad(orbGrad);
  } else {
    // Compute default gradient from fock and mo's
    size_t disp = 0;

    std::vector<std::shared_ptr<SquareMatrix<MatsT>>> fock = this->modOrbOpt.getFock();
    for( size_t i = 0; i < mo.size(); i++ ) {
      size_t nOcc  = this->modOrbOpt.nOcc[i];
      size_t NBC   = fock[i]->dimension();
      size_t NB    = this->modOrbOpt.nC[i] == 4 ? NBC / 2 : NBC;
      size_t nVirt = NB - nOcc;

      if( NBC != mo[i].get().dimension() ) CErr("MO and Fock Matrix are different dimensions");

      SquareMatrix<MatsT> SCR(this->memManager, NBC);
      size_t shift = this->modOrbOpt.nC[i] == 4 ? NB * NBC : 0;
      blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, 
              nVirt, NBC, NBC, 
              MatsT(4.), mo[i].get().pointer()+shift+nOcc*NBC, NBC, 
              fock[i]->pointer(), NBC, 
              MatsT(0.), SCR.pointer(), NBC);

      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, 
              nVirt, nOcc, NBC, 
              MatsT(1.), SCR.pointer(), NBC, 
              mo[i].get().pointer() + shift, NBC, 
              MatsT(0.), orbGrad + disp, nVirt);


      // Precondition gradient with diagonal elements
      shift = this->modOrbOpt.nC[i] == 4 ? NB : 0;
      for( size_t iVirt = 0; iVirt < nVirt; ++iVirt ) {
        double virtEPS = eps[i][iVirt + nOcc + shift] + this->scfControls.nrLevelShift;
        for( size_t iOcc = 0; iOcc < nOcc; ++iOcc ) {
          double freq = virtEPS - eps[i][iOcc + shift];
          if( std::abs(freq) < 1.0E-5 ) freq = 1.0E-5;
          orbDiagHess[iVirt + iOcc*nVirt + disp] = MatsT(4.*freq);
        }
      }
      disp += nOcc * nVirt;
    }

    // Conjugate the gradient for complex wave functions.
    // this is necessary to make quasi-newton and gradient descent
    // consistent with the full NR implementation
    if( std::is_same<MatsT, dcomplex>::value ){
      for(size_t iP=0; iP<nParam; ++iP)
          orbGrad[iP] = SmartConj(orbGrad[iP]);
    }

#ifdef _NRSCF_DEBUG
    printParamVec("Orbital Rotation Parameters", orbRot);
    printParamVec("Computed Gradient", orbGrad);
    printParamVec("Diagonal Orbital Hessian Approximation", orbDiagHess);
#endif
  } // if( computeNROrbGrad ) else
};
// NewtonRaphsonSCF<MatsT> :: computeGradient

/*
 *  Brief: Computes the gradient convergence criteria for optimization
 *         In this case it is the maximum element of the orbital gradient
 */
template<typename MatsT>
double NewtonRaphsonSCF<MatsT>::computeFDCConv() {
  // Compute the Maximum of the gradient or norm of gradient
  double maxGrad = 0.;
  for( size_t i = 0; i < nParam; ++i )
    if( maxGrad < std::abs(orbGrad[i]) ) maxGrad = std::abs(orbGrad[i]);
  return maxGrad;
};
// NewtonRaphsonSCF<MatsT> :: computeFDCConv

/*
 *  Brief: Print Algorithm specific information at the end of the run header
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::printRunHeader(std::ostream& out, EMPerturbation& pert) {
  OptimizeOrbitals<MatsT>::printRunHeader(out, pert);

  // Print DIIS Algorithm info
  if( this->scfControls.nrAlg == GRAD_DESCENT ) {
    out << std::setw(38) << std::left << "  Newton-Raphson Step Approximation: Gradient-Descent" << std::endl;
  } else if( this->scfControls.nrAlg == QUASI_BFGS ) {
    out << std::setw(38) << std::left << "  Newton-Raphson Step Approximation: Quasi-Newton Broyden-Fletcher-Goldfarb-Shanno" << std::endl;
    out << std::left << "      Saving " << this->scfControls.nKeep << " previous iterations for Quasi-Newton" << std::endl;
  } else if( this->scfControls.nrAlg == QUASI_SR1 ) {
    out << std::setw(38) << std::left << "  Newton-Raphson Step Approximation: Quasi-Newton Symmetric Rank One" << std::endl;
    out << std::left << "      Saving " << this->scfControls.nKeep << " previous iterations for Quasi-Newton" << std::endl;
  } else if( this->scfControls.nrAlg == FULL_NR ) {
    out << std::setw(38) << std::left << "  Computing Full Hessian Matrix" << std::endl;
  }
};

/*
 *  Brief: Allocates the parameter vectors
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::alloc() {
  orbRot  = this->memManager.template malloc<MatsT>(nParam);
  orbGrad = this->memManager.template malloc<MatsT>(nParam);
  orbDiagHess = this->memManager.template malloc<MatsT>(nParam);
  std::fill_n(orbRot, nParam, MatsT(0.));
  std::fill_n(orbGrad, nParam, MatsT(0.));
  std::fill_n(orbDiagHess, nParam, MatsT(0.));
  if( this->scfControls.nrAlg == QUASI_BFGS or this->scfControls.nrAlg == QUASI_SR1 ) {
    for( size_t i = 0; i < this->scfControls.nKeep; i++ ) {
      qnOrbRot.emplace_back(this->memManager.template malloc<MatsT>(nParam));
      qnOrbGrad.emplace_back(this->memManager.template malloc<MatsT>(nParam));
      std::fill_n(qnOrbRot[i], nParam, MatsT(0.));
      std::fill_n(qnOrbGrad[i], nParam, MatsT(0.));
    }
  }
};

};   // namespace ChronusQ
// namespace ChronusQ
