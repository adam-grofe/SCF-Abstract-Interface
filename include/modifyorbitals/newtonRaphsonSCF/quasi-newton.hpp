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

#include <cqlinalg/blasutil.hpp>

namespace ChronusQ {

/*
 * Brief: Computes the full Newton-Raphson Step
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::fullNRStep() {

  if( this->modOrbOpt.computeFullNRStep ) {
    MatsT* dx = this->memManager.template malloc<MatsT>(nParam);
    this->modOrbOpt.computeFullNRStep(dx);
    takeStep(dx);
    this->memManager.free(dx);
  } else {
    CErr("Full Newton-Raphson Step for this method is not yet implemented");
  }
};

/*
 * Brief: Computest the Broyden-Fletcher-Goldfarb-Shanno quasi-Newton step
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::qnBFGSStep() {

  qnSetup();

  size_t nExtrap = std::min(this->scfConv.nSCFIter + 1, this->scfControls.nKeep);
  MatsT* dx      = this->memManager.template malloc<MatsT>(nParam);
  computeBFGS(nExtrap, qnOrbRot, qnOrbGrad, dx);
  takeStep(dx);
  this->memManager.free(dx);
};

/*
 * Brief: Computes the Symmetric Rank 1 quasi-Newton step
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::qnSR1Step() {

  qnSetup();

  size_t nExtrap = std::min(this->scfConv.nSCFIter + 1, this->scfControls.nKeep);
  MatsT* dx      = this->memManager.template malloc<MatsT>(nParam);
  computeSR1(nExtrap, qnOrbRot, qnOrbGrad, dx);
  takeStep(dx);
  this->memManager.free(dx);
};

/*
 * Brief: Copy the orbital parameter and gradient to QN data
 *        structures and place them in the correct order for
 *        the quasi-Newton functions
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::qnSetup() {

  size_t iQN = this->scfConv.nSCFIter % this->scfControls.nKeep;
  if( this->scfConv.nSCFIter < this->scfControls.nKeep ) {
    // Copy Gradient to vector
    std::copy_n(orbGrad, nParam, qnOrbGrad[iQN]);
    std::copy_n(orbRot, nParam, qnOrbRot[iQN]);
  } else {
    // Shift pointers to the correct order
    size_t nKeep    = this->scfControls.nKeep;
    MatsT* lastGrad = qnOrbGrad[0];
    MatsT* lastRot  = qnOrbRot[0];
    for( size_t i = 1; i < nKeep; i++ ) {
      qnOrbGrad[i - 1] = qnOrbGrad[i];
      qnOrbRot[i - 1]  = qnOrbRot[i];
    }
    qnOrbGrad[nKeep - 1] = lastGrad;
    qnOrbRot[nKeep - 1]  = lastRot;

    // Copy new data to elements
    std::copy_n(orbGrad, nParam, qnOrbGrad[nKeep - 1]);
    std::copy_n(orbRot, nParam, qnOrbRot[nKeep - 1]);
  }
};

/*
 *   Brief: This function computes the limited memory BFGS step using the two
 *          loop recursion algorithm.
 *           Nocedal, J. Updating Quasi-Newton Matrices with Limited Storage. Math. Comput. 1980, 35 (151), 773–773.
 *
 *   Note: the order of the vectors should be least recent to most recent with the current parameters
 *         as the last element for both the x and g vectors
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::computeBFGS(size_t N, std::vector<MatsT*> x, std::vector<MatsT*> g, MatsT* dx) {

  // If only 1 iteration, perform gradient descent
  if( N == 1 ) {
    std::copy_n(g[0], nParam, dx);
    return;
  }

  // Compute s and y
  std::vector<MatsT*> s;   // s_k = x_k+1 - x_k
  std::vector<MatsT*> y;   // y_k = g_k+1 - g_k
  for( size_t i = 0; i < N - 1; i++ ) {
    s.emplace_back(this->memManager.template malloc<MatsT>(nParam));
    y.emplace_back(this->memManager.template malloc<MatsT>(nParam));
    std::copy_n(x[i + 1], nParam, s[i]);
    std::copy_n(g[i + 1], nParam, y[i]);
    blas::axpy(nParam, MatsT(-1.), x[i], 1, s[i], 1);
    blas::axpy(nParam, MatsT(-1.), g[i], 1, y[i], 1);
#ifdef _NRSCF_DEBUG_BFGS
    printParamVec("s", s[i]);
    printParamVec("y", y[i]);
#endif
  }
  std::copy_n(g[N - 1], nParam, dx);

  // Compute rho_i = 1. / y_i*s_i
  std::vector<MatsT> rho(N - 1, MatsT(0.));
  std::vector<MatsT> alpha(N - 1, MatsT(0.));
  MatsT SCR = MatsT(0.);
  for( size_t i = 0; i < N - 1; i++ ) {
    size_t k = (N - 2) - i;
    rho[k] = MatsT(1.) / blas::dot(nParam,y[k],1,s[k],1);
    alpha[k] = rho[k] * blas::dot(nParam,s[k],1,dx,1);
    blas::axpy(nParam, -alpha[k], y[k], 1, dx, 1);
  }

  for( size_t i=0; i<nParam; ++i )
      dx[i] /= orbDiagHess[i];

  MatsT beta = MatsT(0.);
  for( size_t i = 0; i < N - 1; i++ ) {
    beta = rho[i] * blas::dot(nParam,y[i],1, dx,1);
    blas::axpy(nParam, alpha[i] - beta, s[i], 1, dx, 1);
  }

#ifdef _NRSCF_DEBUG_BFGS
  printParamVec("dx Final", dx);
#endif

  // Clean up memory
  for( auto* sP : s )
    this->memManager.free(sP);
  for( auto* yP : y )
    this->memManager.free(yP);
};


/*
 *   Brief: This function computes the limited memory SR1 step.
 *          Byrd, R. H. Representations of Quasi-Newton Matrices and Their Use in Limited Memory Methods. Math. Program. 1994, 63, 129–156.
 *
 *   Note: the order of the vectors should be least recent to most recent with the current parameters
 *         as the last element for both the x and g vectors
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::computeSR1(size_t N, std::vector<MatsT*> x, std::vector<MatsT*> g, MatsT* dx) {

  // If only 1 iteration, perform gradient descent
  if( N == 1 ) {
    std::copy_n(g[0], nParam, dx);
#ifdef _NRSCF_DEBUG_SR1
    printParamVec("dx Final", dx);
#endif
    return;
  }

  // Compute S and Y Matrices
  size_t nUpdate = N - 1;
  MatsT* s       = this->memManager.template malloc<MatsT>(nParam * nUpdate);
  MatsT* y       = this->memManager.template malloc<MatsT>(nParam * nUpdate);
  size_t disp    = 0;
  for( size_t i = 0; i < nUpdate; i++ ) {
    std::copy_n(x[i + 1], nParam, s + disp);
    std::copy_n(g[i + 1], nParam, y + disp);
    blas::axpy(nParam, MatsT(-1.), x[i], 1, s + disp, 1);
    blas::axpy(nParam, MatsT(-1.), g[i], 1, y + disp, 1);
#ifdef _NRSCF_DEBUG_SR1
    printParamVec("s", s + disp);
    printParamVec("y", y + disp);
#endif
    disp += nParam;
  }

  // Compute S - H_0 Y
  MatsT* SCR = this->memManager.template malloc<MatsT>(nParam*nUpdate);
  for( size_t i=0; i<nUpdate; ++i )
    for( size_t j=0; j<nParam; ++j ) 
        SCR[j + i*nParam] = y[j + i*nParam] / orbDiagHess[j];
  MatsT* shy = this->memManager.template malloc<MatsT>(nParam * nUpdate);
  std::copy_n(s, nParam * nUpdate, shy);
  blas::axpy(nParam * nUpdate, MatsT(-1.), SCR, 1, shy, 1);

  // Initialize dx = H_0 g_k
  for( size_t i=0; i<nParam; ++i )
      dx[i] = g[N-1][i] / orbDiagHess[i];

  // Compute (R - Y^T H_0 Y)
  MatsT* r = this->memManager.template malloc<MatsT>(nUpdate * nUpdate);
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, nUpdate, nUpdate, nParam, MatsT(1.), s, nParam, y, nParam, MatsT(0.), r, nUpdate);
  if( nUpdate > 1){
	  for( size_t i=0; i<nUpdate-1; i++ )
		  for( size_t j=1; j<nUpdate; j++ )
			  r[j+i*nUpdate] = SmartConj(r[i + j*nUpdate]);
  }
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, nUpdate, nUpdate, nParam, MatsT(-1.), y, nParam, SCR, nParam, MatsT(1.), r, nUpdate);

#ifdef _NRSCF_DEBUG_SR1
  prettyPrintSmart(std::cout, "R", r, nUpdate, nUpdate, nUpdate);
  SquareMatrix<MatsT> rCopy(this->memManager, nUpdate);
  std::copy_n(r, nUpdate * nUpdate, rCopy.pointer());
#endif

  double normSHY = blas::nrm2(nParam, shy + (nUpdate - 1) * nParam, 1);
  computeMatrixInverse(nUpdate, r, nUpdate, normSHY);

#ifdef _NRSCF_DEBUG_SR1
  std::cout << "Magnitude of (S - H Y) = " << normSHY << std::endl;
  prettyPrintSmart(std::cout, "R Inverse", r, nUpdate, nUpdate, nUpdate);
#endif

  // Compute shy * r * shy^T g
  MatsT* vec1 = this->memManager.template malloc<MatsT>(nUpdate);
  MatsT* vec2 = this->memManager.template malloc<MatsT>(nUpdate);
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, nUpdate, 1, nParam, MatsT(1.), shy, nParam, g[N - 1], nParam, MatsT(0.), vec1, nUpdate);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nUpdate, 1, nUpdate, MatsT(1.), r, nUpdate, vec1, nUpdate, MatsT(0.), vec2, nUpdate);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, nParam, 1, nUpdate, MatsT(1.), shy, nParam, vec2, nUpdate, MatsT(1.), dx, nParam);

#ifdef _NRSCF_DEBUG_SR1
  printParamVec("dx Final", dx);
#endif

  this->memManager.free(s, y, shy, r, vec1, vec2, SCR);
};

/*
 *   Brief: This function computes an approximate matrix inverse. If the matrix
 *          is singular then it approximates the inverse by removing
 *          the singular columns from the matrix. Result is stored
 *          in place (A). Num is used to compare if the vectors are
 *          singular compared to the magnitude of the update.
 */
template<typename MatsT>
void NewtonRaphsonSCF<MatsT>::computeMatrixInverse(const size_t N, MatsT* A, const size_t LDA, const double num) {

  MatsT* ACopy = this->memManager.template malloc<MatsT>(N * N);
  std::copy_n(A, N * N, ACopy);

  SetMat('N', N, N, MatsT(1.), ACopy, N, A, LDA);

  // Compute SVD to determine which vectors are singular
  SquareMatrix<MatsT> U(this->memManager, N);
  SquareMatrix<MatsT> VT(this->memManager, N);
  double* sV = this->memManager.template malloc<double>(N);

  int info = lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, N, N, A, LDA, 
          sV, U.pointer(), N, VT.pointer(), N);
  if( info != 0 ) CErr("SVD Failed in Newton-RaphsonSCF");

#ifdef _NRSCF_DEBUG_SR1
    std::cout << "Singular Values:" << std::endl;
    for( size_t i = 0; i < N; i++ ) {
      std::cout << std::setw(18) << std::scientific << std::right << sV[i];
      if( (i + 1) % 5 == 0 ) std::cout << std::endl;
    }
    std::cout << std::endl;
#endif

  // Zero out vectors that are singular or scale by inverse sing. value
  for( size_t i = 0; i < N; i++ ) {
    if( (sV[i] / num) > 1.E-10 ) {
      blas::scal(N, MatsT(1. / sV[i]), U.pointer() + i * N, 1);
    } else {
      blas::scal(M, MatsT(0.), U.pointer() + i * N, 1);
      blas::scal(M, MatsT(0.), VT.pointer() + i * N, 1);
    }
  }

  // Compute Matrix Inverse
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, N, N, MatsT(1.), U.pointer(), N, VT.pointer(), N, MatsT(0.), A, LDA);


  this->memManager.free(ACopy,sV);
}


/*
 * Brief: Function to print out the parameter vectors for debugging
 */
template<typename MatsT>
template<typename MatsU>
void NewtonRaphsonSCF<MatsT>::printParamVec(std::string title, MatsU* vec) {
  std::cout << title << ":" << std::endl;
  for( size_t i = 0; i < nParam; i++ ) {
    MatsU val = std::abs(vec[i]) > 1E-10 ? vec[i] : MatsU(0.);
    std::cout << std::setw(25) << std::scientific << std::setprecision(3) << std::right << val;
    if( (i + 1) % 5 == 0 ) std::cout << std::endl;
  }
  std::cout << std::endl << std::endl;
}

};   // namespace ChronusQ
