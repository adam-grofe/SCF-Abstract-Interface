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

#include <modifyorbitals.hpp>

namespace ChronusQ {

struct SCFConvergence {

    double deltaEnergy;   ///< Convergence of Energy
    double RMSDen;        ///< RMS change in Scalar density
    // double RMSDenScalar;   ///< RMS change in Scalar density
    // double RMSDenMag;      ///< RMS change in magnetization (X,Y,Z) density
    double nrmFDC;   ///< 2-Norm of [F,D]
    double maxFDC;   ///< Maximum element of [F,D]
    double prevEnergy;  ///< Previous Energy to test convergence

    size_t nSCFIter      = 0;   ///< Number of SCF Iterations
    size_t nSCFMacroIter = 0;   ///< Number of macro SCF iteration in NEO-SCF

};   // SCFConvergence struct

/*
 *     Brief: Object to holds the overlapping functions for orbital optimization
 *            such as printing the header and iteration progress. Thus, getNewOrbitals
 *            is modified depending on which step is used. ie. SCF or Newton-Raphson
 */
template<typename MatsT>
class OptimizeOrbitals : public ModifyOrbitals<MatsT> {
  protected:
    std::vector<SquareMatrix<MatsT>> prevOnePDM;   ///< Previous density used to test convergence

  public:
    SCFControls scfControls;
    SCFConvergence scfConv;
    double nTotalElectrons = 0.;

    // Constructor
    OptimizeOrbitals(SCFControls sC, MPI_Comm comm, ModifyOrbitalsOptions<MatsT> modOpt, CQMemManager& mem):
        scfControls(sC), ModifyOrbitals<MatsT>(comm, modOpt, mem) {

        // Allocate prevOnePDM
        std::vector<std::shared_ptr<SquareMatrix<MatsT>>> den = this->modOrbOpt.getOnePDM();
        for( size_t a = 0; a < den.size(); a++ ) {
            prevOnePDM.emplace_back(den[a]->memManager(), den[a]->dimension());
            prevOnePDM[a] = *den[a];
        }

        for(auto &i : this->modOrbOpt.nOcc )
            nTotalElectrons += double(i);
    };

    // Destructor
    ~OptimizeOrbitals() {};

    // Perform an SCF procedure (see include/singleslater/scf.hpp for docs)
    void runModifyOrbitals(EMPerturbation&, std::vector<std::reference_wrapper<SquareMatrix<MatsT>>>, std::vector<double*>);

    // Evaluate convergence
    bool evalProgress(EMPerturbation&);
    double computeDensityConv();
    virtual double computeFDCConv() { return 0.; };

    //   Print SCF header, footer and progress
    void printRunHeader(std::ostream& out, EMPerturbation&);
    void printHeaderFinal(std::ostream& out);
    void printIteration(std::ostream& out = std::cout, bool printDiff = true);
};
};   // namespace ChronusQ
