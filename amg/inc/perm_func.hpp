#pragma once
#ifndef _PERM_FUNC_HPP
#define _PERM_FUNC_HPP

#include <mfem.hpp>

namespace saamge
{

class PermeabilityCoefficient : public mfem::VectorCoefficient
{
public:
    enum SliceOrientation {NONE, XY, XZ, YZ};

    /**
       @brief MFEM Coefficient constructed from permeability data set
       @param comm MPI communicator
       @param fileName file name of permeability data set
       @param N number of data in each direction intended to store from data set
       @param max_N data set size in each direction.
              For SPE10, it should be {60, 220, 85}
       @param h element size in each direction
       @param orientation if NONE (default), full data set will be read (3D);
              otherwise, it tells which 2D plane {XY, XZ, or YZ} to read
       @param slice which slice of the selected 2D plane to read
    */
    PermeabilityCoefficient(MPI_Comm comm,
                            const std::string &fileName,
                            const mfem::Array<int> &N,
                            const mfem::Array<int> &max_N,
                            const mfem::Vector &h,
                            SliceOrientation orientation=NONE,
                            int slice=-1);

    /**
       This fakes a uniform anisotropic field,
       with given horizontal and vertical permeabilities.
       (the typical use case they are more like *covariances* than
       permeabilities, but this class does not know about that.
    */
    PermeabilityCoefficient(MPI_Comm comm,
                            const mfem::Array<int> &N,
                            const mfem::Vector &h,
                            double horizontal_perm,
                            double vertical_perm);

    virtual void Eval(mfem::Vector &V, mfem::ElementTransformation &T,
                      const mfem::IntegrationPoint &ip)
    {
        mfem::Vector transip(vdim);
        T.Transform(ip, transip);
        Permeability(transip, V);
    }

    /// Frobenius norm of permeability
    double FroNorm(const mfem::Vector &x);

    std::vector<double> &GetRawPermeability() { return permeability_; }

private:
    void ReadPermeabilityFile(const std::string &fileName,
                              const mfem::Array<int> &max_N);
    void ReadPermeabilityFile(MPI_Comm comm,
                              const std::string &fileName,
                              const mfem::Array<int> &max_N);
    void BlankPermeability();
    void Permeability(const mfem::Vector &x, mfem::Vector &val);

    mfem::Array<int> N_;
    mfem::Vector h_;
    int slice_;
    SliceOrientation orientation_;

    int N_slice_;
    int N_all_;
    std::vector<double> permeability_;
};

} // namespace saamge

#endif // _PERM_FUNC_HPP
