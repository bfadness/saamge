#pragma once
#ifndef _SPE10_HPP
#define _SPE10_HPP

#include <array>
#include <mfem.hpp>
#include <string>
#include <vector>

using namespace mfem;

namespace saamge
{

class PermeabilityCoefficient : public Coefficient
{
public:
    enum SliceOrientation {NONE, XY, XZ, YZ};

    PermeabilityCoefficient(MPI_Comm comm,
                            const std::string &fileName,
                            SliceOrientation orientation=NONE,
                            int slice = -1);

    virtual double Eval(ElementTransformation &trans,
                        const IntegrationPoint &ip) override;

    std::vector<double> &GetPermeability();

private:
    void ReadPermeabilityFile(const std::string &fileName);

    void ReadPermeabilityFile(MPI_Comm comm,
                              const std::string &fileName);

    double Permeability(const Vector &x);

    static constexpr int N_[3] = {60, 220, 85};
    static constexpr double h_[3] = {20.0, 10.0, 2.0};

    std::size_t initial_skip_;
    std::size_t line_skip_;
    std::size_t plane_skip_;

    int normal_axis_;
    int slice_;

    SliceOrientation orientation_;
    std::vector<double> permeability_;

    int r_[3][2];
};

} // namespace saamge

#endif // _SPE10_HPP
