#include "spe10.hpp"
#include "process.hpp" // PROC_RANK

#include <algorithm>   // std::minmax
#include <cmath>       // std::floor
#include <cstdlib>     // std::strtod
#include <fstream>     // std::ifstream
#include <iostream>    // std::cerr

using namespace mfem;

namespace saamge
{

constexpr int PermeabilityCoefficient::N_[3];
constexpr double PermeabilityCoefficient::h_[3];

PermeabilityCoefficient::PermeabilityCoefficient(
    MPI_Comm comm,
    const std::string &file_name,
    SliceOrientation orientation,
    int slice)
    :
    normal_axis_(-1),
    slice_(slice),
    orientation_(orientation)
{
    for (int d = 0; d < 3; ++d)
    {
        r_[d][0] = 0;
        r_[d][1] = N_[d];
    }

    if (NONE != orientation_)
    {
        normal_axis_ = (orientation_ == XY) ? 2 : (orientation_ == XZ) ? 1 : 0;
        r_[normal_axis_][0] = slice_;
        r_[normal_axis_][1] = slice_ + 1;
    }

    initial_skip_ = r_[0][0] + r_[1][0] * N_[0] + r_[2][0] * N_[0] * N_[1];
    const int dy = r_[1][1] - r_[1][0];
    plane_skip_ =  (N_[1] - dy) * N_[0];
    const int dx = r_[0][1] - r_[0][0];
    line_skip_ = N_[0] - dx;

    const int perm_size = (r_[2][1] - r_[2][0]) * dx * dy;
    permeability_.resize(perm_size);
    ReadPermeabilityFile(comm, file_name);
}

void PermeabilityCoefficient::ReadPermeabilityFile(const std::string &fileName)
{
    std::ifstream perm_file(fileName, std::ios::binary | std::ios::ate);

    if (!perm_file.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem_error("File does not exist");
    }

    const std::streamsize size = perm_file.tellg();
    perm_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size+1);
    if (!perm_file.read(buffer.data(), size))
        mfem_error("Failed to read permeability file into memory");
    buffer[size] = '\0';

    const char *ptr = buffer.data();
    char *endptr = nullptr;

    auto parse_next = [&ptr, &endptr]() -> double
    {
        double val = std::strtod(ptr, &endptr);
        ptr = endptr;
        return val;
    };

    auto skip_values = [&ptr, &endptr](std::size_t count) -> void
    {
        for (std::size_t n = 0; n < count; ++n)
        {
            std::strtof(ptr, &endptr);
            ptr = endptr;
        }
    };

    double *ip = permeability_.data();
    if (initial_skip_) skip_values(initial_skip_);

    // only read the first component for now
    for (int k = r_[2][0]; k < r_[2][1]; ++k)
    {
        for (int j = r_[1][0]; j < r_[1][1]; ++j)
        {
            for (int i = r_[0][0]; i < r_[0][1]; ++i, ++ip)
                *ip = parse_next();
            if (line_skip_) skip_values(line_skip_);
        }
        if (plane_skip_) skip_values(plane_skip_);
    }
}

void PermeabilityCoefficient::ReadPermeabilityFile(
    MPI_Comm comm, const std::string &fileName)
{
    if (0 == PROC_RANK)
        ReadPermeabilityFile(fileName);

    MPI_Bcast(permeability_.data(), static_cast<int>(permeability_.size()),
        MPI_DOUBLE, 0, comm);
}

double PermeabilityCoefficient::Permeability(const Vector &x)
{
    int index[3] = {r_[0][0], r_[1][0], r_[2][0]};
    std::array<int, 3> axis_map = {0, 1, 2};
    switch (orientation_)
    {
        case NONE:
            break;

        case XY:
            break;

        case XZ:
            axis_map = {0, 2, 1};
            break;

        case YZ:
            axis_map = {1, 2, 0};
            break;

        default:
            mfem_error("PermeabilityCoefficient::Permeability - invalid orientation");
    }

    for (int d = 0; d < x.Size(); ++d)
    {
        const int axis = axis_map[d];
        index[axis] = std::max(0,
            std::min(static_cast<int>(std::floor(x[d]/h_[axis])), r_[axis][1]-1));
    }

    for (int d = 0; d < 3; ++d)
        index[d] -= r_[d][0];

    const int dx = r_[0][1] - r_[0][0];
    const int dy = r_[1][1] - r_[1][0];

    const int offset = index[0] + dx * index[1] + (dx * dy) * index[2];
    return permeability_[offset];
}

std::vector<double> &PermeabilityCoefficient::GetPermeability()
{
    return permeability_;
}

double PermeabilityCoefficient::Eval(ElementTransformation &trans,
                                     const IntegrationPoint &ip)
{
    Vector phys_ip(trans.GetSpaceDim());
    trans.Transform(ip, phys_ip);
    return Permeability(phys_ip);
}

} // namespace saamge
