#include "perm_func.hpp"
#include <fstream>

namespace saamge
{

PermeabilityCoefficient::PermeabilityCoefficient(
    MPI_Comm comm, const std::string &file_name,
    const mfem::Array<int> &N, const mfem::Array<int> &max_N,
    const mfem::Vector &h, SliceOrientation orientation, int slice)
    :
    mfem::VectorCoefficient(orientation == NONE ? 3 : 2), h_(h),
    slice_(slice), orientation_(orientation), N_slice_(N[0] * N[1]),
    N_all_(N_slice_ * N[2]), permeability_(3 * N_all_)
{
    MFEM_ASSERT(N.Size() == max_N.Size(), "");
    for (int i = 0; i < N.Size(); ++i)
        MFEM_ASSERT(N[i] <= max_N[i], "");
    N.Copy(N_);

    if (file_name == "")
        BlankPermeability();
    else
        ReadPermeabilityFile(comm, file_name, max_N);
}

PermeabilityCoefficient::PermeabilityCoefficient(
    MPI_Comm comm, const mfem::Array<int> &N, const mfem::Vector &h,
    double horizontal_perm, double vertical_perm)
    :
    mfem::VectorCoefficient(3), h_(h), slice_(-1), orientation_(NONE),
    N_slice_(N[0] * N[1]), N_all_(N_slice_ * N[2]), permeability_(3 * N_all_)
{
    N.Copy(N_);
    double *ip = permeability_.data();
    for (int l = 0; l < 3; l++)
    {
        const double perm_val = (l == 2) ? vertical_perm : horizontal_perm;
        for (int k = 0; k < N_[2]; ++k)
            for (int j = 0; j < N_[1]; ++j)
                for (int i = 0; i < N_[0]; ++i, ++ip)
                    *ip = perm_val;
    }
}

void PermeabilityCoefficient::ReadPermeabilityFile(const std::string &fileName,
                                                   const mfem::Array<int> &max_N)
{
    std::ifstream perm_file(fileName, std::ios::binary | std::ios::ate);

    if (!perm_file.is_open())
    {
        std::cerr << "Error in opening file " << fileName << std::endl;
        mfem::mfem_error("File does not exist");
    }

    const std::streamsize size = perm_file.tellg();
    perm_file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size+1);
    if (!perm_file.read(buffer.data(), size))
        mfem::mfem_error("Failed to read permeability file into memory");
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
            std::strtod(ptr, &endptr);
            ptr = endptr;
        }
    };

    const int i_skip = max_N[0] - N_[0];
    const int j_skip = (max_N[1] - N_[1]) * max_N[0];
    const int k_skip = (max_N[2] - N_[2]) * max_N[1] * max_N[0];

    double *ip = permeability_.data();
    for (int l = 0; l < 3; l++)
    {
        for (int k = 0; k < N_[2]; ++k)
        {
            for (int j = 0; j < N_[1]; ++j)
            {
                for (int i = 0; i < N_[0]; ++i, ++ip)
                    *ip = parse_next(); 
                if (i_skip) skip_values(i_skip);
            }
            if (j_skip) skip_values(j_skip);
        }
        if (l < 2 && k_skip) skip_values(k_skip);
    }
}

void PermeabilityCoefficient::ReadPermeabilityFile(
    MPI_Comm comm, const std::string &fileName, const mfem::Array<int> &max_N)
{
    int myid;
    MPI_Comm_rank(comm, &myid);

    if (myid == 0)
        ReadPermeabilityFile(fileName, max_N);

    MPI_Bcast(permeability_.data(), 3 * N_all_, MPI_DOUBLE, 0, comm);
}

void PermeabilityCoefficient::BlankPermeability()
{
    std::fill(permeability_.begin(), permeability_.end(), 1.0);
}

void PermeabilityCoefficient::Permeability(const mfem::Vector &x,
                                           mfem::Vector &val)
{
    val.SetSize(x.Size());

    int index[3] = {0, 0, 0};
    int axis_map[3] = {0, 1, 2};

    switch (orientation_)
    {
        case NONE:
            break;

        case XY:
            index[2] = slice_;
            break;

        case XZ:
            axis_map[1] = 2; axis_map[2] = 1; 
            index[1] = slice_;
            break;

        case YZ:
            axis_map[0] = 1; axis_map[1] = 2; axis_map[2] = 0;
            index[0] = slice_;
            break;

        default:
            mfem::mfem_error("PermeabilityCoefficient::Permeability - invalid orientation");
    }

    for (int d = 0; d < vdim; ++d)
    {
        const int axis = axis_map[d];
        index[axis] = std::max(0, std::min(static_cast<int>(std::floor(x[d]/h_[axis])), N_[axis]-1));
    }

    const int offset = index[0] + N_[0] * index[1] * N_slice_ * index[2];

    for (int d = 0; d < vdim; ++d)
        val[d] = permeability_[offset + axis_map[d] * N_all_];
}

double PermeabilityCoefficient::FroNorm(const mfem::Vector &x)
{
    mfem::Vector val(3);
    Permeability(x, val);
    return val.Norml2();
}

} // namespace saamge
