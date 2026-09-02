#include <algorithm> // std::minmax_element
#include <cmath>     // std::floor, std::pow
#include <memory>    // std::unique_ptr
#include <mfem.hpp>
#include <mpi.h>
#include <numeric>   // std::accumulate
#include <random>    // std::mt19937

#include "saamge.hpp"

using namespace mfem;
using namespace saamge;

using std::endl;
double tau(2*M_PI);

double sol_func(Vector &x)
{
    // const double xi(x(0));
    // const double xj(x(1));
    // return sin(tau*xi) * sin(tau*xj);
    return 0.0;
}

double rhs_func(Vector &x)
{
    SA_ASSERT(2 == x.Size());
    // return 2*pow(tau, 2) * sol_func(x);
    return 0.0;
}

double bdr_cond(Vector &x)
{
    SA_ASSERT(2 == x.Size());
    return 0.0;
}

void tensor_func(const Vector &x, DenseMatrix &K)
{
    const int dim = x.Size();
    SA_ASSERT(2 == dim || 3 == dim);

    K.SetSize(dim);

    double b[3];
    const double theta = 2.0 * M_PI * (x(0) + x(1));
    b[0] = std::cos(theta);
    b[1] = std::sin(theta);
    if (3 == dim)
    {
        const double phi = M_PI * x(2);
        b[0] *= std::sin(phi);
        b[1] *= std::sin(phi);
        b[2] = std::cos(phi);
    }
    // norm squared gives magnitude for anisotropy ratio
    const double norm = 1e2;
    for (int i = 0; i < dim; ++i)
        b[i] *= norm;

    for (int i = 0; i < dim; ++i)
    {
        for (int j = 0; j < dim; ++j)
            K(i, j) = b[i] * b[j];
        K(i, i) += 1.0;
    }
}

double trace_func(const Vector &x)
{
    DenseMatrix K;
    tensor_func(x, K);
    return K.Trace();
}

double rand_contrast_func(Vector &x)
{
    const int dim = x.Size();
    SA_ASSERT(2 == dim || 3 == dim);

    const int num_blocks = 32; // must divide nxy
    const double eps = 1e-12;

    int ix = static_cast<int>(std::floor((x(0) + eps) * num_blocks));
    int iy = static_cast<int>(std::floor((x(1) + eps) * num_blocks));

    ix = std::max(0, std::min(ix, num_blocks - 1));
    iy = std::max(0, std::min(iy, num_blocks - 1));

    const unsigned int u_ix = static_cast<unsigned int>(ix);
    const unsigned int u_iy = static_cast<unsigned int>(iy);

    unsigned int seed = (u_ix * 73856093u) ^ (u_iy * 19349663u);

    if (3 == dim)
    {
        int iz = static_cast<int>(std::floor((x(2) + eps) * num_blocks));
        iz = std::max(0, std::min(iz, num_blocks - 1));
        const unsigned int u_iz = static_cast<unsigned int>(iz);
        seed ^= (u_iz * 83492791u);
    }

    std::mt19937 gen(seed);
    // std::uniform_int_distribution<int> dist(0, 6);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // const int exponent = dist(gen);
    const double val = dist(gen);
    const int exponent = (val < 0.15) ? 6 : 0;

    return std::pow(10.0, static_cast<double>(exponent));
}

double checkboard_func(Vector &x)
{
    const int dim = x.Size();
    SA_ASSERT(2 == dim || 3 == dim);

    const int num_blocks = 8; // must divide nxy
    const double eps = 1e-12;

    int ix = static_cast<int>(std::floor((x(0) + eps) * num_blocks));
    int iy = static_cast<int>(std::floor((x(1) + eps) * num_blocks));

    ix = std::max(0, std::min(ix, num_blocks - 1));
    iy = std::max(0, std::min(iy, num_blocks - 1));

    int parity = ix + iy;

    if (3 == dim)
    {
        int iz = static_cast<int>(std::floor((x(2) + eps) * num_blocks));
        iz = std::max(0, std::min(iz, num_blocks - 1));
        parity += iz;
    }

    return (0 == parity % 2) ? 1e6 : 1e0;
}

void print_mises_info(const agg_partitioning_relations_t &agg_part_rels);

/**
   do the agglomerate partitioning for mltest mesh
   this partitions elements into four agglomerates
   this only makes sense for the first coarsening
*/
agg_partitioning_relations_t *
fem_create_test_partitioning(HypreParMatrix &A, ParFiniteElementSpace &fes,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             bool do_aggregates)
{
    Mesh *mesh = fes.GetMesh();

    // the following two tables stay allocated until the end of main
    Table *elem_to_elem = mbox_copy_table(&(mesh->ElementToElementTable()));
    Table *elem_to_dof = mbox_copy_table(&(fes.GetElementToDofTable()));

    int *element_agglomerate = nullptr;
    // indices below are all local and not global
    // 12 == mesh->GetNE();
    switch (PROC_NUM)
    {
        case 1:
            element_agglomerate = new int[12]{0, 0, 1, 1, 0, 0, 2, 2, 3, 3, 3, 2};
            break;
        case 2:
            if (0 == PROC_RANK)
                element_agglomerate = new int[6]{0, 0, 1, 1, 0, 0};
            else if (1 == PROC_RANK)
                element_agglomerate = new int[6]{0, 0, 1, 1, 1, 0};
            break;
        case 4:
        {
            // the nested ternary operators here suck,
            // but they reduce eyes glazing over
            int num_elem = (0 == PROC_RANK) ? 4 : (1 == PROC_RANK) ? 2 : 3;
            element_agglomerate = new int[num_elem]();
            break;
        }
        default:
            SA_ASSERT(false);
            break;
    }

    // bdr_dofs is only used as info to copy onto coarser level
    // it does not actually affect partitioning
    agg_partitioning_relations_t *agg_part_rels =
        agg_create_partitioning_fine(A, fes.GetNE(), elem_to_dof, elem_to_elem,
                                     element_agglomerate, bdr_dofs, nparts,
                                     fes.Dof_TrueDof_Matrix(), do_aggregates, true);
    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

// Create the element to process index array for the serial mesh
int *fem_partition_test_mesh(Mesh &mesh)
{
    SA_RPRINTF_NOTS(0, "Using the test mesh with %d processes\n", PROC_NUM);
    const int num_elements = mesh.GetNE();

    switch (PROC_NUM)
    {
        case 1:
            return new int[num_elements]();
        case 2:
            return new int[num_elements]{0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1};
        case 4:
            return new int[num_elements]{0, 0, 1, 1, 0, 0, 2, 2, 3, 3, 3, 2};
        default:
            SA_ASSERT(false);
            return nullptr;
    }
}

int main(int argc, char *argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int myid = mpi.WorldRank();
    const int num_procs = mpi.WorldSize();
    MPI_Comm active_comm = MPI_COMM_WORLD;
    proc_init(active_comm);

    OptionsParser args(argc, argv);
    const char *mesh_file = "/Users/bfadness/src/forks/saamge/amg/test/mltest.mesh";
    bool visualize = true;
    args.AddOption(&visualize, "-vis", "--visualization", "-no-vis",
                   "--no-visualization",
                   "Enable or disable GLVis visualization.");
    int serial_times_refine = 0;
    int times_refine = 0;
    args.AddOption(&times_refine, "-r", "--refine", 
                   "How many times to refine the mesh (in parallel).");
    int nu_pro = 0;
    args.AddOption(&nu_pro, "-p", "--nu-pro",
                   "Degree of the smoother for the smoothed aggregation for first coarsening.");
    int first_nu_pro = -1;
    args.AddOption(&first_nu_pro, "-fp", "--first-nu-pro",
                   "Degree of smoother for smoothed aggregation on later coarsenings.");
    int nu_relax = 3;
    args.AddOption(&nu_relax, "-n", "--nu-relax",
                   "Degree for smoother in the relaxation.");
    int order = 1;
    args.AddOption(&order, "-o", "--order",
                   "Polynomial order of finite element space.");
    double theta = 0.003;
    args.AddOption(&theta, "-t", "--theta",
                   "Tolerance for eigenvalue problems.");
    double first_theta = -1.0;
    args.AddOption(&first_theta, "-ft", "--first-theta",
                   "Tolerance for eigenvalue problems for first (finest) coarsening.");
    int num_levels = 2;
    args.AddOption(&num_levels, "-l", "--num-levels",
                   "Number of levels in multilevel algorithm.");
    int elems_per_agg = 256;
    args.AddOption(&elems_per_agg, "-e", "--elems-per-agg",
                   "Number of elements per agglomerated element.");
    int first_elems_per_agg = -1;
    args.AddOption(&first_elems_per_agg, "-fe", "--first-elems-per-agg",
                   "Number of elements per AE for first (finest) coarsening.");
    int nxy = 4;
    args.AddOption(&nxy, "-nxy", "--num_elem_per_dim",
                   "Generate 2D triangular mesh with this number of elements per side.");
    bool minimal_coarse = false;
    bool linear_coarse = false;
    args.AddOption(&linear_coarse, "-lc", "--linear-coarse",
                   "-nlc", "--no-linear-coarse",
                   "Add linear functions to coarse basis (only for finest coarsening).");
    bool correct_nullspace = false;
    args.AddOption(&correct_nullspace, "-c", "--correct-nullspace",
                   "-nc", "--no-correct-nullspace",
                   "Use the corrected nullspace technique on coarsest level.");
    bool double_cycle = false;
    args.AddOption(&double_cycle, "-d", "--double-cycle",
                   "-nd", "--no-double-cycle",
                   "Use the double cycle combined preconditioner.");
    bool coarse_direct = true;
    args.AddOption(&coarse_direct, "--coarse-direct", "--coarse-direct",
                   "--coarse-amg", "--coarse-amg",
                   "Use a direct solver on coarsest level, rather than default AMG V-cycle.");
    bool direct_eigensolver = true;
    args.AddOption(&direct_eigensolver, "-q", "--direct-eigensolver",
                   "-nq", "--no-direct-eigensolver",
                   "Use direct eigensolver from LAPACK instead of default ARPACK.");
    bool do_aggregates = false;
    args.AddOption(&do_aggregates, "-agg", "--do-aggregates",
                   "-nagg", "--no-do-aggregates",
                   "On coarsest level, use aggregates instead of MISes for lower complexity.");
    bool elasticity = false;
    bool identity_partition = false;
    bool adapt = false;
    args.AddOption(&adapt, "-ad", "--adapt",
                   "-nad", "--no-adapt",
                   "Perturbs the matrix and reuses the spaces.");

    args.Parse();
    if (!args.Good())
    {
        if (0 == myid)
            args.PrintUsage(cout);
        MPI_Finalize();
        return 1;
    }
    if (0 == myid)
    {
        args.PrintOptions(cout);
        cout << endl;
    }

    if (first_elems_per_agg < 0) first_elems_per_agg = elems_per_agg;
    if (first_theta < 0.0) first_theta = theta;
    if (first_nu_pro < 0) first_nu_pro = nu_pro;

    // Do not do both corrected nullspace technique and coarse space of just ones vector
    SA_ASSERT(!(correct_nullspace && minimal_coarse));

    MPI_Barrier(active_comm); // try to make MFEM's debug element orientation prints
                              // not mess up the parameters above
    bool mltest = false;
    Mesh *mesh;
    if (nxy > 0)
    {
        mesh = new Mesh(nxy, nxy, Element::TRIANGLE, 1);
    }
    else
    {
        mesh = fem_read_mesh(mesh_file);
        if (20 == mesh->GetNV() && 12 == mesh->GetNE() &&
            0 == times_refine && 2 == num_levels) // not very general...
            mltest = true;
    }
    SA_RPRINTF_NOTS(0, "<<<< bool mltest = %d\n", mltest);
    const int dim = mesh->Dimension();
    int nprocs = mpi.WorldSize();
    float ratio = (float)nprocs/mesh->GetNE();
    if (ratio > 1)
    {
        serial_times_refine = std::ceil(std::log2(ratio)/dim);
        for (int i=0; i<serial_times_refine; ++i)
        {
            mesh->UniformRefinement();
        }
    }

    int *proc_partitioning = nullptr;
    if (mltest)
        proc_partitioning = fem_partition_test_mesh(*mesh);
    else
        proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    ParMesh pmesh(active_comm, *mesh, proc_partitioning);
    delete mesh;
    delete []proc_partitioning;

    for (int i=0; i<times_refine; ++i)
        pmesh.UniformRefinement();

    H1_FECollection fec(order, dim);
    ParFiniteElementSpace fes(&pmesh, &fec);

    const int pNV = pmesh.GetNV();
    const int pNE = pmesh.GetNE();
    const int pND = fes.GetNDofs();
    const int gNE = pmesh.GetGlobalNE();
    const int gND = fes.GlobalTrueVSize();

    SA_RPRINTF_NOTS(0, "<<<< %d agglomerates\n", pNE / first_elems_per_agg);
    SA_PRINTF("gNE: %d, gND: %d\n", gNE, gND);
    SA_PRINTF("pNE: %d, pND: %d\n", pNE, pND);

    std::ostringstream mesh_name;
    mesh_name << "mesh." << std::setfill('0') << std::setw(6) << myid;
    std::ofstream mesh_ofs(mesh_name.str().c_str());
    mesh_ofs.precision(8);
    pmesh.Print(mesh_ofs);

    FunctionCoefficient sol(sol_func);
    FunctionCoefficient rhs(rhs_func);
    ConstantCoefficient conduct(1.0);
    // FunctionCoefficient conduct(checkboard_func);
    // FunctionCoefficient conduct(rotated_channel_func);
    // MatrixFunctionCoefficient conduct(dim, tensor_func);
    // FunctionCoefficient trace(trace_func);

    L2_FECollection cfec(0, dim);
    ParFiniteElementSpace cfes(&pmesh, &cfec);
    ParGridFunction conduct_gf(&cfes);
    conduct_gf.ProjectCoefficient(conduct);
    // conduct_gf.ProjectCoefficient(trace);
    if (visualize)
        fem_parallel_visualize_gf(pmesh, conduct_gf);

    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 1;

    ParGridFunction x(&fes);
    FunctionCoefficient bdr_coeff(bdr_cond);
    const int seed = 0;
    x.Randomize(seed);
    x.ProjectBdrCoefficient(bdr_coeff, ess_bdr);

    ParLinearForm b(&fes);
    b.AddDomainIntegrator(new DomainLFIntegrator(rhs));
    b.Assemble();

    ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new DiffusionIntegrator(conduct));
    a.Assemble();

    const bool keep_diag = true;
    a.EliminateEssentialBC(ess_bdr, x, b, keep_diag);
    a.Finalize();

/*
    SparseMatrix &Al = a.SpMat();
    {
        std::stringstream filename;
        filename << "global_stiffness" << myid;
        std::ofstream out(filename.str().c_str());
        Al.Print(out);
    }
*/

    std::unique_ptr<HypreParVector> X(x.GetTrueDofs());
    std::unique_ptr<HypreParVector> B(b.ParallelAssemble());
    std::unique_ptr<HypreParMatrix> A(a.ParallelAssemble());

    HypreBoomerAMG amg(*A);
    amg.SetPrintLevel(0);

    // use the same tolerance for all conjugate gradient runs
    const double rel_tol = 1e-6;
    const double max_iter = 1000;

    CGSolver pcg(active_comm);
    pcg.SetPreconditioner(amg);
    pcg.SetOperator(*A);
    pcg.SetRelTol(rel_tol); // for some reason MFEM squares this...
    pcg.SetMaxIter(max_iter);
    pcg.SetPrintLevel(1);
    pcg.Mult(*B, *X);
    x.Distribute(*X);

    double error = x.ComputeL2Error(sol);
    SA_RPRINTF_NOTS(0, "<<<< |u_h - u|_2 = %12.5e\n\n", error);

    std::ostringstream sol_name;
    sol_name << "sol1." << std::setfill('0') << std::setw(6) << myid;
    std::ofstream sol_ofs(sol_name.str().c_str());
    sol_ofs.precision(8);
    x.Save(sol_ofs);
    sol_ofs.close();

//////////////////////////////////////////////////////////////////////////////////////////////

    // AMGe code begins here
    int *nparts_arr = new int[num_levels-1];
    const bool do_aggregates_here = do_aggregates && (num_levels == 2);
    agg_dof_status_t *bdr_dofs = fem_find_bdr_dofs(fes, &ess_bdr);
    agg_partitioning_relations_t *agg_part_rels;
    if (mltest)
    {
        nparts_arr[0] = 4/PROC_NUM;
        SA_RPRINTF_NOTS(0, "%s", "Get the agglomerate test mesh partition relations\n");
        agg_part_rels = fem_create_test_partitioning(
            *A, fes, bdr_dofs, nparts_arr, do_aggregates_here);
    }
    else
    {
        nparts_arr[0] = std::max(1, pNE / first_elems_per_agg);
        for (int i=1; i<num_levels-1; ++i)
            nparts_arr[i] = std::lround(
                static_cast<double>(nparts_arr[i-1]) / elems_per_agg);
        agg_part_rels = fem_create_partitioning(
            *A, fes, bdr_dofs, nparts_arr, do_aggregates_here);
    }
    delete []bdr_dofs;

    if (visualize)
    {
        fem_parallel_visualize_partitioning(
            pmesh, agg_part_rels->partitioning, nparts_arr[0]);
    }
    ElementMatrixProvider *emp(
        new ElementMatrixStandardGeometric(*agg_part_rels, a.SpMat(), &a));
    int polynomial_coarse(minimal_coarse ? 0 : -1);
    MultilevelParameters mlp(
        num_levels-1, nparts_arr, first_nu_pro, nu_pro, nu_relax,
        first_theta, theta, polynomial_coarse, correct_nullspace,
        !direct_eigensolver, do_aggregates);
    mlp.set_use_double_cycle(double_cycle);
    mlp.set_coarse_direct(coarse_direct);
    if (linear_coarse)
        mlp.set_polynomial_coarse_space(0, 1);

    ml_data_t *ml_data(ml_produce_data(*A, agg_part_rels, emp, mlp));
    print_mises_info(*agg_part_rels);

    // argument 0 means fine level
    levels_level_t *level = levels_list_get_level(ml_data->levels_list, 0);

    // reset the values in X
    x.Randomize(seed);
    x.ProjectBdrCoefficient(bdr_coeff, ess_bdr);
    x.GetTrueDofs(*X);

//////////////////////////////////////////////////////////////////////////////////////////////

    const int iter = tg_run(*A, agg_part_rels, *X, *B, max_iter,
        rel_tol, 0.0, 1.0, level->tg_data, false);
    x.Distribute(*X);

    error = x.ComputeL2Error(sol);
    SA_RPRINTF_NOTS(0, "<<<< |u_h - u|_2 = %12.5e\n\n", error);

    sol_name.str("");
    sol_name.clear();
    sol_name << "sol2." << std::setfill('0') << std::setw(6) << myid;
    sol_ofs.open(sol_name.str().c_str());
    sol_ofs.precision(8);
    x.Save(sol_ofs);
    sol_ofs.close();

//////////////////////////////////////////////////////////////////////////////////////////////

    x.Randomize(seed);
    x.ProjectBdrCoefficient(bdr_coeff, ess_bdr);
    x.GetTrueDofs(*X);

    Solver *amge = new VCycleSolver(level->tg_data, false); // interactive_mode
    amge->SetOperator(*A);
    pcg.SetPreconditioner(*amge);
    pcg.SetOperator(*A);
    pcg.Mult(*B, *X);
    x.Distribute(*X);

    error = x.ComputeL2Error(sol);
    SA_RPRINTF_NOTS(0, "<<<< |u_h - u|_2 = %12.5e\n\n", error);

    sol_name.str("");
    sol_name.clear();
    sol_name << "sol3." << std::setfill('0') << std::setw(6) << myid;
    sol_ofs.open(sol_name.str().c_str());
    sol_ofs.precision(8);
    x.Save(sol_ofs);
    sol_ofs.close();

    ml_free_data(ml_data);
    agg_free_partitioning(agg_part_rels);
    delete []nparts_arr;

    return 0;
}

void print_mises_info(const agg_partitioning_relations_t &agg_part_rels)
{
    const int *arr = agg_part_rels.mises_size;
    const int n = agg_part_rels.num_owned_mises;

    auto result = std::minmax_element(arr, arr + n);
    double sum = std::accumulate(arr, arr + n, 0.0);

    SA_RPRINTF(0, "Size of MISes (min | avg | max): (%d | %g | %d)\n",
        *result.first, sum / static_cast<double>(n), *result.second);
}
