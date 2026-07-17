#include <mfem.hpp>
#include <mpi.h>
#include "saamge.hpp"

using namespace mfem;
using namespace saamge;

double tau(2*M_PI);

double sol_func(Vector& x)
{
    const double xi(x(0));
    const double xj(x(1));
    return sin(tau*xi) * sin(tau*xj);
}

double rhs_func(Vector &x)
{
    SA_ASSERT(2 == x.Size());
    return 2*pow(tau, 2) * sol_func(x);
}

double bdr_cond(Vector &x)
{
    SA_ASSERT(2 == x.Size());
    return 0.0;
}

/**
   do the agglomerate partitioning for mltest mesh
   this partitions elements into agglomerates
   this only makes sense for the first coarsening
*/
agg_partitioning_relations_t *
fem_create_test_partitioning(HypreParMatrix &A, ParFiniteElementSpace &fes,
                             const agg_dof_status_t *bdr_dofs, int *nparts,
                             bool do_aggregates)
{
    Table *elem_to_dof, *elem_to_elem;
    Mesh *mesh = fes.GetMesh();

    //XXX: This will stay allocated in MESH till the end.
    elem_to_elem = mbox_copy_table(&(mesh->ElementToElementTable()));

    //XXX: This remains allocated in FES till the end.
    elem_to_dof = mbox_copy_table(&(fes.GetElementToDofTable()));

    int *partitioning = NULL;
    partitioning = new int[12];
    partitioning[0] = partitioning[1] = partitioning[4] = partitioning[5] = 0;
    partitioning[2] = partitioning[3] = 1;
    partitioning[6] = partitioning[7] = partitioning[11] = 2;
    partitioning[8] = partitioning[9] = partitioning[10] = 3;

    std::cout << "partitioning_array:";
    for (int i=0; i<12; ++i)
        std::cout << " " << partitioning[i];
    std::cout << std::endl;

    // in what follows, bdr_dofs is only used as info to copy onto coarser level, 
    // does not actually affect partitioning
    agg_partitioning_relations_t *agg_part_rels =
        agg_create_partitioning_fine(A, fes.GetNE(), elem_to_dof, elem_to_elem,
                                     partitioning, bdr_dofs, nparts, 
                                     fes.Dof_TrueDof_Matrix(), do_aggregates, true);

    SA_ASSERT(agg_part_rels);
    return agg_part_rels;
}

int *fem_partition_test_mesh(Mesh &mesh, int *nparts)
{
    SA_ASSERT(*nparts == 4);

    int *out = new int[mesh.GetNE()];
    out[0] = out[1] = out[4] = out[5] = 0;
    out[2] = out[3] = 1;
    out[6] = out[7] = out[11] = 2;
    out[8] = out[9] = out[10] = 3;
    return out;
}

int main(int argc, char *argv[])
{
    mfem::MPI_Session mpi(argc, argv);
    const int myid = mpi.WorldRank();
    const int num_procs = mpi.WorldSize();
    MPI_Comm active_comm = MPI_COMM_WORLD;
    proc_init(active_comm);

    agg_partitioning_relations_t *agg_part_rels;
    ml_data_t *ml_data;

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
    int generate_mesh = -1;
    args.AddOption(&generate_mesh, "--generate-mesh", "--generate-mesh",
                   "Generate 2D quad mesh with this number of elements per side.");
    bool minimal_coarse = false;
    bool linear_coarse = false;
    args.AddOption(&linear_coarse, "-lc", "--linear-coarse",
                   "-nlc", "--no-linear-coarse",
                   "Add linear functions to coarse basis (only for finest coarsening).");
    bool correct_nulspace = true;
    args.AddOption(&correct_nulspace, "-c", "--correct-nulspace",
                   "-nc", "--no-correct-nulspace",
                   "Use the corrected nulspace technique on coarsest level.");
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
        args.PrintOptions(cout);

    if (first_elems_per_agg < 0) first_elems_per_agg = elems_per_agg;
    if (first_theta < 0.0) first_theta = theta;
    if (first_nu_pro < 0) first_nu_pro = nu_pro;

    // Do not do both corrected nullspace technique and coarse space of just ones vector
    SA_ASSERT(!(correct_nulspace && minimal_coarse));

    MPI_Barrier(active_comm); // try to make MFEM's debug element orientation prints
                            // not mess up the parameters above
    bool mltest = false;
    Mesh *mesh;
    if (generate_mesh > 0)
    {
        mesh = new Mesh(generate_mesh, generate_mesh, Element::QUADRILATERAL, 1);
    }
    else
    {
        mesh = fem_read_mesh(mesh_file);
        if (mesh->GetNV() == 20 && mesh->GetNE() == 12 && 
            times_refine == 0 && serial_times_refine == 0) // not very general...
            mltest = true;
        if (0 == myid)
            std::cout << "<<<< bool mltest = " << mltest << std::endl;
    }
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
    SA_RPRINTF(0,"NV: %d, NE: %d\n", mesh->GetNV(), mesh->GetNE());

    int *proc_partitioning = fem_partition_mesh(*mesh, &nprocs);
    if (0 == myid && visualize)
        fem_serial_visualize_partitioning(*mesh, proc_partitioning);
    ParMesh pmesh(active_comm, *mesh, proc_partitioning);
    delete mesh;
    delete []proc_partitioning;

    for (int i=0; i<times_refine; ++i)
    {
        pmesh.UniformRefinement();
    }

    H1_FECollection fec(order, dim);
    ParFiniteElementSpace fes(&pmesh, &fec);

    const int pNV = pmesh.GetNV();
    const int pNE = pmesh.GetNE();
    const int pND = fes.GetNDofs();
    const int ND = fes.GlobalTrueVSize();
    SA_RPRINTF(0,"pNV: %d, pNE: %d, pND: %d, ND: %d\n", pNV, pNE, pND, ND);

    ofstream mesh_ofs("mltest.mesh");
    mesh_ofs.precision(8);
    pmesh.Print(mesh_ofs);

    FunctionCoefficient sol(sol_func);
    FunctionCoefficient rhs(rhs_func);
    ConstantCoefficient conduct_func(1.0);

    ParGridFunction x(&fes);
    FunctionCoefficient bdr_coeff(bdr_cond);
    x.ProjectCoefficient(bdr_coeff);

    ParLinearForm b(&fes);
    b.AddDomainIntegrator(new DomainLFIntegrator(rhs));
    b.Assemble();

    ParBilinearForm a(&fes);
    a.AddDomainIntegrator(new mfem::DiffusionIntegrator(conduct_func));
    a.Assemble();

    Array<int> ess_bdr(pmesh.bdr_attributes.Max());
    ess_bdr = 1;
    Array<int> ess_dofs_list;
    fes.GetEssentialVDofs(ess_bdr, ess_dofs_list);
    const bool keep_diag = true;
    a.EliminateEssentialBCFromDofs(ess_dofs_list, x, b, keep_diag);
    a.Finalize(0);

    SparseMatrix &Al = a.SpMat();
    {
        std::stringstream filename;
        filename << "global_stiffness" << myid;
        std::ofstream out(filename.str().c_str());
        Al.Print(out);
    }
    HypreParMatrix Ag = *a.ParallelAssemble();

    HypreBoomerAMG hbamg(Ag);
    hbamg.SetPrintLevel(0);

    CGSolver pcg(active_comm);
    pcg.SetPreconditioner(hbamg);
    pcg.SetOperator(Ag);
    pcg.SetRelTol(1e-6); // for some reason MFEM squares this...
    pcg.SetMaxIter(1000);
    pcg.SetPrintLevel(2);

    SA_RPRINTF(0, "x.Norml2() = %f\n", x.Norml2());
    pcg.Mult(b, x);
    std::cout << "<<<< |u_h - u|_2 = " << x.ComputeL2Error(sol) << std::endl;
    ofstream solh_ofs("solh.gf");
    solh_ofs.precision(8);
    x.Save(solh_ofs);

    return 0;
}
