/*! \file

    SAAMGE: smoothed aggregation element based algebraic multigrid hierarchies
            and solvers.

    Copyright (c) 2018, Lawrence Livermore National Security,
    LLC. Developed under the auspices of the U.S. Department of Energy by
    Lawrence Livermore National Laboratory under Contract
    No. DE-AC52-07NA27344. Written by Delyan Kalchev, Andrew T. Barker,
    and Panayot S. Vassilevski. Released under LLNL-CODE-667453.

    This file is part of SAAMGE. 

    Please also read the full notice of copyright and license in the file
    LICENSE.

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License (as
    published by the Free Software Foundation) version 2.1 dated February
    1999.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
    conditions of the GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this program; if not, see
    <http://www.gnu.org/licenses/>.
*/

#include "smpr.hpp"
#define _USE_MATH_DEFINES

#include <algorithm>        // std::sort, std::unique
#include <cmath>
#include <mfem.hpp>
#include <utility>          // std::pair
#include <vector>

#include "common.hpp"
#include "helpers.hpp"
#include "mbox.hpp"
#include "tg.hpp"

using std::sqrt;
using std::sin;
using std::cos;
using std::pow;
using std::log;

namespace saamge
{
using namespace mfem;

/* Macros and Static Functions */

static inline
double smpr_compute_theta0(int nu, double a)
{
    double tmp, theta0;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);

    tmp = sqrt(a);
    tmp = (1. - tmp) / (1. + tmp);
    tmp = 1. + pow(tmp, nu<<1);

    theta0 = - ((pow(1. - a, 2) * tmp) / (8. * a));

    return theta0;
}

static inline
double smpr_compute_theta1(int nu, double a)
{
    double tmp, theta1;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);

    tmp = sqrt(a);
    tmp = (1. - tmp) / (1. + tmp);
    tmp = pow(1. / tmp, 2) + pow(tmp, nu<<1);

    theta1 = (pow(1. - a, 2)  * tmp) / (16. * a);

    return theta1;
}

static inline
double smpr_compute_theta_1(int nu, double a)
{
    double tmp, theta_1;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);

    tmp = sqrt(a);
    tmp = (1. - tmp) / (1. + tmp);
    tmp = pow(tmp, 2) + pow(tmp, nu<<1);

    theta_1 = (pow(1. - a, 2)  * tmp) / (16. * a);

    return theta_1;
}

static inline
double smpr_compute_theta(int nu, double a, double theta0, double theta1)
{
    double tmp, theta;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);

    tmp = - ((1. + a) / (1. - a));
    tmp = (smpr_cheb_firstkind(nu, tmp) * (1. + a)) /
          (smpr_cheb_firstkind(nu + 1, tmp) * (1. - a));

    theta = theta0 - 2. * theta1 * tmp;

    return theta;
}

static inline
double smpr_compute_tau0(int nu, double a, double theta0, double theta1)
{
    double tmp, tau0;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);

    tmp = - ((1. + a) / (1. - a));
    tmp = (smpr_cheb_firstkind(nu + 1, tmp) * (1. - a) * theta0) /
          (smpr_cheb_firstkind(nu, tmp) * 4. * theta1);

    tau0 = (1. + a) * 0.5 - tmp;
    SA_ASSERT(0. != tau0);

    return tau0;
}

static inline
double smpr_compute_tauk(int nu, double a, int k)
{
    double tmp, tauk, sin_val, cos_val;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 0);
    SA_ASSERT(1 <= k && k <= nu);

    tmp = ((2. * (double)k - 1.) * M_PI_4) / (double)nu;
    cos_val = cos(tmp);
    sin_val = sin(tmp);

    tauk = a * cos_val * cos_val + sin_val * sin_val;
    SA_ASSERT(0. != tauk);

    return tauk;
}

/* Functions */

void SmootherSolver::SetOperator(const Operator &op)
{
    A = const_cast<HypreParMatrix *>(dynamic_cast<const HypreParMatrix *>(&op));
    if (A == NULL)
        mfem_error("SmootherSolver::SetOperator : not HypreParMatrix!");
}

void SmootherSolver::Mult(const Vector &b, Vector &x) const
{
    SA_ASSERT(smoother);
    SA_ASSERT(data);
    SA_ASSERT(A);
    SA_ASSERT(A->Width() == b.Size());
    SA_ASSERT(b.Size() == x.Size());

    if (!iterative_mode)
        x = 0.0;

    for (int i=0; i < iters; ++i)
        smoother(*A, b, x, data);
}

int smpr_nu_from_a(double a)
{
    double tmp;

    SA_ASSERT(0. < a && a < 1.);

    tmp = sqrt(a);
    tmp = (1. - tmp) / (1. + tmp);

    return (int)(log(a / (1. - a)) / log(tmp) + 1.);
}

double smpr_cheb_firstkind(int n, double x)
{
    double prevprev, prev, value = 0.;

    SA_ASSERT(n >= 0);

    if(0 == n)
        return 1.;
    if(1 == n)
        return x;

    SA_ASSERT(n >= 2);
    prevprev = 1.;
    prev = x;
    for (int i=2; i <= n; ++i)
    {
        value = 2. * x * prev - prevprev;
        prevprev = prev;
        prev = value;
    }
    return value;
}

/// the name here does not *quite* match the implementation
void smpr_gauss_seidel(HypreParMatrix& A, const Vector& b, Vector& x, void *data)
{
    Vector res(x);
    Vector temp(x);
    A.Mult(x,res);
    res *= -1.0;
    res += b;

    // HypreSmoother hsmoother(A, HypreSmoother::l1GS);
    // HypreSmoother hsmoother(A, HypreSmoother::GS);
    // HypreSmoother hsmoother(A, HypreSmoother::Chebyshev);
    HypreSmoother hsmoother(A, HypreSmoother::l1GS, 3);
    // hsmoother.Mult(b,x);

    hsmoother.Mult(res,temp);
    x += temp;
}

void smpr_sym_poly(HypreParMatrix& A, const Vector& b, Vector& x, void *data)
{
    SA_ASSERT(A.GetGlobalNumRows() == A.GetGlobalNumCols());
    smpr_poly_data_t *poly_data = (smpr_poly_data_t *)data;

    Vector y;

    if (poly_data->roots2)
        y = x;

    smpr_compute_poly(A, b, x, poly_data->degree, poly_data->roots,
                      poly_data->Dinv_neg);

    if (poly_data->roots2)
    {
        smpr_compute_poly(A, b, y, poly_data->degree2, poly_data->roots2,
                          poly_data->Dinv_neg);
        x *= poly_data->weightfirst;
        y *= 1. - poly_data->weightfirst;
        x += y;
    }
}

void smpr_tg(HypreParMatrix& A, const Vector& b, Vector& x, void *data)
{
    tg_data_t *tg_data = (tg_data_t *)data;

    SA_ASSERT(tg_data);
    SA_ASSERT(tg_data->Ac);
    SA_ASSERT(tg_data->coarse_solver);

    HypreParVector X(PROC_COMM, A.GetGlobalNumCols(), x.GetData(), A.GetColStarts());
    HypreParVector B(PROC_COMM, A.GetGlobalNumRows(), b.GetData(), A.GetRowStarts());

    SA_ASSERT(!tg_data->use_w_cycle);

    // SA_RPRINTF(0,"---> smpr_tg runs with size %d, tg_data->tag = %d\n",
    //        A.M(), tg_data->tag);

    tg_cycle_atb(A, *(tg_data->Ac), *(tg_data->interp), *(tg_data->restr), B,
                 tg_data->pre_smoother, tg_data->post_smoother, X,
                 *tg_data->coarse_solver, tg_data->smoother_data);
}

double *smpr_oneminusx_poly_roots(int& nu, int *degree)
{
    nu = 1;
    *degree = 1;
    double *roots = new double[1];
    roots[0] = 1.;
    return roots;
}

double *smpr_sa_poly_roots(int& nu, int *degree)
{
    SA_ASSERT(nu >= 0);
    const double denom = (double)(2*nu + 1);
    double sin_val;
    double *roots = new double[*degree = nu];

    for (int i=1; i <= nu; ++i)
    {
        sin_val = sin(((double)i * M_PI) / denom);
        roots[i-1] = sin_val * sin_val;
    }

    return roots;
}

double *smpr_sas_poly_roots(int& nu, int *degree)
{
    if (0 == nu)
    {
        *degree = 0;
        return NULL;
    }
    SA_ASSERT(nu > 0);
    const int twonu = 2*nu;
    const double denom = (double)(2*nu + 1);
    double val;
    int i;
    double *roots = new double[*degree = twonu + nu + 1];

    for (i=0; i <= twonu; ++i)
    {
        val = cos(((double)i * M_PI) / denom);
        SA_ASSERT(i < *degree);
        roots[i] = val * val;
    }
    for (i=1; i <= nu; ++i)
    {
        val = sin(((double)i * M_PI) / denom);
        SA_ASSERT(i + twonu < *degree);
        roots[i + twonu] = val * val;
    }
    SA_ASSERT(i + twonu == *degree);

    return roots;
}

void smpr_invx_poly_init(int nu, double a, smpr_poly_data_t *poly_data)
{
    int i;
    double theta0, theta1, *roots;

    SA_ASSERT(0. < a && a < 1.);
    SA_ASSERT(nu > 1);
    SA_ALERT(pow((1. - sqrt(a)) / (1. + sqrt(a)), nu) < a / (1. - a));

    poly_data->param = a;
    poly_data->degree = nu + 1;
    poly_data->degree2 = nu - 1;

    theta0 = smpr_compute_theta0(nu, a);
    theta1 = smpr_compute_theta1(nu, a);
    poly_data->weightfirst = smpr_compute_theta(nu, a, theta0, theta1);

    if (SA_IS_OUTPUT_LEVEL(5))
    {
        SA_PRINTF("a: %g, nu: %d, theta1: %g, theta0: %g, theta{-1}: %g,"
                  " theta1 + theta0 + theta{-1} = %g, theta: %g\n", a, nu,
                  theta1, theta0, smpr_compute_theta_1(nu, a),
                  theta1 + theta0 + smpr_compute_theta_1(nu, a),
                  poly_data->weightfirst);
        SA_PRINTF("((1. - sqrt(a)) / (1. + sqrt(a)))^nu = %g < %g ="
                  " a / (1. - a) ?\n",
                  pow((1. - sqrt(a)) / (1. + sqrt(a)), nu), a / (1. - a));
    }

    roots = new double[poly_data->degree];
    roots[nu] = smpr_compute_tau0(nu, a, theta0, theta1);
    for (i=0; i < nu; ++i)
        roots[i] = smpr_compute_tauk(nu, a, i+1);
    poly_data->roots = roots;

    roots = new double[poly_data->degree2];
    for (i=0; i < nu-1; ++i)
        roots[i] = smpr_compute_tauk(nu-1, a, i+1);
    poly_data->roots2 = roots;
}

HypreParVector *smpr_update_Dinv_neg(HypreParMatrix& A,
                                     smpr_poly_data_t *poly_data)
{
    SA_ASSERT(poly_data);
    delete poly_data->Dinv_neg;

    poly_data->Dinv_neg = mbox_build_Dinv_neg_parallel_matrix(A);
    return poly_data->Dinv_neg;
}

smpr_poly_data_t *smpr_init_poly_data(HypreParMatrix& A, int nu, double param)
{
    smpr_poly_data_t *poly_data = new smpr_poly_data_t;
    SA_ASSERT(poly_data);
    SA_ASSERT(nu >= 0);
    memset(poly_data, 0, sizeof(*poly_data));

    poly_data->nu = nu;
    poly_data->degree = 0;
    poly_data->roots = NULL;
    poly_data->Dinv_neg = NULL;
    smpr_update_Dinv_neg(A, poly_data);
    poly_data->weightfirst = 1.;
    poly_data->degree2 = 0;
    poly_data->roots2 = NULL;
    poly_data->param = param;

    const int smpr_poly = SMPR_POLY_SAS;
    switch (smpr_poly)
    {
        case SMPR_POLY_ONEMINUSX:
            poly_data->roots = smpr_oneminusx_poly_roots(poly_data->nu,
                                                         &(poly_data->degree));
            break;
        case SMPR_POLY_SA:
            poly_data->roots = smpr_sa_poly_roots(poly_data->nu,
                                                  &(poly_data->degree));
            break;

        default:
            SA_ASSERT(false);
            // Fall through to get to some default behavior in bad cases.

        case SMPR_POLY_SAS:
            poly_data->roots = smpr_sas_poly_roots(poly_data->nu,
                                                   &(poly_data->degree));
            break;
        case SMPR_POLY_INVX:
            smpr_invx_poly_init(poly_data->nu, param, poly_data);
            break;
    }

    if (SA_IS_OUTPUT_LEVEL(6))
    {
        SA_PRINTF("degree: %d\n", poly_data->degree);
        PROC_STR_STREAM << "roots: ";
        for (int i=0; i < poly_data->degree; ++i)
            PROC_STR_STREAM << poly_data->roots[i] << " ";
        PROC_STR_STREAM << "\n";
        SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
        PROC_CLEAR_STR_STREAM;
        if (poly_data->roots2)
        {
            SA_PRINTF("degree2: %d\n", poly_data->degree2);
            PROC_STR_STREAM << "roots2: ";
            for (int i=0; i < poly_data->degree2; ++i)
                PROC_STR_STREAM << poly_data->roots2[i] << " ";
            PROC_STR_STREAM << "\n";
            SA_PRINTF("%s", PROC_STR_STREAM.str().c_str());
            PROC_CLEAR_STR_STREAM;
        }
    }

    return poly_data;
}

void smpr_free_poly_data(smpr_poly_data_t *data)
{
    if (!data) return;
    delete [] data->roots;
    delete data->Dinv_neg;
    delete [] data->roots2;
    delete data;
}

smpr_poly_data_t *smpr_copy_poly_data(const smpr_poly_data_t *src)
{
    if (!src) return NULL;
    smpr_poly_data_t *dst = new smpr_poly_data_t;
    dst->nu = src->nu;
    dst->degree = src->degree;
    dst->roots = helpers_copy_dbl_arr(src->roots, src->degree);
    dst->Dinv_neg = mbox_clone_parallel_vector(src->Dinv_neg);
    dst->weightfirst = src->weightfirst;
    dst->degree2 = src->degree2;
    dst->roots2 = helpers_copy_dbl_arr(src->roots2, src->degree2);
    dst->param = src->param;
    dst->S = src->S;
    dst->cDof_TruecDof = src->cDof_TruecDof;
    dst->TruecDof_cDof = src->TruecDof_cDof;
    dst->interp_data = src->interp_data;
    dst->agg_part_rels = src->agg_part_rels;
    dst->schur_smoother = src->schur_smoother;
    return dst;
}

smpr_block_data_t *smpr_init_block_data(
    const HypreParMatrix &A,
    const agg_partitioning_relations_t &agg_part_rels,
    const Mesh &mesh,
    int num_sweeps)
{
    SA_ASSERT(agg_part_rels.AE_to_dof);
    SA_ASSERT(agg_part_rels.elem_to_dof);
    SA_ASSERT(agg_part_rels.elem_to_AE);

    SparseMatrix Alocal;
    A.GetDiag(Alocal);

    const int num_blocks = agg_part_rels.AE_to_dof->Size();
    const int num_tdofs = Alocal.Height();

    smpr_block_data_t *data = new smpr_block_data_t;
    data->num_blocks = num_blocks;
    data->num_sweeps = num_sweeps;
    data->block_solvers.SetSize(num_blocks);
    data->block_solvers = nullptr;
    data->inv_sqrt_mult.SetSize(num_tdofs);

    const Table &elem_to_face = (mesh.Dimension() > 2)
        ? mesh.ElementToFaceTable()
        : mesh.ElementToEdgeTable();

    Table face_to_elem;
    Transpose(elem_to_face, face_to_elem);

    // Intermediate storage for the overlapping block dofs
    std::vector<std::pair<int, int>> flat_pairs;

    // The vector holds duplicates,
    // so we want to reserve an adequate capacity,
    // especially in three dimensions
    flat_pairs.reserve(2 * num_tdofs);

    SA_ASSERT(agg_part_rels.Dof_TrueDof != nullptr);
    SparseMatrix dof_owned_tdof;
    agg_part_rels.Dof_TrueDof->GetDiag(dof_owned_tdof);
    SA_ASSERT(dof_owned_tdof.Width() == num_tdofs);
    SA_ASSERT(agg_part_rels.AE_to_dof->Width() == dof_owned_tdof.Height());
    SA_ASSERT(agg_part_rels.elem_to_dof->Width() == dof_owned_tdof.Height());

    // Helper lambda adds true dofs after getting from local dofs
    auto append_owned_tdofs = [&](int block, int dof)
    {
        const int *owned_tdofs = dof_owned_tdof.GetRowColumns(dof);
        const int num_owned_tdofs = dof_owned_tdof.RowSize(dof);
        for (int k = 0; k < num_owned_tdofs; ++k)
            flat_pairs.emplace_back(block, owned_tdofs[k]);
    };

    for (int i = 0; i < num_blocks; ++i)
    {
        const int *dofs = agg_part_rels.AE_to_dof->GetRow(i);
        const int num_dofs = agg_part_rels.AE_to_dof->RowSize(i);
        for (int j = 0; j < num_dofs; ++j)
            append_owned_tdofs(i, dofs[j]);
    }

    const int num_faces = face_to_elem.Size();
    for (int face_index = 0; face_index < num_faces; ++face_index)
    {
        if (face_to_elem.RowSize(face_index) < 2)
            continue;

        const int *elems = face_to_elem.GetRow(face_index);
        const int elem1_index = elems[0];
        const int elem2_index = elems[1];

        const int block1 = agg_part_rels.elem_to_AE->GetRow(elem1_index)[0];
        const int block2 = agg_part_rels.elem_to_AE->GetRow(elem2_index)[0];

        if (block1 == block2)
            continue;

        // Add the overlapping tdofs to each block
        const int *elem2_dofs = agg_part_rels.elem_to_dof->GetRow(elem2_index);
        const int num_elem2_dofs = agg_part_rels.elem_to_dof->RowSize(elem2_index);
        for (int j = 0; j < num_elem2_dofs; ++j)
            append_owned_tdofs(block1, elem2_dofs[j]);

        const int *elem1_dofs = agg_part_rels.elem_to_dof->GetRow(elem1_index);
        const int num_elem1_dofs = agg_part_rels.elem_to_dof->RowSize(elem1_index);
        for (int j = 0; j < num_elem1_dofs; ++j)
            append_owned_tdofs(block2, elem1_dofs[j]);
    }

    // We need to sort the dofs so that std::unique
    // can push duplicates to the end in a single pass
    std::sort(flat_pairs.begin(), flat_pairs.end());
    flat_pairs.erase(std::unique(flat_pairs.begin(), flat_pairs.end()), flat_pairs.end());

    // The multiplicity of a DoF is the
    // number of blocks to which it belongs
    Array<int> mult(num_tdofs);
    mult = 0;

    const size_t total_unique = flat_pairs.size();
    int *block_ptr = new int[num_blocks + 1](); // Parentheses zero initialize
    int *block_tdof_map = new int[total_unique];

    for (size_t i = 0; i < total_unique; ++i)
    {
        const int block = flat_pairs[i].first;
        const int tdof = flat_pairs[i].second;
        block_ptr[block + 1] += 1;
        block_tdof_map[i] = tdof;
        mult[tdof] += 1;
    }

    for (int i = 0; i < num_blocks; ++i)
        block_ptr[i + 1] += block_ptr[i];
    data->block_tdofs.SetIJ(block_ptr, block_tdof_map, num_blocks);

    data->inv_sqrt_mult = 1.0;
    for (int i = 0; i < num_tdofs; ++i)
    {
        if (mult[i] > 1)
            data->inv_sqrt_mult(i) = 1.0 / sqrt((double)mult[i]);
    }

    DenseMatrix block_mat;
    for (int i = 0; i < num_blocks; ++i)
    {
        Array<int> block_tdofs(data->block_tdofs.GetRow(i), data->block_tdofs.RowSize(i));
        SA_ASSERT(block_tdofs.Size());
        block_mat.SetSize(block_tdofs.Size());
        Alocal.GetSubMatrix(block_tdofs, block_tdofs, block_mat);
        data->block_solvers[i] = new DenseMatrixInverse(block_mat);
    }

    return data;
}

void smpr_additive_block(
    HypreParMatrix &A,
    const mfem::Vector &b,
    Vector &x,
    void *data)
{
    SA_ASSERT(data);
    smpr_block_data_t *block_data = static_cast<smpr_block_data_t *>(data);

    SA_ASSERT(A.Width() == x.Size());
    SA_ASSERT(A.Height() == b.Size());
    SA_ASSERT(x.Size() == b.Size());
    SA_ASSERT(block_data->inv_sqrt_mult.Size() == x.Size());

    const int num_tdofs = b.Size();

    Vector res(num_tdofs);
    Vector dx(num_tdofs);
    Vector block_res;
    Vector block_dx;

    const double *weights = block_data->inv_sqrt_mult.GetData();
    const double *dx_data = dx.GetData();
    double *res_data = res.GetData();
    double *x_data = x.GetData();

    for (int i = 0; i < block_data->num_sweeps; ++i)
    {
        A.Mult(x, res);
        subtract(b, res, res);
        for (int j = 0; j < num_tdofs; ++j)
            res_data[j] *= weights[j];

        dx = 0.0;
        for (int j = 0; j < block_data->num_blocks; ++j)
        {
            const int block_size = block_data->block_tdofs.RowSize(j);
            Array<int> block_tdofs(block_data->block_tdofs.GetRow(j), block_size);

            res.GetSubVector(block_tdofs, block_res);
            block_data->block_solvers[j]->Mult(block_res, block_dx);
            dx.AddElementVector(block_tdofs, block_dx);
        }

        for (int j = 0; j < num_tdofs; ++j)
            x_data[j] += weights[j] * dx_data[j];
    }
}

void smpr_free_block_data(void *data)
{
    if (!data)
        return;
    smpr_block_data_t *block_data = static_cast<smpr_block_data_t *>(data);
    for (int i = 0; i < block_data->block_solvers.Size(); ++i)
        delete block_data->block_solvers[i];
    delete block_data;
}

smpr_block_data_t *smpr_copy_block_data(const smpr_block_data_t *src)
{
    if (!src)
        return nullptr;
    smpr_block_data_t *dst = new smpr_block_data_t;
    dst->num_blocks = src->num_blocks;
    dst->num_sweeps = src->num_sweeps;
    dst->block_tdofs = src->block_tdofs;
    dst->inv_sqrt_mult = src->inv_sqrt_mult;
    dst->block_solvers.SetSize(src->num_blocks);
    for (int i = 0; i < src->num_blocks; ++i)
    {
        dst->block_solvers[i] = src->block_solvers[i]
            ? new DenseMatrixInverse(*src->block_solvers[i])
            : nullptr;
    }

    return dst;
}

} // namespace saamge
