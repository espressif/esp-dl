#include "dl_feat_verification_engine.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <eigen3/Eigen/Dense>

namespace dl {
namespace feat {

FeatSubspace::FeatSubspace() : m_is_ready(false), m_embedding_dim(0), m_num_embeddings(0)
{
}

FeatSubspace::~FeatSubspace()
{
    clear();
}

void FeatSubspace::clear()
{
    m_mean.clear();
    m_basis.clear();
    m_variances.clear();
    m_is_ready = false;
    m_num_embeddings = 0;
    m_embedding_dim = 0;
}

void FeatSubspace::build(const std::vector<const float *> &enroll_embeds, int embedding_dim)
{
    clear();

    int num_samples = enroll_embeds.size();
    m_embedding_dim = embedding_dim;
    m_num_embeddings = std::min(num_samples - 1, embedding_dim);

    // Build matrix X on heap to avoid stack overflow
    Eigen::MatrixXf *X = new Eigen::MatrixXf(num_samples, embedding_dim);
    for (int row = 0; row < num_samples; row++) {
        for (int col = 0; col < embedding_dim; col++) {
            (*X)(row, col) = enroll_embeds[row][col];
        }
    }

    // Compute mean and center data
    Eigen::VectorXf mean_vec = X->colwise().mean();
    m_mean.resize(embedding_dim);
    for (int i = 0; i < embedding_dim; i++) m_mean[i] = mean_vec(i);
    *X = X->rowwise() - mean_vec.transpose();

    // SVD: X = U * S * Vt (thin SVD)
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(*X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::VectorXf S = svd.singularValues();
    Eigen::MatrixXf V = svd.matrixV();

    delete X;

    // Store variances
    m_variances.resize(m_num_embeddings);
    for (int k = 0; k < m_num_embeddings; k++) m_variances[k] = S(k);

    // Store basis (V columns)
    m_basis.resize(m_num_embeddings * embedding_dim);
    for (int k = 0; k < m_num_embeddings; k++) {
        for (int i = 0; i < embedding_dim; i++) m_basis[k * embedding_dim + i] = V(i, k);
    }

    m_is_ready = true;
}

float FeatSubspace::compute_neg_distance(const float *target_embed, float inv_norm) const
{
    std::vector<float> coords(m_num_embeddings, 0.0f);

    // Center: diff = x_norm - mean, Project: coords = basis.T @ diff
    for (int k = 0; k < m_num_embeddings; k++) {
        const float *basis_row = &m_basis[k * m_embedding_dim];
        for (int i = 0; i < m_embedding_dim; i++) {
            coords[k] += basis_row[i] * (target_embed[i] * inv_norm - m_mean[i]);
        }
    }

    // Reconstruct: proj = basis @ coords, compute residual on-the-fly
    float residual_norm_sq = 0.0f;
    for (int i = 0; i < m_embedding_dim; i++) {
        float proj_i = 0.0f;
        for (int k = 0; k < m_num_embeddings; k++) {
            proj_i += m_basis[k * m_embedding_dim + i] * coords[k];
        }
        float diff_i = target_embed[i] * inv_norm - m_mean[i];
        float r = diff_i - proj_i;
        residual_norm_sq += r * r;
    }

    // Return normalized negative value between -1 and 0
    return -sqrtf(residual_norm_sq / 2.0f);
}

void FeatSubspace::load(
    bool ready, int embedding_dim, int num_embeddings, const float *mean, const float *basis, const float *variances)
{
    m_is_ready = ready;
    m_embedding_dim = embedding_dim;
    m_num_embeddings = num_embeddings;
    m_mean.assign(mean, mean + embedding_dim);
    m_basis.assign(basis, basis + num_embeddings * embedding_dim);
    m_variances.assign(variances, variances + num_embeddings);
}

} // namespace feat
} // namespace dl
