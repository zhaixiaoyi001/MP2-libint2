/*
 *  By Xiaoyi Zhai, code for ao2mo integral transformation.
 */

#include "hartree_fock.h"

std::vector<double> ao2mo_incore_all(const BasisSet& obs,
                                    const Matrix& C_occ,
                                    const Matrix& C_virt) {
    using libint2::Engine;
    using libint2::Operator;
    using libint2::BraKet;
    
    const auto n_ao = obs.nbf();
    const auto n_occ = C_occ.cols();
    const auto n_virt = C_virt.cols();
    const size_t ov = n_occ * n_virt;
    const auto nshells = obs.size();
    auto shell2bf = obs.shell2bf();

    int nthreads = libint2::nthreads;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    for (size_t i = 1; i != nthreads; ++i) engines[i] = engines[0];

    std::cout << "Starting AO2MO Transformation..." << std::endl;

    // --- 阶段一：半变换 (AO, AO | lambda, sigma) -> (AO, AO | j, b) ---

    // 存储半变换结果 (mu, nu | j, b) -> 空间复杂度 O(N^2 * OV)
    std::vector<double> half_mo(n_ao * n_ao * ov, 0.0);

    auto transform_lsjb = [&](int thread_id) {
        auto& engine = engines[thread_id];
        const auto& buf = engine.results();

        for (auto s1 = 0l; s1 < nshells; ++s1) {
            int n1 = obs[s1].size();
            int bf1_start = shell2bf[s1];
            
            for (auto s2 = 0l; s2 < nshells; ++s2) {
                // 按壳层分配负载
                if ((s1 * nshells + s2) % nthreads != thread_id) continue;
                
                int n2 = obs[s2].size();
                int bf2_start = shell2bf[s2];

                // 为当前 (s1, s2) 壳层对下的所有 (mu, nu) 基函数对分配 V_munu 缓存
                // 大小为 n1 * n2，每个元素是一个 N_ao x N_ao 的矩阵
                std::vector<Matrix> V_munu_list(n1 * n2, Matrix::Zero(n_ao, n_ao));

                // 1. 遍历 s3, s4，计算积分并拼装完整的 V_munu
                for (auto s3 = 0l; s3 < nshells; ++s3) {
                    int n3 = obs[s3].size();
                    int bf3_start = shell2bf[s3];
                    for (auto s4 = 0l; s4 < nshells; ++s4) {
                        int n4 = obs[s4].size(); 
                        int bf4_start = shell2bf[s4];
                        
                        // 积分计算仅执行一次
                        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4]);
                        const auto* buf_ptr = buf[0];
                        if (buf_ptr == nullptr) continue;

                        // 将积分数据分发到对应的 V_munu 矩阵中
                        for (int f1 = 0; f1 < n1; ++f1) {
                            for (int f2 = 0; f2 < n2; ++f2) {
                                int f12_idx = f1 * n2 + f2;
                                for (int f3 = 0; f3 < n3; ++f3) {
                                    for (int f4 = 0; f4 < n4; ++f4) {
                                        V_munu_list[f12_idx](bf3_start + f3, bf4_start + f4) = 
                                            buf_ptr[(f1 * n2 + f2) * (n3 * n4) + (f3 * n4 + f4)];
                                    }
                                }
                            }
                        }
                    }
                }

                // 2. 积分拼装完成后，再进行矩阵乘法（每个 mu, nu 对仅执行一次）
                for (int f1 = 0; f1 < n1; ++f1) {
                    int mu = bf1_start + f1;
                    for (int f2 = 0; f2 < n2; ++f2) {
                        int nu = bf2_start + f2;
                        int f12_idx = f1 * n2 + f2;

                        // (mu, nu | lambda, sigma) -> (mu, nu | j, b)
                        Matrix B_munu = C_occ.transpose() * V_munu_list[f12_idx] * C_virt;
                        
                        // 将结果拷入连续内存
                        memcpy(&half_mo[(mu * n_ao + nu) * ov], B_munu.data(), ov * sizeof(double));
                    }
                }
            }
        }
    };

    libint2::parallel_do(transform_lsjb);

    // --- 阶段二：全变换 (mu, nu | j, b) -> (i, a | j, b) ---
    // 复杂度：O(OV * N^3) = O(N^5)
    std::vector<double> mo_ints(ov * ov, 0.0);
    auto transform_all = [&](int thread_id) {
        for (size_t jb = 0; jb < ov; ++jb) {
            if (jb % nthreads != thread_id) continue;

            // 取出所有 (mu, nu) 组成的 N_ao x N_ao 矩阵（固定 j, b）
            Matrix B_jb = Matrix::Zero(n_ao, n_ao);
            for (int mu = 0; mu < n_ao; ++mu) {
                for (int nu = 0; nu < n_ao; ++nu) {
                    B_jb(mu, nu) = half_mo[(mu * n_ao + nu) * ov + jb];
                }
            }

            // 最后一步变换：(mu, nu) -> (i, a)
            Matrix M_jb = C_occ.transpose() * B_jb * C_virt; // 结果大小为 [n_occ x n_virt]

            // 存入最终 MO 积分表
            for (int ia = 0; ia < ov; ++ia) {
                mo_ints[ia * ov + jb] = M_jb.data()[ia];
            }
        }
    };

    libint2::parallel_do(transform_all);

    return mo_ints;
}


std::vector<double> ao2mo_incore_batch(const BasisSet& obs,
                                    const Matrix& C_occ,
                                    const Matrix& C_virt,
                                    const Matrix& C_virt_batch) {
    using libint2::Engine;
    using libint2::Operator;
    using libint2::BraKet;
    
    const auto n_ao = obs.nbf();
    const auto n_occ = C_occ.cols();
    const auto n_virt = C_virt.cols();
    const auto batch_size = C_virt_batch.cols();
    const size_t o_batch = n_occ * batch_size;
    const size_t ov = n_occ * n_virt;
    const auto nshells = obs.size();
    auto shell2bf = obs.shell2bf();

    int nthreads = libint2::nthreads;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    for (size_t i = 1; i != nthreads; ++i) engines[i] = engines[0];

    std::cout << "Starting AO2MO Transformation..." << std::endl;

    // --- 阶段一：半变换 (AO, AO | lambda, sigma) -> (AO, AO | j, b) ---
    // b 固定为一个batch
    // 存储半变换结果 (mu, nu | j, b) -> 空间复杂度 O(N^2 * O * batch_size)
    // 索引方式 b + j * batch_size + nu * O * batch_size + mu * N * O * batch_size
    std::vector<double> half_mo(n_ao * n_ao * o_batch, 0.0);

    auto transform_lsjb = [&](int thread_id) {
        auto& engine = engines[thread_id];
        const auto& buf = engine.results();

        for (auto s1 = 0l; s1 < nshells; ++s1) {
            int n1 = obs[s1].size();
            int bf1_start = shell2bf[s1];
            
            for (auto s2 = 0l; s2 < nshells; ++s2) {
                // 按壳层分配负载
                if ((s1 * nshells + s2) % nthreads != thread_id) continue;
                
                int n2 = obs[s2].size();
                int bf2_start = shell2bf[s2];

                // 为当前 (s1, s2) 壳层对下的所有 (mu, nu) 基函数对分配 V_munu 缓存
                // 大小为 n1 * n2，每个元素是一个 N_ao x N_ao 的矩阵
                std::vector<Matrix> V_munu_list(n1 * n2, Matrix::Zero(n_ao, n_ao));

                // 1. 遍历 s3, s4，计算积分并拼装完整的 V_munu
                for (auto s3 = 0l; s3 < nshells; ++s3) {
                    int n3 = obs[s3].size();
                    int bf3_start = shell2bf[s3];
                    for (auto s4 = 0l; s4 < nshells; ++s4) {
                        int n4 = obs[s4].size(); 
                        int bf4_start = shell2bf[s4];
                        
                        // 积分计算仅执行一次
                        engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4]);
                        const auto* buf_ptr = buf[0];
                        if (buf_ptr == nullptr) continue;

                        // 将积分数据分发到对应的 V_munu 矩阵中
                        for (int f1 = 0; f1 < n1; ++f1) {
                            for (int f2 = 0; f2 < n2; ++f2) {
                                int f12_idx = f1 * n2 + f2;
                                for (int f3 = 0; f3 < n3; ++f3) {
                                    for (int f4 = 0; f4 < n4; ++f4) {
                                        V_munu_list[f12_idx](bf3_start + f3, bf4_start + f4) = 
                                            buf_ptr[(f1 * n2 + f2) * (n3 * n4) + (f3 * n4 + f4)];
                                    }
                                }
                            }
                        }
                    }
                }

                // 2. 积分拼装完成后，再进行矩阵乘法（每个 mu, nu 对仅执行一次）
                for (int f1 = 0; f1 < n1; ++f1) {
                    int mu = bf1_start + f1;
                    for (int f2 = 0; f2 < n2; ++f2) {
                        int nu = bf2_start + f2;
                        int f12_idx = f1 * n2 + f2;

                        // (mu, nu | lambda, sigma) -> (mu, nu | j, b)
                        Matrix B_munu = C_occ.transpose() * V_munu_list[f12_idx] * C_virt_batch;
                        
                        // 将结果拷入连续内存
                        memcpy(&half_mo[(mu * n_ao + nu) * o_batch], B_munu.data(), o_batch * sizeof(double));
                    }
                }
            }
        }
    };

    libint2::parallel_do(transform_lsjb);

    // --- 阶段二：全变换 (mu, nu | j, b) -> (i, a | j, b) ---
    // 复杂度：O(OV * N^3) = O(N^5)
    std::vector<double> mo_ints(ov * o_batch, 0.0);
    auto transform_all = [&](int thread_id) {
        for (size_t jb = 0; jb < o_batch; ++jb) {
            if (jb % nthreads != thread_id) continue;

            // 取出所有 (mu, nu) 组成的 N_ao x N_ao 矩阵（固定 j, b）
            Matrix B_jb = Matrix::Zero(n_ao, n_ao);
            for (int mu = 0; mu < n_ao; ++mu) {
                for (int nu = 0; nu < n_ao; ++nu) {
                    B_jb(mu, nu) = half_mo[(mu * n_ao + nu) * o_batch + jb];
                }
            }

            // 最后一步变换：(mu, nu) -> (i, a)
            Matrix M_jb = C_occ.transpose() * B_jb * C_virt; // 结果大小为 [n_occ x n_virt]

            // 存入最终 MO 积分表
            for (int ia = 0; ia < ov; ++ia) {
                mo_ints[ia * o_batch + jb] = M_jb.data()[ia];
            }
        }
    };

    libint2::parallel_do(transform_all);

    return mo_ints;
}
