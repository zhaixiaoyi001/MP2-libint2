/*
 *  Original file: hartree-fock++.cc
 *  Modified by Xiaoyi Zhai, code for MP2 energy was added.
 *  Copyright (C) 2004-2026 Edward F. Valeev
 *
 *  This file is part of Libint library.
 *
 *  Libint library is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  Libint library is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with Libint library.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "hartree_fock.h"

// Explicit instantiation of template functions
template std::array<Matrix, libint2::operator_traits<libint2::Operator::overlap>::nopers>
compute_1body_ints<libint2::Operator::overlap>(const BasisSet& obs, libint2::operator_traits<libint2::Operator::overlap>::oper_params_type oparams);

template std::array<Matrix, libint2::operator_traits<libint2::Operator::kinetic>::nopers>
compute_1body_ints<libint2::Operator::kinetic>(const BasisSet& obs, libint2::operator_traits<libint2::Operator::kinetic>::oper_params_type oparams);

template std::array<Matrix, libint2::operator_traits<libint2::Operator::nuclear>::nopers>
compute_1body_ints<libint2::Operator::nuclear>(const BasisSet& obs, libint2::operator_traits<libint2::Operator::nuclear>::oper_params_type oparams);

template Matrix compute_schwarz_ints<libint2::Operator::coulomb>(
    const BasisSet& bs1, const BasisSet& bs2,
    bool use_2norm,
    libint2::operator_traits<libint2::Operator::coulomb>::oper_params_type params);

Matrix compute_do_ints(const BasisSet& bs1, const BasisSet& bs2,
                       bool use_2norm) {
  return compute_schwarz_ints<libint2::Operator::delta>(bs1, bs2, use_2norm);
}

std::tuple<shellpair_list_t, shellpair_data_t> compute_shellpairs(
    const BasisSet& bs1, const BasisSet& _bs2, const double threshold) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  using libint2::nthreads;

  // construct the overlap integral engines
  using libint2::Engine;
  std::vector<Engine> engines;
  engines.reserve(nthreads);
  engines.emplace_back(Operator::overlap,
                       std::max(bs1.max_nprim(), bs2.max_nprim()),
                       std::max(bs1.max_l(), bs2.max_l()), 0);
  engines[0].set_precision(0.);
  for (size_t i = 1; i != nthreads; ++i) {
    engines.push_back(engines[0]);
  }

  std::cout << "computing non-negligible shell-pair list ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t splist;

  std::mutex mx;

  auto compute = [&](int thread_id) {
    auto& engine = engines[thread_id];
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      mx.unlock();

      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id) continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          engines[thread_id].compute(bs1[s1], bs2[s2]);
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          mx.lock();
          splist[s1].emplace_back(s2);
          mx.unlock();
        }
      }
    }
  };  // end of compute

  libint2::parallel_do(compute);

  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2]
  // if s1 < s2 N.B. only parallelized over 1 shell index
  auto sort = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        auto& list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    }
  };  // end of sort

  libint2::parallel_do(sort);

  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  // assume shellpair data are used for Coulomb ints
  for (auto&& eng : engines) {
    eng.set(Operator::coulomb);
  }
  shellpair_data_t spdata(splist.size());
  auto make_spdata = [&](int thread_id) {
    auto schwarz_factor_evaluator = [&](const Shell& s1, size_t p1,
                                        const Shell& s2, size_t p2) -> double {
      auto& engine = engines[thread_id];
      auto& buf = engine.results();
      auto ps1 = s1.extract_primitive(p1, false);
      auto ps2 = s2.extract_primitive(p2, false);
      const auto n12 = ps1.size() * ps2.size();
      engine.compute(ps1, ps2, ps1, ps2);
      if (buf[0]) {
        Eigen::Map<const Matrix> buf_mat(buf[0], n12, n12);
        auto norm2 = screening_method == ScreeningMethod::SchwarzInf
                         ? buf_mat.lpNorm<Eigen::Infinity>()
                         : buf_mat.norm();
        return std::sqrt(norm2);
      } else
        return 0.;
    };
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        for (const auto& s2 : splist[s1]) {
          if (screening_method == ScreeningMethod::Original ||
              screening_method == ScreeningMethod::Conservative)
            spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(
                bs1[s1], bs2[s2], ln_max_engine_precision, screening_method));
          else {  // Schwarz screening of primitives
            spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(
                bs1[s1], bs2[s2], ln_max_engine_precision, screening_method,
                schwarz_factor_evaluator));
          }
        }
      }
    }
  };  // end of make_spdata

  libint2::parallel_do(make_spdata);

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)" << std::endl;

  return std::make_tuple(splist, spdata);
}

Matrix compute_2body_fock(const BasisSet& obs, const Matrix& D,
                          double precision, const Matrix& Schwarz) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  using libint2::nthreads;
  std::vector<Matrix> G(nthreads, Matrix::Zero(n, n));

  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
  Matrix D_shblk_norm =
      compute_shellblock_norm(obs, D);  // matrix of infty-norms of shell blocks

  auto fock_precision = precision;
  // standard approach is to omit *contributions* to the Fock matrix smaller
  // than fock_precision ... this relies on massive amount of error cancellation
  auto max_nprim = obs.max_nprim();
  auto needed_engine_precision = (fock_precision / D_shblk_norm.maxCoeff());
  assert(needed_engine_precision > max_engine_precision &&
         "using precomputed shell pair data limits the max engine precision"
         " ... make max_engine_precision smaller and recompile");

  // construct the 2-electron repulsion integrals engine pool
  using libint2::Engine;
  std::vector<Engine> engines(nthreads);
  engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
  engines[0].set(screening_method);
  engines[0].set_precision(
      needed_engine_precision);  // N.B. precision will be adjusted for each
                                 // shellset
  std::cout << "compute_2body_fock:precision = " << precision << std::endl;
  std::cout << "will set Engine::precision as low as = "
            << engines[0].precision() << std::endl;
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }
  std::atomic<size_t> num_ints_computed{0};

#if defined(REPORT_INTEGRAL_TIMINGS)
  std::vector<libint2::Timers<1>> timers(nthreads);
#endif

  auto shell2bf = obs.shell2bf();

  auto lambda = [&](int thread_id) {
    auto& engine = engines[thread_id];
    auto& g = G[thread_id];
    const auto& buf = engine.results();

#if defined(REPORT_INTEGRAL_TIMINGS)
    auto& timer = timers[thread_id];
    timer.clear();
    timer.set_now_overhead(25);
#endif

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      auto sp12_iter = obs_shellpair_data.at(s1).begin();

      for (const auto& s2 : obs_shellpair_list[s1]) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        const auto* sp12 = sp12_iter->get();
        ++sp12_iter;

        const auto Dnorm12 = do_schwarz_screen ? D_shblk_norm(s1, s2) : 0.;

        for (auto s3 = 0; s3 <= s1; ++s3) {
          auto bf3_first = shell2bf[s3];
          auto n3 = obs[s3].size();

          const auto Dnorm123 =
              do_schwarz_screen
                  ? std::max(D_shblk_norm(s1, s3),
                             std::max(D_shblk_norm(s2, s3), Dnorm12))
                  : 0.;

          auto sp34_iter = obs_shellpair_data.at(s3).begin();

          const auto s4_max = (s1 == s3) ? s2 : s3;
          for (const auto& s4 : obs_shellpair_list[s3]) {
            if (s4 > s4_max)
              break;  // for each s3, s4 are stored in monotonically increasing
                      // order

            // must update the iter even if going to skip s4
            const auto* sp34 = sp34_iter->get();
            ++sp34_iter;

            if ((s1234++) % nthreads != thread_id) continue;

            const auto Dnorm1234 =
                do_schwarz_screen
                    ? std::max(
                          D_shblk_norm(s1, s4),
                          std::max(D_shblk_norm(s2, s4),
                                   std::max(D_shblk_norm(s3, s4), Dnorm123)))
                    : 0.;

            if (do_schwarz_screen &&
                Dnorm1234 * Schwarz(s1, s2) * Schwarz(s3, s4) < fock_precision)
              continue;

            auto bf4_first = shell2bf[s4];
            auto n4 = obs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.start(0);
#endif

            // vary precision for each shellset to guarantee precision of the
            // contribution to the Fock matrix
            engine.set_precision(Dnorm1234 != 0. ? fock_precision / Dnorm1234
                                                 : needed_engine_precision);
            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
            const auto* buf_1234 = buf[0];
            if (buf_1234 == nullptr)
              continue;  // if all integrals screened out, skip to next quartet

            num_ints_computed += n1 * n2 * n3 * n4;

#if defined(REPORT_INTEGRAL_TIMINGS)
            timer.stop(0);
#endif

            // 1) each shell set of integrals contributes up to 6 shell sets of
            // the Fock matrix:
            //    F(a,b) += (ab|cd) * D(c,d)
            //    F(c,d) += (ab|cd) * D(a,b)
            //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
            //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
            //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
            //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
            // 2) each permutationally-unique integral (shell set) must be
            // scaled by its degeneracy,
            //    i.e. the number of the integrals/sets equivalent to it
            // 3) the end result must be symmetrized
            for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f2 = 0; f2 != n2; ++f2) {
                const auto bf2 = f2 + bf2_first;
                for (auto f3 = 0; f3 != n3; ++f3) {
                  const auto bf3 = f3 + bf3_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_1234[f1234];

                    const auto value_scal_by_deg = value * s1234_deg;

                    g(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
                    g(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
                    g(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
                    g(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
                    g(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
                    g(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }
      }
    }
  };  // end of lambda

  libint2::parallel_do(lambda);

  // accumulate contributions from all threads
  for (size_t i = 1; i != nthreads; ++i) {
    G[0] += G[i];
  }

#if defined(REPORT_INTEGRAL_TIMINGS)
  double time_for_ints = 0.0;
  for (auto& t : timers) {
    time_for_ints += t.read(0);
  }
  std::cout << "time for integrals = " << time_for_ints << std::endl;
  for (int t = 0; t != nthreads; ++t) engines[t].print_timers();
#endif

  Matrix GG = 0.5 * (G[0] + G[0].transpose());

  std::cout << "# of integrals = " << num_ints_computed << std::endl;

  // symmetrize the result and return
  return GG;
}

Matrix compute_2body_fock_general(const BasisSet& obs, const Matrix& D,
                                  const BasisSet& D_bs, bool D_is_shelldiagonal,
                                  double precision) {
  const auto n = obs.nbf();
  const auto nshells = obs.size();
  const auto n_D = D_bs.nbf();
  assert(D.cols() == D.rows() && D.cols() == n_D);

  using libint2::nthreads;
  std::vector<Matrix> G(nthreads, Matrix::Zero(n, n));

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  std::vector<Engine> engines(nthreads);
  engines[0] = Engine(libint2::Operator::coulomb,
                      std::max(obs.max_nprim(), D_bs.max_nprim()),
                      std::max(obs.max_l(), D_bs.max_l()), 0);
  engines[0].set_precision(precision);
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }
  auto shell2bf = obs.shell2bf();
  auto shell2bf_D = D_bs.shell2bf();

  auto lambda = [&](int thread_id) {
    auto& engine = engines[thread_id];
    auto& g = G[thread_id];
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
      auto bf1_first = shell2bf[s1];  // first basis function in this shell
      auto n1 = obs[s1].size();       // number of basis functions in this shell

      for (auto s2 = 0; s2 <= s1; ++s2) {
        auto bf2_first = shell2bf[s2];
        auto n2 = obs[s2].size();

        for (auto s3 = 0; s3 < D_bs.size(); ++s3) {
          auto bf3_first = shell2bf_D[s3];
          auto n3 = D_bs[s3].size();

          auto s4_begin = D_is_shelldiagonal ? s3 : 0;
          auto s4_fence = D_is_shelldiagonal ? s3 + 1 : D_bs.size();

          for (auto s4 = s4_begin; s4 != s4_fence; ++s4, ++s1234) {
            if (s1234 % nthreads != thread_id) continue;

            auto bf4_first = shell2bf_D[s4];
            auto n4 = D_bs[s4].size();

            // compute the permutational degeneracy (i.e. # of equivalents) of
            // the given shell set
            auto s12_deg = (s1 == s2) ? 1.0 : 2.0;

            if (s3 >= s4) {
              auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
              auto s1234_deg = s12_deg * s34_deg;
              // auto s1234_deg = s12_deg;
              engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                  obs[s1], obs[s2], D_bs[s3], D_bs[s4]);
              const auto* buf_1234 = buf[0];
              if (buf_1234 != nullptr) {
                for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
                  const auto bf1 = f1 + bf1_first;
                  for (auto f2 = 0; f2 != n2; ++f2) {
                    const auto bf2 = f2 + bf2_first;
                    for (auto f3 = 0; f3 != n3; ++f3) {
                      const auto bf3 = f3 + bf3_first;
                      for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                        const auto bf4 = f4 + bf4_first;

                        const auto value = buf_1234[f1234];
                        const auto value_scal_by_deg = value * s1234_deg;
                        g(bf1, bf2) += 2.0 * D(bf3, bf4) * value_scal_by_deg;
                      }
                    }
                  }
                }
              }
            }

            engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                obs[s1], D_bs[s3], obs[s2], D_bs[s4]);
            const auto* buf_1324 = buf[0];
            if (buf_1324 == nullptr)
              continue;  // if all integrals screened out, skip to next quartet

            for (auto f1 = 0, f1324 = 0; f1 != n1; ++f1) {
              const auto bf1 = f1 + bf1_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f2 = 0; f2 != n2; ++f2) {
                  const auto bf2 = f2 + bf2_first;
                  for (auto f4 = 0; f4 != n4; ++f4, ++f1324) {
                    const auto bf4 = f4 + bf4_first;

                    const auto value = buf_1324[f1324];
                    const auto value_scal_by_deg = value * s12_deg;
                    g(bf1, bf2) -= D(bf3, bf4) * value_scal_by_deg;
                  }
                }
              }
            }
          }
        }
      }
    }
  };  // thread lambda

  libint2::parallel_do(lambda);

  // accumulate contributions from all threads
  for (size_t i = 1; i != nthreads; ++i) {
    G[0] += G[i];
  }

  // symmetrize the result and return
  return 0.5 * (G[0] + G[0].transpose());
}


/**
 * 计算 MP2 相关能
 * C_occ: 占据轨道系数 (n_ao x n_occ)
 * C_virt: 虚拟轨道系数 (n_ao x n_virt)
 * eps: 轨道能量向量
 */
double cal_mp2_in_memory(const BasisSet& obs, 
                            const Matrix& C_occ, 
                            const Matrix& C_virt, 
                            const Eigen::VectorXd& eps) {
    const auto n_ao = obs.nbf();
    const auto n_occ = C_occ.cols();
    const auto n_virt = C_virt.cols();
    const auto nshells = obs.size();
    auto shell2bf = obs.shell2bf();

    // 1. 分配存储 (ia|jb) 积分的内存: O^2 * V^2
    // 索引公式: (i * n_virt + a) * (n_occ * n_virt) + (j * n_virt + b)
    // 即 i*O*V*V + a*O*V + j*V + b
    std::vector<double> mo_ints(n_occ * n_virt * n_occ * n_virt, 0.0);
    std::mutex mtx; // 用于多线程更新 mo_ints

    using libint2::nthreads;
    using libint2::Engine;
    std::vector<Engine> engines(nthreads);
    engines[0] = Engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
    engines[0].set_precision(max_engine_precision);
    std::cout << "Starting MP2 Integral Transformation..." << std::endl;
    std::cout << "will set Engine::precision = "
              << engines[0].precision() << std::endl;
    for (size_t i = 1; i != nthreads; ++i) {
      engines[i] = engines[0];
    }

    // 2. 第一步变换：(mu nu | lambda sigma) -> (mu nu | j b)
    // 每次处理一对 (mu, nu) 壳层块，计算其对应的所有 (j, b)
    auto transform_lambda = [&](int thread_id) {
        auto& engine = engines[thread_id];
        const auto& buf = engine.results();

        for (auto s1 = 0l; s1 < nshells; ++s1) {
            for (auto s2 = 0l; s2 < nshells; ++s2) {
                
                // 仅并行化最外层壳层对
                if ((s1 * nshells + s2) % libint2::nthreads != thread_id) continue;

                int n1 = obs[s1].size();
                int n2 = obs[s2].size();
                
                // lambda,sigma 构成 N x N 矩阵
                for (int f1 = 0; f1 < n1; ++f1) {
                    for (int f2 = 0; f2 < n2; ++f2) {
                        int mu = shell2bf[s1] + f1;
                        int nu = shell2bf[s2] + f2;

                        Matrix eri_ao = Matrix::Zero(n_ao, n_ao);

                        // 内部循环计算所有 (lambda, sigma) 贡献给当前的 (mu, nu)
                        for (auto s3 = 0l; s3 < nshells; ++s3) {
                            for (auto s4 = 0l; s4 < nshells; ++s4) {
                                engine.compute2<Operator::coulomb, BraKet::xx_xx, 0>(
                                    obs[s1], obs[s2], obs[s3], obs[s4]);
                                
                                if (buf[0] == nullptr) continue;
                                
                                int n3 = obs[s3].size();
                                int n4 = obs[s4].size();
                                // 提取 buffer 对应当前 f1, f2 的值
                                for (int f3 = 0; f3 < n3; ++f3) {
                                    for (int f4 = 0; f4 < n4; ++f4) {
                                        int idx = ((f1 * n2 + f2) * n3 + f3) * n4 + f4;
                                        eri_ao(shell2bf[s3] + f3, shell2bf[s4] + f4) = buf[0][idx];
                                    }
                                }
                            }
                        }

                        // 执行变换: (mu nu | j b) = C_occ^T (n_occ*n) * ERI(mu,nu) (n*n) * C_virt (n*n_virt)
                        // 结果为 n_occ x n_virt 矩阵 half_mo
                        Matrix half_mo = C_occ.transpose() * eri_ao * C_virt;

                        // 第二步变换：将 mu, nu 变换为 i, a
                        // 直接收缩
                        for (int i = 0; i < n_occ; ++i) {
                            for (int a = 0; a < n_virt; ++a) {
                                double c_ia = C_occ(mu, i) * C_virt(nu, a);
                                
                                for (int j = 0; j < n_occ; ++j) {
                                    for (int b = 0; b < n_virt; ++b) {
                                        double val = c_ia * half_mo(j, b);
                                        // 索引映射并累加到全局 mo_ints
                                        size_t mo_idx = (size_t(i) * n_virt + a) * (n_occ * n_virt) + (j * n_virt + b);
                                        
                                        mtx.lock();
                                        mo_ints[mo_idx] += val;
                                        mtx.unlock();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    libint2::parallel_do(transform_lambda);

    // 计算 MP2 能量
    // E = sum_{iajb} (ia|jb) * [2(ia|jb) - (ib|ja)] / (ei + ej - ea - eb)
    double emp2 = 0.0;

    #pragma omp parallel for reduction(+:emp2) collapse(4) schedule(static)
    for (int i = 0; i < n_occ; ++i) {
        for (int j = 0; j < n_occ; ++j) {
            for (int a = 0; a < n_virt; ++a) {
                for (int b = 0; b < n_virt; ++b) {
                    size_t idx1 = (size_t(i) * n_virt + a) * (n_occ * n_virt) + (j * n_virt + b);
                    size_t idx2 = (size_t(i) * n_virt + b) * (n_occ * n_virt) + (j * n_virt + a);
                    
                    double iajb = mo_ints[idx1];
                    double ibja = mo_ints[idx2];
                    double denominator = eps(i) + eps(j) - eps(n_occ + a) - eps(n_occ + b);
                    
                    emp2 += iajb * (2.0 * iajb - ibja) / denominator;
                }
            }
        }
    }

    return emp2;
}
