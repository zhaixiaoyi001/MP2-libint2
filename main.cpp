/*
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

int main(int argc, char* argv[]) {
  using std::cerr;
  using std::cout;
  using std::endl;

  try {
    /*** =========================== ***/
    /*** initialize molecule         ***/
    /*** =========================== ***/

    // read geometry from a file; by default read from h2o.xyz, else take
    // filename (.xyz) from the command line
    const auto filename = (argc > 1) ? argv[1] : "h2o.xyz";
    const auto basisname = (argc > 2) ? argv[2] : "aug-cc-pVDZ";
    bool do_density_fitting = false;
    double cholesky_threshold = 1e-4;
    std::vector<Atom> atoms = read_geometry(filename);

    // set up thread pool
    {
      using libint2::nthreads;
      auto nthreads_cstr = getenv("LIBINT_NUM_THREADS");
      nthreads = 1;
      if (nthreads_cstr && strcmp(nthreads_cstr, "")) {
        std::istringstream iss(nthreads_cstr);
        iss >> nthreads;
        if (nthreads > 1 << 16 || nthreads <= 0) nthreads = 1;
      }
#if defined(_OPENMP)
      omp_set_num_threads(nthreads);
#endif
      std::cout << "Will scale over " << nthreads
#if defined(_OPENMP)
                << " OpenMP"
#else
                << " C++11"
#endif
                << " threads" << std::endl;
    }

    // count the number of electrons
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;
    cout << "# of electrons = " << nelectron << endl;

    // compute the nuclear repulsion energy
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij * xij + yij * yij + zij * zij;
        auto r = sqrt(r2);
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }
    cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc
         << endl;

    libint2::Shell::do_enforce_unit_normalization(false);

    cout << "Atomic Cartesian coordinates (a.u.):" << endl;
    for (const auto& a : atoms)
      std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z
                << std::endl;

    BasisSet obs(basisname, atoms);
    cout << "orbital basis set rank = " << obs.nbf() << endl;

    // initializes the Libint integrals library ... now ready to compute
    libint2::initialize();

    /*** =========================== ***/
    /*** compute 1-e integrals       ***/
    /*** =========================== ***/

    // compute OBS non-negligible shell-pair list
    {
      const auto tstart = std::chrono::high_resolution_clock::now();
      std::tie(obs_shellpair_list, obs_shellpair_data) =
          compute_shellpairs(obs);
      size_t nsp = 0;
      for (auto& sp : obs_shellpair_list) {
        nsp += sp.second.size();
      }
      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;
      std::cout << "computed shell-pair data in " << time_elapsed.count()
                << " seconds: # of {all,non-negligible} shell-pairs = {"
                << obs.size() * (obs.size() + 1) / 2 << "," << nsp << "}"
                << std::endl;
    }

    // compute one-body integrals
    auto S = compute_1body_ints<Operator::overlap>(obs)[0];
    auto T = compute_1body_ints<Operator::kinetic>(obs)[0];
    auto V = compute_1body_ints<Operator::nuclear>(
        obs, libint2::make_point_charges(atoms))[0];
    Matrix H = T + V;
    T.resize(0, 0);
    V.resize(0, 0);

    // compute orthogonalizer X such that X.transpose() . S . X = I
    Matrix X, Xinv;
    double XtX_condition_number;  // condition number of "re-conditioned"
                                  // overlap obtained as Xinv.transpose() . Xinv
    // one should think of columns of Xinv as the conditioned basis
    // Re: name ... cond # (Xinv.transpose() . Xinv) = cond # (X.transpose() .
    // X)
    // by default assume can manage to compute with condition number of S <=
    // 1/eps
    // this is probably too optimistic, but in well-behaved cases even 10^11 is
    // OK
    double S_condition_number_threshold =
        1.0 / std::numeric_limits<double>::epsilon();
    std::tie(X, Xinv, XtX_condition_number) =
        conditioning_orthogonalizer(S, S_condition_number_threshold);

    Matrix D;
    Matrix C;
    Matrix C_occ;
    Matrix evals;
    {  // use SOAD as the guess density
      const auto tstart = std::chrono::high_resolution_clock::now();

      auto D_minbs =
          compute_soad_guess(atoms);  // compute guess in minimal basis
      BasisSet minbs("STO-3G", atoms);
      if (minbs == obs)
        D = D_minbs;
      else {  // if basis != minimal basis, map non-representable SOAD guess
              // into the AO basis
              // by diagonalizing a Fock matrix
        std::cout << "projecting SOAD into AO basis ... ";
        auto F = H;
        F += compute_2body_fock_general(
            obs, D_minbs, minbs, true /* SOAD_D_is_shelldiagonal */,
            std::numeric_limits<double>::epsilon()  // this is cheap, no reason
                                                    // to be cheaper
        );

        // solve F C = e S C by (conditioned) transformation to F' C' = e C',
        // where
        // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
        Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F * X);
        C = X * eig_solver.eigenvectors();

        // compute density, D = C(occ) . C(occ)T
        C_occ = C.leftCols(ndocc);
        D = C_occ * C_occ.transpose();

        const auto tstop = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> time_elapsed = tstop - tstart;
        std::cout << "done (" << time_elapsed.count() << " s)" << std::endl;
      }
    }

    // pre-compute data for Schwarz bounds
    auto K = compute_schwarz_ints<>(obs);


    /*** =========================== ***/
    /***          SCF loop           ***/
    /*** =========================== ***/

    const auto maxiter = 100;
    const auto conv = 1e-12;
    auto iter = 0;
    auto rms_error = 1.0;
    auto ediff_rel = 0.0;
    auto ehf = 0.0;
    auto n2 = D.cols() * D.rows();
    libint2::DIIS<Matrix> diis(3);  // start DIIS on second iteration

    // prepare for incremental Fock build ...
    Matrix D_diff = D;
    Matrix F = H;
    bool reset_incremental_fock_formation = false;
    bool incremental_Fbuild_started = false;
    double start_incremental_F_threshold = 1e-5;
    double next_reset_threshold = 0.0;
    size_t last_reset_iteration = 0;
    // ... unless doing DF, then use MO coefficients, hence not "incremental"
    if (do_density_fitting) start_incremental_F_threshold = 0.0;

    do {
      const auto tstart = std::chrono::high_resolution_clock::now();
      ++iter;

      // Last iteration's energy and density
      auto ehf_last = ehf;
      Matrix D_last = D;

      if (not incremental_Fbuild_started &&
          rms_error < start_incremental_F_threshold) {
        incremental_Fbuild_started = true;
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter - 1;
        next_reset_threshold = rms_error / 1e1;
        std::cout << "== started incremental fock build" << std::endl;
      }
      if (reset_incremental_fock_formation || not incremental_Fbuild_started) {
        F = H;
        D_diff = D;
      }
      if (reset_incremental_fock_formation && incremental_Fbuild_started) {
        reset_incremental_fock_formation = false;
        last_reset_iteration = iter;
        next_reset_threshold = rms_error / 1e1;
        std::cout << "== reset incremental fock build" << std::endl;
      }

      // build a new Fock matrix
      if (not do_density_fitting) {
        // totally empirical precision variation, involves the condition number
        const auto precision_F = std::min(
            std::min(1e-3 / XtX_condition_number, 1e-7),
            std::max(rms_error / 1e4, std::numeric_limits<double>::epsilon()));
        F += compute_2body_fock(obs, D_diff, precision_F, K);
      }

      // compute HF energy with the non-extrapolated Fock matrix
      ehf = D.cwiseProduct(H + F).sum();
      ediff_rel = std::abs((ehf - ehf_last) / ehf);

      // compute SCF error
      Matrix FD_comm = F * D * S - S * D * F;
      rms_error = FD_comm.norm() / n2;
      if (rms_error < next_reset_threshold || iter - last_reset_iteration >= 8)
        reset_incremental_fock_formation = true;

      // DIIS extrapolate F
      Matrix F_diis = F;  // extrapolated F cannot be used in incremental Fock
                          // build; only used to produce the density
                          // make a copy of the unextrapolated matrix
      diis.extrapolate(F_diis, FD_comm);

      // solve F C = e S C by (conditioned) transformation to F' C' = e C',
      // where
      // F' = X.transpose() . F . X; the original C is obtained as C = X . C'
      Eigen::SelfAdjointEigenSolver<Matrix> eig_solver(X.transpose() * F_diis *
                                                       X);
      evals = eig_solver.eigenvalues();
      C = X * eig_solver.eigenvectors();

      // compute density, D = C(occ) . C(occ)T
      C_occ = C.leftCols(ndocc);
      D = C_occ * C_occ.transpose();
      D_diff = D - D_last;

      const auto tstop = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> time_elapsed = tstop - tstart;

      if (iter == 1)
        std::cout << "\n\nIter         E(HF)                 D(E)/E         "
                     "RMS([F,D])/nn       Time(s)\n";
      printf(" %02d %20.12f %20.12e %20.12e %10.5lf\n", iter, ehf + enuc,
             ediff_rel, rms_error, time_elapsed.count());

    } while (((ediff_rel > conv) || (rms_error > conv)) && (iter < maxiter));

    printf("** Hartree-Fock energy = %20.12f\n", ehf + enuc);

    // try dumping orbs to a molden file
    try {
      Eigen::VectorXd occs(C.cols());
      occs.setZero();
      for (size_t o = 0; o != ndocc; ++o) occs[o] = 2.0;

      libint2::molden::Export xport(atoms, obs, C, occs, evals);
      const auto molden_file_name = "hf++.molden";
      std::ofstream molden_file(molden_file_name);
      xport.write(molden_file);
      std::cout << "wrote orbitals to " << molden_file_name << std::endl;
    } catch (std::logic_error& e) {
      if (std::strstr(e.what(), "molden::Export") == nullptr) throw e;
    }

  libint2::finalize();  // done with libint

  }  // end of try block; if any exceptions occurred, report them and exit
     // cleanly

  catch (const char* ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  } catch (std::string& ex) {
    cerr << "caught exception: " << ex << endl;
    return 1;
  } catch (std::exception& ex) {
    cerr << ex.what() << endl;
    return 1;
  } catch (...) {
    cerr << "caught unknown exception\n";
    return 1;
  }

  return 0;
}
