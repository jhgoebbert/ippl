// Landau Damping Test, variant with mixed precision.
// In order to avoid eccessive error when scattering from grid-points to the particles,
// the charge and the scalar field are kept in double precision. The Mesh object is also
// in double precision, as it leads to a higher precision without affecting memory negatively.
// Everything else (namely the vector field E and the particle position) are in single
// precision, since the choice increases memory saving, without losing precision.
//    Usage:
//     srun ./LandauDamping
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type e.g., FFT
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./LandauDamping 128 128 128 10000 10 FFT 0.01 2.0 --info 10
//
// Copyright (c) 2021, Sriramkrishnan Muralikrishnan,
// Paul Scherrer Institut, Villigen PSI, Switzerland
// All rights reserved
//
// This file is part of IPPL.
//
// IPPL is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// You should have received a copy of the GNU General Public License
// along with IPPL. If not, see <https://www.gnu.org/licenses/>.
//

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

template <typename T>
struct Newton1D {
    double tol   = 1e-12;
    int max_iter = 20;
    T pi         = Kokkos::numbers::pi_v<T>;

    T k, alpha, u;

    KOKKOS_INLINE_FUNCTION Newton1D() {}

    KOKKOS_INLINE_FUNCTION Newton1D(const T& k_, const T& alpha_, const T& u_)
        : k(k_)
        , alpha(alpha_)
        , u(u_) {}

    KOKKOS_INLINE_FUNCTION ~Newton1D() {}

    KOKKOS_INLINE_FUNCTION T f(T& x) {
        T F;
        F = x + (alpha * (Kokkos::sin(k * x) / k)) - u;
        return F;
    }

    KOKKOS_INLINE_FUNCTION T fprime(T& x) {
        T Fprime;
        Fprime = 1 + (alpha * Kokkos::cos(k * x));
        return Fprime;
    }

    KOKKOS_FUNCTION
    void solve(T& x) {
        int iterations = 0;
        while (iterations < max_iter && Kokkos::fabs(f(x)) > tol) {
            x = x - (f(x) / fprime(x));
            iterations += 1;
        }
    }
};

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type  = typename ippl::detail::ViewType<T, 1>::view_type;
    using value_type = typename T::value_type;
    // Output View for the random numbers
    view_type x, v;

    // The GeneratorPool
    GeneratorPool rand_pool;

    value_type alpha;

    T k, minU, maxU;

    // Initialize all members
    generate_random(view_type x_, view_type v_, GeneratorPool rand_pool_, value_type& alpha_, T& k_,
                    T& minU_, T& maxU_)
        : x(x_)
        , v(v_)
        , rand_pool(rand_pool_)
        , alpha(alpha_)
        , k(k_)
        , minU(minU_)
        , maxU(maxU_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        value_type u;
        for (unsigned d = 0; d < Dim; ++d) {
            u       = rand_gen.drand(minU[d], maxU[d]);
            x(i)[d] = u / (1 + alpha);
            Newton1D<value_type> solver(k[d], alpha, u);
            solver.solve(x(i)[d]);
            v(i)[d] = rand_gen.normal(0.0, 1.0);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

float CDF(const float& x, const float& alpha, const float& k) {
    float cdf = x + (alpha / k) * std::sin(k * x);
    return cdf;
}

KOKKOS_FUNCTION
double PDF(const Vector_t<double, Dim>& xvec, const double& alpha, const Vector_t<double, Dim>& kw,
           const unsigned Dim) {
    double pdf = 1.0;

    for (unsigned d = 0; d < Dim; ++d) {
        pdf *= (1.0 + alpha * Kokkos::cos(kw[d] * xvec[d]));
    }
    return pdf;
}

const char* TestName = "LandauDamping";

int main(int argc, char* argv[]) {
    Ippl ippl(argc, argv);

    Inform msg("LandauDamping");
    Inform msg2all("LandauDamping", INFORM_ALL_NODES);

    auto start = std::chrono::high_resolution_clock::now();
    int arg    = 1;

    Vector_t<int, Dim> nr;
    for (unsigned d = 0; d < Dim; d++) {
        nr[d] = std::atoi(argv[arg++]);
    }

    static IpplTimings::TimerRef mainTimer           = IpplTimings::getTimer("total");
    static IpplTimings::TimerRef particleCreation    = IpplTimings::getTimer("particlesCreation");
    static IpplTimings::TimerRef dumpDataTimer       = IpplTimings::getTimer("dumpData");
    static IpplTimings::TimerRef PTimer              = IpplTimings::getTimer("pushVelocity");
    static IpplTimings::TimerRef RTimer              = IpplTimings::getTimer("pushPosition");
    static IpplTimings::TimerRef updateTimer         = IpplTimings::getTimer("update");
    static IpplTimings::TimerRef DummySolveTimer     = IpplTimings::getTimer("solveWarmup");
    static IpplTimings::TimerRef SolveTimer          = IpplTimings::getTimer("solve");
    static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

    IpplTimings::startTimer(mainTimer);

    const size_type totalP = std::atoll(argv[arg++]);
    const unsigned int nt  = std::atoi(argv[arg++]);

    msg << "Landau damping" << endl << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

    using bunch_type = ChargedParticles<PLayout_t<float, Dim>, float, Dim>;

    std::unique_ptr<bunch_type> P;

    ippl::NDIndex<Dim> domain;
    for (unsigned i = 0; i < Dim; i++) {
        domain[i] = ippl::Index(nr[i]);
    }

    ippl::e_dim_tag decomp[Dim];
    for (unsigned d = 0; d < Dim; ++d) {
        decomp[d] = ippl::PARALLEL;
    }

    // create mesh and layout objects for this problem domain
    Vector_t<float, Dim> kw = 0.5;
    float alpha             = 0.05;
    Vector_t<double, Dim> rmin(0.0);
    Vector_t<double, Dim> rmax = 2 * pi / kw;

    Vector_t<double, Dim> hr = rmax / nr;
    // Q = -\int\int f dx dv
    double Q = std::reduce(rmax.begin(), rmax.end(), -1., std::multiplies<double>());
    Vector_t<double, Dim> origin = rmin;
    const double dt              = 0.5 * hr[0];

    const bool isAllPeriodic = true;
    Mesh_t<Dim> mesh(domain, hr, origin);
    FieldLayout_t<Dim> FL(domain, decomp, isAllPeriodic);
    PLayout_t<float, Dim> PL(FL, mesh);

    std::string solver = argv[arg++];
    P                  = std::make_unique<bunch_type>(PL, hr, rmin, rmax, decomp, Q, solver);

    P->nr_m = nr;

    P->initializeFields(mesh, FL);

    bunch_type bunchBuffer(PL);

    P->initSolver();
    P->time_m                 = 0.0;
    P->loadbalancethreshold_m = std::atof(argv[arg++]);

    bool isFirstRepartition;

    if ((P->loadbalancethreshold_m != 1.0) && (Ippl::Comm->size() > 1)) {
        msg << "Starting first repartition" << endl;
        IpplTimings::startTimer(domainDecomposition);
        isFirstRepartition             = true;
        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        const int nghost               = P->rho_m.getNghost();
        auto rhoview                   = P->rho_m.getView();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign initial rho based on PDF", ippl::getRangePolicy(rhoview, nghost),
            KOKKOS_LAMBDA(const index_array_type& args) {
                // local to global index conversion
                Vector_t<double, Dim> xvec = (args + lDom.first() - nghost + 0.5) * hr + origin;

                // ippl::apply accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                ippl::apply(rhoview, args) = PDF(xvec, alpha, kw, Dim);
            });

        Kokkos::fence();

        P->initializeORB(FL, mesh);
        P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
        IpplTimings::stopTimer(domainDecomposition);
    }

    msg << "First domain decomposition done" << endl;
    IpplTimings::startTimer(particleCreation);

    typedef ippl::detail::RegionLayout<float, Dim, Mesh_t<Dim>> RegionLayout_t;
    const RegionLayout_t& RLayout                           = PL.getRegionLayout();
    const typename RegionLayout_t::host_mirror_type Regions = RLayout.gethLocalRegions();
    Vector_t<float, Dim> Nr, Dr, minU, maxU;
    int myRank   = Ippl::Comm->rank();
    float factor = 1;
    for (unsigned d = 0; d < Dim; ++d) {
        Nr[d] = CDF(Regions(myRank)[d].max(), alpha, kw[d])
                - CDF(Regions(myRank)[d].min(), alpha, kw[d]);
        Dr[d]   = CDF(rmax[d], alpha, kw[d]) - CDF(rmin[d], alpha, kw[d]);
        minU[d] = CDF(Regions(myRank)[d].min(), alpha, kw[d]);
        maxU[d] = CDF(Regions(myRank)[d].max(), alpha, kw[d]);
        factor *= Nr[d] / Dr[d];
    }

    size_type nloc            = (size_type)(factor * totalP);
    size_type Total_particles = 0;

    MPI_Allreduce(&nloc, &Total_particles, 1, MPI_UNSIGNED_LONG, MPI_SUM, Ippl::getComm());

    int rest = (int)(totalP - Total_particles);

    if (Ippl::Comm->rank() < rest) {
        ++nloc;
    }

    P->create(nloc);
    Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * Ippl::Comm->rank()));
    Kokkos::parallel_for(
        nloc, generate_random<Vector_t<float, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                  P->R.getView(), P->P.getView(), rand_pool64, alpha, kw, minU, maxU));

    Kokkos::fence();
    Ippl::Comm->barrier();
    IpplTimings::stopTimer(particleCreation);

    P->q = P->Q_m / totalP;
    msg << "particles created and initial conditions assigned " << endl;
    isFirstRepartition = false;
    // The update after the particle creation is not needed as the
    // particles are generated locally

    IpplTimings::startTimer(DummySolveTimer);
    P->rho_m = 0.0;
    P->runSolver();
    IpplTimings::stopTimer(DummySolveTimer);

    P->scatterCIC(totalP, 0, hr);

    IpplTimings::startTimer(SolveTimer);
    P->runSolver();
    IpplTimings::stopTimer(SolveTimer);

    P->gatherCIC();

    IpplTimings::startTimer(dumpDataTimer);
    P->dumpLandau();
    P->gatherStatistics(totalP);
    // P->dumpLocalDomains(FL, 0);
    IpplTimings::stopTimer(dumpDataTimer);

    // begin main timestep loop
    msg << "Starting iterations ..." << endl;
    for (unsigned int it = 0; it < nt; it++) {
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
        // Here, we assume a constant charge-to-mass ratio of -1 for
        // all the particles hence eliminating the need to store mass as
        // an attribute
        // kick

        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        // drift
        IpplTimings::startTimer(RTimer);
        P->R = P->R + dt * P->P;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        PL.update(*P, bunchBuffer);
        IpplTimings::stopTimer(updateTimer);

        // Domain Decomposition
        if (P->balance(totalP, it + 1)) {
            msg << "Starting repartition" << endl;
            IpplTimings::startTimer(domainDecomposition);
            P->repartition(FL, mesh, bunchBuffer, isFirstRepartition);
            IpplTimings::stopTimer(domainDecomposition);
            // IpplTimings::startTimer(dumpDataTimer);
            // P->dumpLocalDomains(FL, it+1);
            // IpplTimings::stopTimer(dumpDataTimer);
        }

        // scatter the charge onto the underlying grid
        P->scatterCIC(totalP, it + 1, hr);

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather E field
        P->gatherCIC();

        // kick
        IpplTimings::startTimer(PTimer);
        P->P = P->P - 0.5 * dt * P->E;
        IpplTimings::stopTimer(PTimer);

        P->time_m += dt;
        IpplTimings::startTimer(dumpDataTimer);
        P->dumpLandau();
        P->gatherStatistics(totalP);
        IpplTimings::stopTimer(dumpDataTimer);
        msg << "Finished time step: " << it + 1 << " time: " << P->time_m << endl;
    }

    msg << "LandauDamping: End." << endl;
    IpplTimings::stopTimer(mainTimer);
    IpplTimings::print();
    IpplTimings::print(std::string("timing.dat"));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> time_chrono =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

    return 0;
}
