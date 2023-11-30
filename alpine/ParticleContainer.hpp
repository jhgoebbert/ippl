#ifndef IPPL_PARTICLE_CONTAINER_H
#define IPPL_PARTICLE_CONTAINER_H

#include <memory>
#include "Manager/BaseManager.h"

// Define the ParticlesContainer class
template <typename T, unsigned Dim = 3>
class ParticleContainer : public ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>{
    using Base = ippl::ParticleBase<ippl::ParticleSpatialLayout<T, Dim>>;

    public:
        ippl::ParticleAttrib<double> q;                 // charge
        typename Base::particle_position_type P;  // particle velocity
        typename Base::particle_position_type E;  // electric field at particle position
    private:
        std::shared_ptr<PLayout_t<T, Dim>> pl_m;
    public:
        ParticleContainer(std::shared_ptr<PLayout_t<T, Dim>> pl)
        : Base(*pl.get()) {
        this->initialize(*pl.get());
        registerAttributes();
        setupBCs();
        pl_m = pl;
        }

        ~ParticleContainer(){}

        inline ippl::ParticleAttrib<double>& getQ() { return q; }
        inline void setQ(ippl::ParticleAttrib<double>& q_) { q = q_; }

        inline typename Base::particle_position_type& getP() { return P; }
        inline void setP(typename Base::particle_position_type& P_) { P = P_; }

        inline typename Base::particle_position_type& getE() { return E; }
        inline void setE(typename Base::particle_position_type& E_) { E = E_; }

        inline std::shared_ptr<PLayout_t<T, Dim>> getPL() { return pl_m; }
        inline void setPL(std::shared_ptr<PLayout_t<T, Dim>>& pl) { pl_m = pl; }

	void registerAttributes() {
		// register the particle attributes
		this->addAttribute(q);
		this->addAttribute(P);
		this->addAttribute(E);
	}
	void setupBCs() { setBCAllPeriodic(); }

    private:
       void setBCAllPeriodic() { this->setParticleBC(ippl::BC::PERIODIC); }
};

#endif
