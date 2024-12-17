// SPDX-FileCopyrightText: Copyright (c) Kitware Inc.
// SPDX-License-Identifier: BSD-3-Clause
#ifndef CatalystAdaptor_h
#define CatalystAdaptor_h

#include "Ippl.h"

#include <catalyst.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include "Utility/IpplException.h"

class ParticleDataContainer {
public:
    typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror R_host;
    typename ippl::ParticleAttrib<ippl::Vector<double, 3>>::HostMirror P_host;
    typename ippl::ParticleAttrib<double>::HostMirror q_host;
    typename ippl::ParticleAttrib<std::int64_t>::HostMirror ID_host;
    int getLocalNum;

    ParticleDataContainer() 
    : R_host(), P_host(), q_host(), ID_host(), getLocalNum(0) {}

    template <typename ParticleContainer>
    void copyFrom(const ParticleContainer& particleContainer) {
        R_host = particleContainer->R.getHostMirror();
        P_host = particleContainer->P.getHostMirror();
        q_host = particleContainer->q.getHostMirror();
        ID_host = particleContainer->ID.getHostMirror();

        // deep_copy로 데이터 복사
        Kokkos::deep_copy(R_host, particleContainer->R.getView());
        Kokkos::deep_copy(P_host, particleContainer->P.getView());
        Kokkos::deep_copy(q_host, particleContainer->q.getView());
        Kokkos::deep_copy(ID_host, particleContainer->ID.getView());
        getLocalNum = particleContainer->getLocalNum();
    }
}; // class ParticleDataContainer

namespace CatalystAdaptor {

    using View_vector =
        Kokkos::View<ippl::Vector<double, 3>***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_vector& view) {
        node["electrostatic/association"].set_string("element");
        node["electrostatic/topology"].set_string("mesh");
        node["electrostatic/volume_dependent"].set_string("false");

        auto length = std::size(view);

        // offset is zero as we start without the ghost cells
        // stride is 1 as we have every index of the array
        node["electrostatic/values/x"].set_external(&view.data()[0][0], length, 0, 1);
        node["electrostatic/values/y"].set_external(&view.data()[0][1], length, 0, 1);
        node["electrostatic/values/z"].set_external(&view.data()[0][2], length, 0, 1);
    }

    using View_scalar = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace>;
    inline void setData(conduit_cpp::Node& node, const View_scalar& view) {
        node["density/association"].set_string("element");
        node["density/topology"].set_string("mesh");
        node["density/volume_dependent"].set_string("false");

        node["density/values"].set_external(view.data(), view.size());
    }

    void Initialize(char* argv) {
        conduit_cpp::Node node;
        node["catalyst/scripts/script0"].set_string(argv);

        try {
            node["catalyst_load/implementation"]        = getenv("CATALYST_IMPLEMENTATION_NAME");
            node["catalyst_load/search_paths/paraview"] = getenv("CATALYST_IMPLEMENTATION_PATHS");
        } catch (...) {
            throw IpplException("CatalystAdaptor::Initialize",
                                "no environmental variable for CATALYST_IMPLEMENTATION_NAME or "
                                "CATALYST_IMPLEMENTATION_PATHS found");
        }
        // TODO: catch catalyst error also with IpplException
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }


    void Initialize_Adios(int argc, char* argv[])
    {
        conduit_cpp::Node node;
        for (int cc = 1; cc < argc; ++cc)
        {
            if (strstr(argv[cc], "xml"))
            {
                node["adios/config_filepath"].set_string(argv[cc]);
            }
            else
            {
                node["catalyst/scripts/script" +std::to_string(cc - 1)].set_string(argv[cc]);
            }
        }
        node["catalyst_load/implementation"] = getenv("CATALYST_IMPLEMENTATION_NAME");
        catalyst_status err = catalyst_initialize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok)
        {
            std::cerr << "Failed to initialize Catalyst: " << err << std::endl;
        }
    }


    template <class Field1, class Field2>
    void Execute_Field(int cycle, double time, int rank, Field1& rho, Field2& E, conduit_cpp::Node& node) {
        static_assert(Field1::dim == 3, "CatalystAdaptor only supports 3D");
        static_assert(Field2::dim == 3, "CatalystAdaptor only supports 3D");
        // catalyst blueprint definition
        // https://docs.paraview.org/en/latest/Catalyst/blueprints.html
        //
        // conduit blueprint definition (v.8.3)
        // https://llnl-conduit.readthedocs.io/en/latest/blueprint_mesh.html

        // Set Conduit Node for rho
        auto nGhost = rho.getNghost();

        typename Field1::view_type::host_mirror_type host_view =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rho.getView());

        Kokkos::View<typename Field1::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>
            host_view_layout_left("host_view_layout_left",
                                  rho.getLayout().getLocalNDIndex()[0].length(),
                                  rho.getLayout().getLocalNDIndex()[1].length(),
                                  rho.getLayout().getLocalNDIndex()[2].length());

        for (size_t i = 0; i < rho.getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < rho.getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < rho.getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left(i, j, k) = host_view(i + nGhost, j + nGhost, k + nGhost);
                }
            }
        }
        
        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // add catalyst channel named ippl_field, as fields is reserved
        auto channel = node["catalyst/channels/ippl_density"];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim{"coordsets/coords/dims/i"};
        std::string field_node_origin{"coordsets/coords/origin/x"};
        std::string field_node_spacing{"coordsets/coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < rho.get_mesh().getGridsize().dim; ++iDim) {
            // add dimension
            mesh[field_node_dim].set(rho.getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            mesh[field_node_origin].set(
                rho.get_mesh().getOrigin()[iDim] + rho.getLayout().getLocalNDIndex()[iDim].first()
                      * rho.get_mesh().getMeshSpacing(iDim));

            // add spacing
            mesh[field_node_spacing].set(rho.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim.back();
            ++field_node_origin.back();
            ++field_node_spacing.back();
        }

        // add topology
        mesh["topologies/mesh/type"].set_string("uniform");
        mesh["topologies/mesh/coordset"].set_string("coords");
        std::string field_node_origin_topo{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < rho.get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh[field_node_origin_topo].set(rho.get_mesh().getOrigin()[iDim]
                                             + rho.getLayout().getLocalNDIndex()[iDim].first()
                                                   * rho.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_origin_topo.back();
        }

        // add values and subscribe to data
        auto fields = mesh["fields"];
        setData(fields, host_view_layout_left);

        // rho end
               
               
               
        // Set Conduit Node for E
        auto nGhost2 = E.getNghost();

        typename Field2::view_type::host_mirror_type host_view2 =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), E.getView());

        Kokkos::View<typename Field2::view_type::data_type, Kokkos::LayoutLeft, Kokkos::HostSpace>
            host_view_layout_left2("host_view_layout_left2",
                                  E.getLayout().getLocalNDIndex()[0].length(),
                                  E.getLayout().getLocalNDIndex()[1].length(),
                                  E.getLayout().getLocalNDIndex()[2].length());

        for (size_t i = 0; i < E.getLayout().getLocalNDIndex()[0].length(); ++i) {
            for (size_t j = 0; j < E.getLayout().getLocalNDIndex()[1].length(); ++j) {
                for (size_t k = 0; k < E.getLayout().getLocalNDIndex()[2].length(); ++k) {
                    host_view_layout_left2(i, j, k) = host_view2(i + nGhost2, j + nGhost2, k + nGhost2);
                }
            }
        }
        
        // add catalyst channel named ippl_field, as fields is reserved
        auto channel2 = node["catalyst/channels/ippl_electric"];
        channel2["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh2 = channel2["data"];
        mesh2["coordsets/coords/type"].set_string("uniform");

        // number of points in specific dimension
        std::string field_node_dim2{"coordsets/coords/dims/i"};
        std::string field_node_origin2{"coordsets/coords/origin/x"};
        std::string field_node_spacing2{"coordsets/coords/spacing/dx"};

        for (unsigned int iDim = 0; iDim < E.get_mesh().getGridsize().dim; ++iDim) {
            // add dimension
            mesh2[field_node_dim2].set(E.getLayout().getLocalNDIndex()[iDim].length() + 1);

            // add origin
            mesh2[field_node_origin2].set(
                E.get_mesh().getOrigin()[iDim] + rho.getLayout().getLocalNDIndex()[iDim].first()
                      * rho.get_mesh().getMeshSpacing(iDim));

            // add spacing
            mesh2[field_node_spacing2].set(E.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_dim2.back();
            ++field_node_origin2.back();
            ++field_node_spacing2.back();
        }

        // add topology
        mesh2["topologies/mesh/type"].set_string("uniform");
        mesh2["topologies/mesh/coordset"].set_string("coords");
        std::string field_node_origin_topo2{"topologies/mesh/origin/x"};
        for (unsigned int iDim = 0; iDim < rho.get_mesh().getGridsize().dim; ++iDim) {
            // shift origin
            mesh2[field_node_origin_topo2].set(rho.get_mesh().getOrigin()[iDim]
                                             + rho.getLayout().getLocalNDIndex()[iDim].first()
                                                   * rho.get_mesh().getMeshSpacing(iDim));

            // increment last char in string
            ++field_node_origin_topo2.back();
        }

        // add values and subscribe to data
        auto fields2 = mesh2["fields"];
        setData(fields2, host_view_layout_left2);

    }

    template <class ParticleDataContainer>
    void Execute_Particle(int cycle, double time, int rank, ParticleDataContainer& particleContainer, conduit_cpp::Node& node) {
      assert((particleContainer.ID_host.data() != nullptr) && "ID view should not be nullptr, might be missing the right execution space");


        // add time/cycle information
        auto state = node["catalyst/state"];
        state["cycle"].set(cycle);
        state["time"].set(time);
        state["domain_id"].set(rank);

        // channel for particles
        auto channel = node["catalyst/channels/ippl_particle"];
        channel["type"].set_string("mesh");

        // in data channel now we adhere to conduits mesh blueprint definition
        auto mesh = channel["data"];
        mesh["coordsets/coords/type"].set_string("explicit");

        //mesh["coordsets/coords/values/x"].set_external(&layout_view.data()[0][0], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/y"].set_external(&layout_view.data()[0][1], particleContainer->getLocalNum(), 0, sizeof(double)*3);
        //mesh["coordsets/coords/values/z"].set_external(&layout_view.data()[0][2], particleContainer->getLocalNum(), 0, sizeof(double)*3);

        mesh["coordsets/coords/values/x"].set_external(&particleContainer.R_host.data()[0][0], particleContainer.getLocalNum, 0, sizeof(double)*3);
        mesh["coordsets/coords/values/y"].set_external(&particleContainer.R_host.data()[0][1], particleContainer.getLocalNum, 0, sizeof(double)*3);
        mesh["coordsets/coords/values/z"].set_external(&particleContainer.R_host.data()[0][2], particleContainer.getLocalNum, 0, sizeof(double)*3);
        
        mesh["topologies/mesh/type"].set_string("unstructured");
        mesh["topologies/mesh/coordset"].set_string("coords");
        mesh["topologies/mesh/elements/shape"].set_string("point");
        mesh["topologies/mesh/elements/connectivity"].set_external(particleContainer.ID_host.data(),particleContainer.getLocalNum);
        
        //auto charge_view = particleContainer->getQ().getView();

        // add values for scalar charge field
        auto fields = mesh["fields"];
        fields["charge/association"].set_string("vertex");
        fields["charge/topology"].set_string("mesh");
        fields["charge/volume_dependent"].set_string("false");

        //fields["charge/values"].set_external(particleContainer->q.getView().data(), particleContainer.getLocalNum);
        fields["charge/values"].set_external(particleContainer.q_host.data(), particleContainer.getLocalNum);

        // add values for vector velocity field
        //auto velocity_view = particleContainer->P.getView();
        fields["velocity/association"].set_string("vertex");
        fields["velocity/topology"].set_string("mesh");
        fields["velocity/volume_dependent"].set_string("false");

        fields["velocity/values/x"].set_external(&particleContainer.P_host.data()[0][0], particleContainer.getLocalNum,0 ,sizeof(double)*3);
        fields["velocity/values/y"].set_external(&particleContainer.P_host.data()[0][1], particleContainer.getLocalNum,0 ,sizeof(double)*3);
        fields["velocity/values/z"].set_external(&particleContainer.P_host.data()[0][2], particleContainer.getLocalNum,0 ,sizeof(double)*3);

        fields["position/association"].set_string("vertex");
        fields["position/topology"].set_string("mesh");
        fields["position/volume_dependent"].set_string("false");

        fields["position/values/x"].set_external(&particleContainer.R_host.data()[0][0], particleContainer.getLocalNum, 0, sizeof(double)*3);
        fields["position/values/y"].set_external(&particleContainer.R_host.data()[0][1], particleContainer.getLocalNum, 0, sizeof(double)*3);
        fields["position/values/z"].set_external(&particleContainer.R_host.data()[0][2], particleContainer.getLocalNum, 0, sizeof(double)*3);

        // this node we can return as the pointer to velocity and charge is globally valid
    }

        inline void callCatalystExecute(const conduit_cpp::Node& node) {

        // TODO: we should add here this IPPL-INFO stuff
        //if ( static auto called {false}; !std::exchange(called, true) ) {
        //    catalyst_conduit_node_print(conduit_cpp::c_node(&node));
        //}

        catalyst_status err = catalyst_execute(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to execute Catalyst: " << err << std::endl;
        }
    }

    template <class Field1, class Field2, class ParticleContainer>
    void Execute(int cycle, double time, int rank, ParticleContainer& particleContainer, Field1& rho, Field2& E) {

        ParticleDataContainer pdata;
        pdata.copyFrom(particleContainer);
        // create conduit node
        conduit_cpp::Node node;
        CatalystAdaptor::Execute_Particle(cycle, time, rank, pdata, node);
        CatalystAdaptor::Execute_Field(cycle, time, rank, rho, E, node);
        callCatalystExecute(node);

    }

    void Finalize() {
        conduit_cpp::Node node;
        catalyst_status err = catalyst_finalize(conduit_cpp::c_node(&node));
        if (err != catalyst_status_ok) {
            std::cerr << "Failed to finalize Catalyst: " << err << std::endl;
        }
    }
}  // namespace CatalystAdaptor


#endif
