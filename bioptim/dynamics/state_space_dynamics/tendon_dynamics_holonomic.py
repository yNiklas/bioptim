from typing import List, Callable, Any

from casadi import DM, vertcat

from ..configure_variables import States, Controls, AlgebraicStates, ConfigureVariables
from ..dynamics_evaluation import DynamicsEvaluation
from ..dynamics_functions import DynamicsFunctions
from ..ode_solvers import OdeSolver
from .abstract_dynamics import StateDynamicsWithContacts
from bioptim.misc.parameters_types import CXOptional


class HolonomicTendonDynamics(StateDynamicsWithContacts):
    def __init__(self):
        super().__init__()

    @property
    def state_configuration_functions(self) -> List[States | Callable]:
        return [States.Q_U, States.QDOT_U]

    @property
    def control_configuration_functions(self) -> List[Controls | Callable]:
        return [Controls.TENDONS, Controls.NON_TENDON_TAU]

    @property
    def algebraic_configuration_functions(self) -> List[AlgebraicStates | Callable]:
        return []

    @property
    def extra_configuration_functions(self) -> List[Callable]:
        return [
            ConfigureVariables.configure_qv,
            ConfigureVariables.configure_qdotv,
            ConfigureVariables.configure_lagrange_multipliers_function
        ]

    def get_basic_variables(
            self,
            nlp,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries
    ):
        q_u = DynamicsFunctions.get(nlp.states["q_u"], states)
        qdot_u = DynamicsFunctions.get(nlp.states["qdot_u"], states)
        tendons_pull_forces = DynamicsFunctions.get(nlp.controls["tendons"], controls)
        non_tendon_tau = DynamicsFunctions.get(nlp.controls["non_tendon_tau"], controls)
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)

        q = nlp.model.compute_q()(q_u, q_v_init)
        qdot = nlp.model.compute_qdot()(q, qdot_u)

        tau = DynamicsFunctions.compute_tau_from_tendons(nlp, q, qdot, tendons_pull_forces)
        if nlp.model.non_tendon_tau_indices:
            tau[nlp.model.non_tendon_tau_indices] += nlp.get_var("non_tendon_tau", states, controls)
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q_u, qdot_u, q, qdot, tau, external_forces

    def dynamics(
            self,
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp
    ) -> DynamicsEvaluation:
        q_u, qdot_u, q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        q_v_init = DM.zeros(nlp.model.nb_dependent_joints)

        qddot = nlp.model.contact_aware_partitioned_forward_dynamics()(q_u, qdot_u, q_v_init, tau)
        dxdt = vertcat(qdot, qddot)

        defects = None
        if isinstance(nlp.dynamics_type.ode_solver, OdeSolver.COLLOCATION):
            slope_q_u = DynamicsFunctions.get(nlp.states_dot["q_u"], nlp.states_dot.scaled.cx)
            slope_qdot_u = DynamicsFunctions.get(nlp.states_dot["qdot_u"], nlp.states_dot.scaled.cx)
            defects = vertcat(slope_q_u, slope_qdot_u) - dxdt

        return DynamicsEvaluation(dxdt=dxdt, defects=defects)

    def get_rigid_contact_forces(
        self,
        time: CXOptional,
        states: CXOptional,
        controls: CXOptional,
        parameters: CXOptional,
        algebraic_states: CXOptional,
        numerical_timeseries: Any,
        nlp: Any,
    ) -> Any:
        q_u, qdot_u, q, qdot, tau, external_forces = self.get_basic_variables(
            nlp, states, controls, parameters, algebraic_states, numerical_timeseries
        )
        return nlp.model.rigid_contact_forces()(q, qdot, tau, external_forces, nlp.parameters.cx)
