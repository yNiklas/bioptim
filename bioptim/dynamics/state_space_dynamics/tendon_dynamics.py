from .torque_dynamics import TorqueDynamics
from ..configure_variables import Controls
from ... import DynamicsFunctions


class TendonDynamics(TorqueDynamics):
    """
    This is used to create a model actuated through tendon activation.

    x = [q, qdot]
    u = [tendons] where tendons are the tendon pull forces in N
    or u=[tendons, non_tendon_tau] if torque_driven_dof_indices is not empty
    """

    def __init__(self, exclude_free_float_actuation=False,
                 has_non_tendon_tau=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.has_non_tendon_tau = has_non_tendon_tau
        self.exclude_free_float_actuation = exclude_free_float_actuation

    @property
    def control_configuration_functions(self):
        if self.has_non_tendon_tau:
            return [Controls.TENDONS, Controls.NON_TENDON_TAU]
        else:
            return [Controls.TENDONS]

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tendon_pull_forces = DynamicsFunctions.get(nlp.controls["tendons"], controls)

        # Compute the joint torques based on the pulled tendons
        tau = DynamicsFunctions.compute_tau_from_tendons(nlp, q, qdot, tendon_pull_forces)
        if self.exclude_free_float_actuation:
            tau[:nlp.model.nb_root] = 0
        if self.has_non_tendon_tau:
            indices = nlp.model.non_tendon_tau_indices
            tau[indices] += nlp.get_var("non_tendon_tau", states, controls)

        # Additional torques, e.g., from friction or ligaments
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, tau, external_forces

    def dynamics(
        self,
        time,
        states,
        controls,
        parameters,
        algebraic_states,
        numerical_timeseries,
        nlp,
    ):
        # Call the super function since the implementation is identical.
        # However, the super function call get_basic_variables to get the tendons tau
        return super().dynamics(time, states, controls, parameters, algebraic_states, numerical_timeseries, nlp)
