from .torque_dynamics import TorqueDynamics
from ..configure_variables import Controls
from ... import DynamicsFunctions


class TendonDynamics(TorqueDynamics):
    """
    This is used to create a model actuated through tendon activation.

    x = [q, qdot]
    u = [tendon_pull_forces]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def control_configuration_functions(self):
        return [Controls.TENDONS]

    def get_basic_variables(self, nlp, states, controls, parameters, algebraic_states, numerical_timeseries):
        q = DynamicsFunctions.get(nlp.states["q"], states)
        qdot = DynamicsFunctions.get(nlp.states["qdot"], states)
        tendon_pull_forces = DynamicsFunctions.get(nlp.controls["tendons"], controls)

        # Compute the joint torques based on the pulled tendons
        tau = DynamicsFunctions.compute_tau_from_tendons(nlp, q, qdot, tendon_pull_forces)

        # Additional torques, e.g., from friction or ligaments
        tau += DynamicsFunctions.collect_tau(nlp, q, qdot, parameters)

        external_forces = nlp.get_external_forces(
            "external_forces", states, controls, algebraic_states, numerical_timeseries
        )
        return q, qdot, tau, external_forces

