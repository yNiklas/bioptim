from .torque_dynamics import TorqueDynamics
from ..configure_variables import Controls

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
