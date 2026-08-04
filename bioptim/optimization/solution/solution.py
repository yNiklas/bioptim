from copy import deepcopy
from typing import Any

from casadi import vertcat, DM, Function
from matplotlib import pyplot as plt
from matplotlib.widgets import CheckButtons, Slider
import numpy as np
from scipy import interpolate as sci_interp

from .solution_data import SolutionData, SolutionMerge, TimeAlignment, TimeResolution
from ..optimization_vector import OptimizationVectorHelper
from ...dynamics.ode_solvers import OdeSolver
from ...interfaces.solve_ivp_interface import solve_ivp_interface
from ...limits.path_conditions import InitialGuess, InitialGuessList
from ...limits.penalty_helpers import PenaltyHelpers
from ...limits.penalty_option import PenaltyOption
from ...misc.enums import (
    ControlType,
    CostType,
    Shooting,
    InterpolationType,
    SolverType,
    SolutionIntegrator,
    Node,
)
from ...misc.parameters_types import (
    Bool,
    Int,
    IntOptional,
    Float,
    FloatOptional,
    Str,
    StrOptional,
    AnyList,
    AnyListOptional,
    AnyDict,
    AnyTuple,
    FloatTuple,
    AnyIterable,
    NpArray,
    NpArrayOptional,
)
from ...models.protocols.stochastic_biomodel import StochasticBioModel


class Solution:
    """
    Data manipulation, graphing and storage

    Attributes
    ----------
    vector: np.ndarray
        The data in the vector format
    _cost: float
        The value of the cost function
    constraints: list
        The values of the constraint
    lam_g: list
        The Lagrange multiplier of the constraints
    lam_p: list
        The Lagrange multiplier of the parameters
    lam_x: list
        The Lagrange multiplier of the states and controls
    inf_pr: list
        The unscaled constraint violation at each iteration
    inf_du: list
        The scaled dual infeasibility at each iteration
    solver_time_to_optimize: float
        The total time to solve the program
    iterations: int
        The number of iterations that were required to solve the program
    status: int
        Optimization success status (Ipopt: 0=Succeeded, 1=Failed)
    _stepwise_times: list
        The time corresponding to _stepwise_states
    _decision_states: SolutionData
        A SolutionData based solely on the solution
    _stepwise_states: SolutionData
        A SolutionData based on the integrated solution directly from the bioptim integrator
    _stepwise_controls: SolutionData
        The data structure that holds the controls
    _parameters: SolutionData
        The data structure that holds the parameters
    _decision_algebraic_states: SolutionData
        The data structure that holds the algebraic_states variables
    phases_dt: list
        The time step for each phases

    Methods
    -------
    copy(self, skip_data: bool = False) -> Any
        Create a deepcopy of the Solution
    @property
    controls(self) -> list | dict
        Returns the controls scaled and unscaled in list if more than one phases, otherwise it returns the only dict
    integrate(self, shooting_type: Shooting = Shooting.MULTIPLE, keep_intermediate_points: bool = True,
              merge_phases: bool = False, continuous: bool = True) -> Solution
        Integrate the states unscaled
    interpolate(self, n_frames: int | list | tuple) -> Solution
        Interpolate the states unscaled
    merge_phases(self) -> Solution
        Get a data structure where all the phases are merged into one
    _merge_phases(self, skip_states: bool = False, skip_controls: bool = False) -> tuple
        Actually performing the phase merging
    _complete_control(self)
        Controls don't necessarily have dimensions that matches the states. This method aligns them
        graphs(self, automatically_organize: bool, show_bounds: bool,
            show_now: bool, shooting_type: Shooting, integrator: SolutionIntegrator,
            save_name: str, show_gcom_plot: bool, show_interactive_stability_plot: bool)
        Show the graphs of the simulation
    animate(self, n_frames: int = 0, show_now: bool = True, **kwargs: Any) -> None | list
        Animate the simulation
    print(self, cost_type: CostType = CostType.ALL)
        Print the objective functions and/or constraints to the console
    """

    def __init__(
        self,
        ocp: "OptimalControlProgram",
        vector: NpArrayOptional | DM = None,
        cost: NpArrayOptional | DM = None,
        constraints: NpArrayOptional | DM = None,
        lam_g: NpArrayOptional | DM = None,
        lam_p: NpArrayOptional | DM = None,
        lam_x: NpArrayOptional | DM = None,
        inf_pr: NpArrayOptional | DM = None,
        inf_du: NpArrayOptional | DM = None,
        solver_time_to_optimize: FloatOptional = None,
        real_time_to_optimize: FloatOptional = None,
        iterations: IntOptional = None,
        status: IntOptional = None,
    ):
        """
        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the ocp to strip
        vector: np.ndarray | DM
            The solution vector, containing all the states, controls, parameters and algebraic_states variables
        cost: np.ndarray | DM
            The cost value of the objective function
        constraints: np.ndarray | DM
            The constraints value
        lam_g: np.ndarray | DM
            The lagrange multipliers for the constraints
        lam_p: np.ndarray | DM
            The lagrange multipliers for the parameters
        lam_x: np.ndarray | DM
            The lagrange multipliers for the states
        lam_g: np.ndarray | DM
            The lagrange multipliers for the constraints
        inf_pr: np.ndarray | DM
            The primal infeasibility
        inf_du: np.ndarray | DM
            The dual infeasibility
        solver_time_to_optimize: float
            The time to optimize
        real_time_to_optimize: float
            The real time to optimize
        iterations: int
            The number of iterations
        status: int
            The status of the solution
        """

        self.ocp = ocp

        # Penalties
        self._cost, self._detailed_cost, self.constraints = cost, None, constraints

        # Solver options
        self.status, self.iterations = status, iterations
        self.lam_g, self.lam_p, self.lam_x, self.inf_pr, self.inf_du = lam_g, lam_p, lam_x, inf_pr, inf_du
        self.solver_time_to_optimize, self.real_time_to_optimize = solver_time_to_optimize, real_time_to_optimize

        # Extract the data now for further use
        self._decision_states = None
        self._stepwise_states = None
        self._stepwise_controls = None
        self._parameters = None
        self._decision_algebraic_states = None

        self.vector = vector
        if self.vector is not None:
            self.phases_dt = OptimizationVectorHelper.extract_phase_dt(ocp, vector)
            self._stepwise_times = OptimizationVectorHelper.extract_step_times(ocp, vector)

            x, u, p, a = self.ocp.vector_layout.unstack_to_dicts(vector)
            u = OptimizationVectorHelper.control_duplication(u, ocp.nlp)

            self._decision_states = SolutionData.from_scaled(ocp, x, "x")
            self._stepwise_controls = SolutionData.from_scaled(ocp, u, "u")
            self._parameters = SolutionData.from_scaled(ocp, p, "p")
            self._decision_algebraic_states = SolutionData.from_scaled(ocp, a, "a")

    @classmethod
    def from_dict(cls, ocp: "OptimalControlProgram", sol: AnyDict):
        """
        Initialize all the attributes from an Ipopt-like dictionary data structure

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: dict
            The solution in a Ipopt-like dictionary
        """

        if not isinstance(sol, dict):
            raise ValueError("The _sol entry should be a dictionary")

        is_ipopt_like = sol["solver"] in (SolverType.IPOPT.value, SolverType.FATROP.value)

        return cls(
            ocp=ocp,
            vector=sol["x"],
            cost=sol["f"] if is_ipopt_like else None,
            constraints=sol["g"] if is_ipopt_like else None,
            lam_g=sol["lam_g"] if is_ipopt_like else None,
            lam_p=sol["lam_p"] if is_ipopt_like else None,
            lam_x=sol["lam_x"] if is_ipopt_like else None,
            inf_pr=sol["inf_pr"] if is_ipopt_like else None,
            inf_du=sol["inf_du"] if is_ipopt_like else None,
            solver_time_to_optimize=sol["solver_time_to_optimize"],
            real_time_to_optimize=sol["real_time_to_optimize"],
            iterations=sol["iter"],
            status=sol["status"],
        )

    @classmethod
    def from_initial_guess(cls, ocp: "OptimalControlProgram", sol: AnyList):
        """
        Initialize all the attributes from a list of initial guesses (states, controls)

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: list
            The list of initial guesses
        """

        if not (isinstance(sol, (list, tuple)) and len(sol) == 5):
            raise ValueError("_sol should be a list of tuple and the length should be 5")

        n_param = len(ocp.parameters)
        all_ns = [nlp.ns for nlp in ocp.nlp]

        # Sanity checks
        for i in range(len(sol)):  # Convert to list if necessary and copy for as many phases there are
            if isinstance(sol[i], InitialGuess):
                tp = InitialGuessList()
                for _ in range(len(all_ns)):
                    tp.add(deepcopy(sol[i].init), interpolation=sol[i].init.type)
                sol[i] = tp
        if sum([isinstance(s, InitialGuessList) for s in sol]) != 4:
            raise ValueError(
                "solution must be a solution dict, "
                "an InitialGuess[List] of len 4 (states, controls, parameters, algebraic_states), "
                "or a None"
            )

        if len(sol[0]) != len(all_ns):
            raise ValueError("The time step dt array len must match the number of phases")

        is_right_size = [
            len(s) != len(all_ns) if p != 3 and len(sol[p + 1].keys()) != 0 else False for p, s in enumerate(sol[:1])
        ]

        if sum(is_right_size) != 0:
            raise ValueError("The InitialGuessList len must match the number of phases")

        if n_param != 0:
            if len(sol) != 3 and len(sol[3]) != 1 and sol[3][0].shape != (n_param, 1):
                raise ValueError(
                    "The 3rd element is the InitialGuess of the parameter and "
                    "should be a unique vector of size equal to n_param"
                )

        dt, sol_states, sol_controls, sol_params, sol_algebraic_states = sol

        vector = np.ndarray((0, 1))

        # For time
        if len(dt.shape) == 1:
            dt = dt[:, np.newaxis]
        vector = np.concatenate((vector, dt))

        # For states
        for p, ss in enumerate(sol_states):
            nb_intermediate_frames = 1
            if isinstance(ocp.nlp[p].dynamics_type.ode_solver, OdeSolver.COLLOCATION):
                nb_intermediate_frames = ocp.nlp[p].dynamics_type.ode_solver.polynomial_degree + 1
            for key in ss.keys():
                ns = (
                    ocp.nlp[p].ns * nb_intermediate_frames
                    if ss[key].init.type == InterpolationType.ALL_POINTS
                    else ocp.nlp[p].ns + 1 if ss[key].init.type != InterpolationType.EACH_FRAME else ocp.nlp[p].ns
                )
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].states[key]), ns, "states")

            for i in range(all_ns[p] * nb_intermediate_frames + 1):
                for key in ss.keys():
                    vector = np.concatenate(
                        (vector, ss[key].init.evaluate_at(i, nb_intermediate_frames)[:, np.newaxis])
                    )

        # For controls
        for p, ss in enumerate(sol_controls):
            control_type = ocp.nlp[p].control_type
            off = 1 if control_type.has_a_final_node else 0

            for key in ss.keys():
                ss[key].init.check_and_adjust_dimensions(len(ocp.nlp[p].controls[key]), all_ns[p] - 1 + off, "controls")

            for i in range(all_ns[p] + off):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

        # For parameters
        if n_param:
            for p, ss in enumerate(sol_params):
                for key in ss.keys():
                    vector = np.concatenate((vector, np.repeat(ss[key].init, 1)[:, np.newaxis]))

        # For algebraic_states variables
        for p, ss in enumerate(sol_algebraic_states):
            for key in ss.keys():
                ss[key].init.check_and_adjust_dimensions(
                    len(ocp.nlp[p].algebraic_states[key]), all_ns[p], "algebraic_states"
                )

            for i in range(all_ns[p] + 1):
                for key in ss.keys():
                    vector = np.concatenate((vector, ss[key].init.evaluate_at(i)[:, np.newaxis]))

        return cls(ocp=ocp, vector=vector)

    @classmethod
    def from_vector(cls, ocp: "OptimalControlProgram", sol: NpArray | DM):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        sol: np.ndarray | DM
            The solution in vector format
        """

        if not isinstance(sol, (np.ndarray, DM)):
            raise ValueError("The _sol entry should be a np.ndarray or a DM.")

        return cls(ocp=ocp, vector=sol)

    @classmethod
    def from_ocp(cls, ocp: "OptimalControlProgram"):
        """
        Initialize all the attributes from a vector of solution

        Parameters
        ----------
        ocp: OptimalControlProgram
            A reference to the OptimalControlProgram
        """

        return cls(ocp=ocp)

    def t_span(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: Bool = True,
    ) -> AnyList | NpArray:
        """
        Returns the time span at each node of each phases
        """
        return self._process_time_vector(
            time_resolution=TimeResolution.NODE_SPAN,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
        )

    def decision_time(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: Bool = True,
    ) -> AnyList | NpArray:
        """
        Returns the time vector at each node that matches decision_states or decision_controls

        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed. It is often useful to merge NODES, but
            is completely useless to merge KEYS
        time_alignment: TimeAlignment
            The type of alignment to perform. If TimeAlignment.STATES, then the time vector is aligned with the states
            (i.e. all the subnodes and the last node time are present). If TimeAlignment.CONTROLS, then the time vector
            is aligned with the controls (i.e. only starting of the node without the last node if CONTROL constant).
        continuous: bool
            If the time should be continuous throughout the whole ocp. If False, then the time is reset at the
            beginning of each phase.
        """

        return self._process_time_vector(
            time_resolution=TimeResolution.DECISION,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
        )

    def stepwise_time(
        self,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        time_alignment: TimeAlignment = TimeAlignment.STATES,
        continuous: Bool = True,
        duplicated_times: Bool = True,
    ) -> AnyList | NpArray:
        """
        Returns the time vector at each node that matches stepwise_states or stepwise_controls

        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed. It is often useful to merge NODES, but
            is completely useless to merge KEYS
        time_alignment: TimeAlignment
            The type of alignment to perform. If TimeAlignment.STATES, then the time vector is aligned with the states
            (i.e. all the subnodes and the last node time are present). If TimeAlignment.CONTROLS, then the time vector
            is aligned with the controls (i.e. only starting of the node without the last node if CONTROL constant).
        continuous: bool
            If the time should be continuous throughout the whole ocp. If False, then the time is reset at the
            beginning of each phase.
        duplicated_times: bool
            If the times should be duplicated for each nodes.
            If False, then the returned time vector will not have any duplicated times

        Returns
        -------
        The time vector at each node that matches stepwise_states or stepwise_controls
        """

        return self._process_time_vector(
            time_resolution=TimeResolution.STEPWISE,
            to_merge=to_merge,
            time_alignment=time_alignment,
            continuous=continuous,
            duplicated_times=duplicated_times,
        )

    def _process_time_vector(
        self,
        time_resolution: TimeResolution,
        to_merge: SolutionMerge | list[SolutionMerge],
        time_alignment: TimeAlignment,
        continuous: Bool,
        duplicated_times: Bool = True,
    ) -> AnyList | NpArray:
        if to_merge is None or isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]

        # Make sure to not return internal structure
        times_tp = deepcopy(self._stepwise_times)

        # Select the appropriate time matrix
        phases_tf = []
        times = []
        for nlp in self.ocp.nlp:
            phases_tf.append(times_tp[nlp.phase_idx][-1])

            if time_resolution == TimeResolution.NODE_SPAN:
                if time_alignment == TimeAlignment.STATES:
                    times.append([t if t.shape == (1, 1) else t[[0, -1]] for t in times_tp[nlp.phase_idx]])
                elif time_alignment == TimeAlignment.CONTROLS:
                    times.append([t[[0, -1]] for t in times_tp[nlp.phase_idx][:-1]])
            else:
                if time_alignment == TimeAlignment.STATES:
                    if nlp.dynamics_type.ode_solver.is_direct_collocation:
                        if nlp.dynamics_type.ode_solver.duplicate_starting_point:
                            times.append(
                                [t if t.shape == (1, 1) else vertcat(t[0], t[:-1]) for t in times_tp[nlp.phase_idx]]
                            )
                        else:
                            times.append([t if t.shape == (1, 1) else t[:-1] for t in times_tp[nlp.phase_idx]])

                    else:
                        if time_resolution == TimeResolution.STEPWISE:
                            times.append(times_tp[nlp.phase_idx])

                        elif time_resolution == TimeResolution.DECISION:
                            times.append([t[0] for t in times_tp[nlp.phase_idx]])

                        else:
                            raise ValueError("Unrecognized time_resolution")

                elif time_alignment == TimeAlignment.CONTROLS:
                    if nlp.control_type == ControlType.LINEAR_CONTINUOUS:
                        times.append([(t if t.shape == (1, 1) else t[[0, -1]]) for t in times_tp[nlp.phase_idx]])
                        if len(times) < len(self.ocp.nlp):
                            # The point is duplicated for internal phases, but not the last one
                            times[-1][-1] = times[-1][-1][[0, 0]].T
                    elif nlp.control_type == ControlType.CONSTANT_WITH_LAST_NODE:
                        times.append([t[0] for t in times_tp[nlp.phase_idx]])
                    elif nlp.control_type == ControlType.CONSTANT:
                        times.append([t[0] for t in times_tp[nlp.phase_idx]][:-1])
                    else:
                        raise ValueError(f"Unrecognized control type {nlp.control_type}")

                else:
                    raise ValueError("time_alignment should be either TimeAlignment.STATES or TimeAlignment.CONTROLS")

        if not duplicated_times:
            for i in range(len(times)):
                for j in range(len(times[i])):
                    # Last node of last phase is always kept
                    keep_condition = times[i][j].shape[0] == 1 and i == len(times) - 1
                    times[i][j] = times[i][j][:] if keep_condition else times[i][j][:-1]
                    if j == len(times[i]) - 1 and i != len(times) - 1:
                        del times[i][j]

        if continuous:
            for phase_idx, phase_time in enumerate(times):
                if phase_idx == 0:
                    continue
                previous_tf = sum(phases_tf[:phase_idx])
                times[phase_idx] = [t + previous_tf for t in phase_time]

        if SolutionMerge.NODES in to_merge or SolutionMerge.ALL in to_merge:
            for phase_idx in range(len(times)):
                np.concatenate((np.concatenate(times[phase_idx][:-1]), times[phase_idx][-1]))
                times[phase_idx] = np.concatenate((np.concatenate(times[phase_idx][:-1]), times[phase_idx][-1]))

        if (
            SolutionMerge.PHASES in to_merge and SolutionMerge.NODES not in to_merge
        ) and SolutionMerge.ALL not in to_merge:
            raise ValueError("Cannot merge phases without nodes")

        if SolutionMerge.PHASES in to_merge or SolutionMerge.ALL in to_merge:
            # NODES is necessarily in to_merge if PHASES is in to_merge
            times = np.concatenate(times)

        return times if len(times) > 1 else times[0]

    def decision_states(
        self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None
    ) -> AnyList | AnyDict:
        """
        Returns the decision states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The decision variables
        """

        data = self._decision_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def stepwise_states(self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the stepwise integrated states

        Parameters
        ----------
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The stepwise integrated states
        """

        if self._stepwise_states is None:
            self._integrate_stepwise()

        data = self._stepwise_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def decision_controls(self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the decision controls

        Parameters
        ----------
        scaled : bool
            If the decision controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge : SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.
        """
        return self.stepwise_controls(scaled=scaled, to_merge=to_merge)

    def stepwise_controls(self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None):
        """
        Returns the controls. Note the final control is always present but set to np.nan if it is not defined

        Parameters
        ----------
        scaled: bool
            If the controls should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The controls
        """

        data = self._stepwise_controls.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    @property
    def parameters(self) -> Any:
        """
        Returns the parameters
        """

        return self.decision_parameters(scaled=False)

    def decision_parameters(self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None) -> Any:
        """
        Returns the decision parameters

        Parameters
        ----------
        scaled: bool
            If the parameters should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as
            the model needs temps). If you don't know what it means, you probably want the unscaled version.

        Returns
        -------
        The decision parameters
        """
        if to_merge is None:
            to_merge = []

        if isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]

        if SolutionMerge.PHASES in to_merge:
            raise ValueError("Cannot merge phases for parameters as it is not bound to phases")
        if SolutionMerge.NODES in to_merge:
            raise ValueError("Cannot merge nodes for parameters as it is not bound to nodes")

        out = self._parameters.to_dict(scaled=scaled, to_merge=to_merge)

        # Remove the residual phases and nodes
        if to_merge:
            out = out[0][0][:, 0]
        else:
            out = out[0]
            out = {key: out[key][0][:, 0] for key in out.keys()}

        return out

    def decision_algebraic_states(
        self, scaled: Bool = False, to_merge: SolutionMerge | list[SolutionMerge] = None
    ) -> AnyList | AnyDict:
        """
        Returns the decision algebraic_states

        Parameters
        ----------
        scaled: bool
            If the decision states should be scaled or not (note that scaled is as Ipopt received them, while unscaled
            is as the model needs temps). If you don't know what it means, you probably want the unscaled version.
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.

        Returns
        -------
        The decision variables
        """

        data = self._decision_algebraic_states.to_dict(to_merge=to_merge, scaled=scaled)
        if not isinstance(data, list):
            return data
        return data if len(data) > 1 else data[0]

    def copy(self, skip_data: Bool = False) -> "Solution":
        """
        Create a deepcopy of the Solution

        Parameters
        ----------
        skip_data: bool
            If data should be ignored in the copy

        Returns
        -------
        Return a Solution data structure
        """

        new = Solution.from_ocp(self.ocp)

        new.vector = deepcopy(self.vector)
        new._cost = deepcopy(self._cost)
        new.constraints = deepcopy(self.constraints)

        new.lam_g = deepcopy(self.lam_g)
        new.lam_p = deepcopy(self.lam_p)
        new.lam_x = deepcopy(self.lam_x)
        new.inf_pr = deepcopy(self.inf_pr)
        new.inf_du = deepcopy(self.inf_du)
        new.solver_time_to_optimize = deepcopy(self.solver_time_to_optimize)
        new.real_time_to_optimize = deepcopy(self.real_time_to_optimize)
        new.iterations = deepcopy(self.iterations)

        new.phases_dt = deepcopy(self.phases_dt)
        new._stepwise_times = deepcopy(self._stepwise_times)

        if not skip_data:
            new._decision_states = deepcopy(self._decision_states)
            new._stepwise_states = deepcopy(self._stepwise_states)

            new._stepwise_controls = deepcopy(self._stepwise_controls)

            new._decision_algebraic_states = deepcopy(self._decision_algebraic_states)
            new._parameters = deepcopy(self._parameters)
        return new

    def _prepare_integrate(self, integrator: SolutionIntegrator) -> AnyTuple:
        """
        Prepare the variables for the states integration and checks if the integrator is compatible with the ocp.

        Parameters
        ----------
        integrator: SolutionIntegrator
            The integrator to use for the integration
        """

        has_direct_collocation = sum([nlp.dynamics_type.ode_solver.is_direct_collocation for nlp in self.ocp.nlp]) > 0
        if has_direct_collocation and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.COLLOCATION, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE"
            )

        has_trapezoidal = (
            sum([isinstance(nlp.dynamics_type.ode_solver, OdeSolver.TRAPEZOIDAL) for nlp in self.ocp.nlp]) > 0
        )
        if has_trapezoidal and integrator == SolutionIntegrator.OCP:
            raise ValueError(
                "When the ode_solver of the Optimal Control Problem is OdeSolver.TRAPEZOIDAL, "
                "we cannot use the SolutionIntegrator.OCP.\n"
                "We must use one of the SolutionIntegrator provided by scipy with any Shooting Enum such as"
                " Shooting.SINGLE, Shooting.MULTIPLE, or Shooting.SINGLE_DISCONTINUOUS_PHASE",
            )

        for i_phase, nlp in enumerate(self.ocp.nlp):
            if nlp.dynamics_func is None:
                raise RuntimeError(
                    "The explicit derivative of the states must be provided to be able to reintegrate the dynamics."
                    f"Please provide a dxdt in your DynamicsEvaluation of phase {i_phase}."
                )

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        t_spans = self.t_span(time_alignment=TimeAlignment.CONTROLS)
        if len(self.ocp.nlp) == 1:
            t_spans = [t_spans]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        return t_spans, x, u, params, a

    def integrate(
        self,
        shooting_type: Shooting = Shooting.SINGLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        duplicated_times: Bool = True,
        return_time: Bool = False,
    ) -> Any:
        """
        Create a deepcopy of the Solution

        Parameters
        ----------
        shooting_type: Shooting
            The integration shooting type to use
        integrator: SolutionIntegrator
            The type of integrator to use
        to_merge: SolutionMerge | list[SolutionMerge]
            The type of merge to perform. If None, then no merge is performed.
        duplicated_times: bool
            If the times should be duplicated for each node.
            If False, then the returned time vector will not have any duplicated times.
            Default is True.
        return_time: bool
            If the time vector should be returned, default is False.

        Returns
        -------
        Return the integrated states
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        t_spans, x, u, params, a = self._prepare_integrate(integrator=integrator)

        out: list = [None] * len(self.ocp.nlp)
        integrated_sol = None
        for p, nlp in enumerate(self.ocp.nlp):
            first_x = self._states_for_phase_integration(shooting_type, p, integrated_sol, x, u, params, a)
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            integrated_sol = solve_ivp_interface(
                list_of_dynamics=[nlp.dynamics_func] * nlp.ns,
                shooting_type=shooting_type,
                nlp=nlp,
                t=t_spans[p],
                x=first_x,
                u=u[p],
                a=a[p],
                d=d,
                p=params,
                method=integrator,
            )

            out[p] = {}
            for key in nlp.states.keys():
                out[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    if duplicated_times:
                        out[p][key][ns] = sol_ns[nlp.states[key].index, :]
                    else:
                        # Last node of last phase is always kept
                        duplicated_times_condition = p == len(self.ocp.nlp) - 1 and ns == nlp.ns
                        out[p][key][ns] = (
                            sol_ns[nlp.states[key].index, :]
                            if duplicated_times_condition
                            else sol_ns[nlp.states[key].index, :-1]
                        )

        if to_merge:
            out = SolutionData.from_unscaled(self.ocp, out, "x").to_dict(to_merge=to_merge, scaled=False)

        if return_time:
            time_vector = self._return_time_vector(to_merge=to_merge, duplicated_times=duplicated_times)
            return out if len(out) > 1 else out[0], time_vector if len(time_vector) > 1 else time_vector[0]
        else:
            return out if len(out) > 1 else out[0]

    def noisy_integrate(
        self,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        to_merge: SolutionMerge | list[SolutionMerge] = None,
        size: Int = 100,
    ) -> AnyDict | AnyList:
        """
        Integrated the states with different noise values sampled from the covariance matrix.
        """
        from ...optimization.stochastic_optimal_control_program import StochasticOptimalControlProgram
        from ...interfaces.interface_utils import get_numerical_timeseries

        if not isinstance(self.ocp, StochasticOptimalControlProgram):
            raise ValueError("This method is only available for StochasticOptimalControlProgram.")

        t_spans, x, u, params, a = self._prepare_integrate(integrator=integrator)

        cov_index = self.ocp.nlp[0].controls["cov"].index
        n_sub_nodes = x[0][0].shape[1]
        motor_noise_index = self.ocp.nlp[0].parameters["motor_noise"].index
        sensory_noise_index = (
            self.ocp.nlp[0].parameters["sensory_noise"].index
            if len(list(self.ocp.nlp[0].parameters["sensory_noise"].index)) > 0
            else None
        )

        # initialize the out dictionary
        out = [None] * len(self.ocp.nlp)
        for p, nlp in enumerate(self.ocp.nlp):
            out[p] = {}
            for key in self.ocp.nlp[0].states.keys():
                out[p][key] = [None] * nlp.n_states_nodes
                for i_node in range(nlp.ns):
                    out[p][key][i_node] = np.zeros((len(nlp.states[key].index), n_sub_nodes, size))
                out[p][key][nlp.ns] = np.zeros((len(nlp.states[key].index), 1, size))

        cov_matrix = StochasticBioModel.reshape_to_matrix(u[0][0][cov_index, 0], self.ocp.nlp[0].model.matrix_shape_cov)
        first_x = np.random.multivariate_normal(x[0][0][:, 0], cov_matrix, size=size).T
        for p, nlp in enumerate(self.ocp.nlp):
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            motor_noise = np.zeros((len(params[motor_noise_index]), nlp.ns, size))
            for i in range(len(params[motor_noise_index])):
                motor_noise[i, :] = np.random.normal(0, params[motor_noise_index[i]], size=(nlp.ns, size))
            sensory_noise = (
                np.zeros((len(sensory_noise_index), nlp.ns, size)) if sensory_noise_index is not None else None
            )
            if sensory_noise_index is not None:
                for i in range(len(params[sensory_noise_index])):
                    sensory_noise[i, :] = np.random.normal(0, params[sensory_noise_index[i]], size=(nlp.ns, size))

            without_noise_idx = [
                i for i in range(len(params)) if i not in motor_noise_index and i not in sensory_noise_index
            ]
            parameters_cx = nlp.parameters.cx[without_noise_idx]
            parameters = params[without_noise_idx]
            for i_random in range(size):
                params_this_time = []
                list_of_dynamics = []
                for node in range(nlp.ns):
                    params_this_time += [nlp.parameters.cx]
                    params_this_time[node][motor_noise_index, :] = motor_noise[:, node, i_random]
                    if sensory_noise_index is not None:
                        params_this_time[node][sensory_noise_index, :] = sensory_noise[:, node, i_random]

                    if len(nlp.extra_dynamics_func) > 1 or len(nlp.extra_dynamics_defects_func) > 1:
                        raise NotImplementedError("Noisy integration is not available for multiple extra dynamics.")
                    cas_func = Function(
                        "noised_extra_dynamics",
                        [
                            nlp.time_cx,
                            nlp.states.cx,
                            nlp.controls.cx,
                            parameters_cx,
                            nlp.algebraic_states.cx,
                            nlp.numerical_timeseries.cx,
                        ],
                        [
                            nlp.extra_dynamics_func[0](
                                nlp.time_cx,
                                nlp.states.cx,
                                nlp.controls.cx,
                                params_this_time[node],
                                nlp.algebraic_states.cx,
                                nlp.numerical_timeseries.cx,
                            )
                        ],
                    )
                    list_of_dynamics += [cas_func]

                integrated_sol = solve_ivp_interface(
                    list_of_dynamics=list_of_dynamics,
                    shooting_type=Shooting.SINGLE,
                    nlp=nlp,
                    t=t_spans[p],
                    x=[np.reshape(first_x[:, i_random], (-1, 1))],
                    u=u[p],  # No need to add noise on the controls, the extra_dynamics should do it for us
                    a=a[p],
                    p=parameters,
                    d=d,
                    method=integrator,
                )
                for i_node in range(nlp.ns + 1):
                    for key in nlp.states.keys():
                        states_integrated = (
                            integrated_sol[i_node][nlp.states[key].index, :]
                            if n_sub_nodes > 1
                            else integrated_sol[i_node][nlp.states[key].index, 0].reshape(-1, 1)
                        )
                        out[p][key][i_node][:, :, i_random] = states_integrated
                first_x[:, i_random] = np.reshape(integrated_sol[-1], (-1,))
        if to_merge:
            out = SolutionData.from_unscaled(self.ocp, out, "x").to_dict(to_merge=to_merge, scaled=False)

        return out if len(out) > 1 else out[0]

    def _states_for_phase_integration(
        self,
        shooting_type: Shooting,
        phase_idx: Int,
        integrated_states: NpArray,
        decision_states: AnyList,
        decision_controls: AnyList,
        params: AnyDict,
        decision_algebraic_states,
    ) -> Any:
        """
        Returns the states to integrate for the phase_idx phase. If there was a phase transition, the last state of the
        previous phase is transformed into the first state of the next phase

        Parameters
        ----------
        shooting_type
            The shooting type to use
        phase_idx
            The phase index of the next phase to integrate
        integrated_states
            The states integrated from the previous phase
        decision_states
            The decision states merged with SolutionMerge.KEYS
        decision_controls
            The decision controls merged with SolutionMerge.KEYS
        params
            The parameters merged with SolutionMerge.KEYS
        decision_algebraic_states
            The algebraic_states merged with SolutionMerge.KEYS

        Returns
        -------
        The states to integrate
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        # In the case of multiple shootings, we don't need to do anything special
        if shooting_type == Shooting.MULTIPLE:
            return decision_states[phase_idx]

        # At first phase, return the normal decision states.
        if phase_idx == 0:
            return [decision_states[phase_idx][0]]

        penalty = self.ocp.phase_transitions[phase_idx - 1]

        t0 = PenaltyHelpers.t0(penalty, 0, lambda p, n: self._stepwise_times[p][n][0])
        dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        # Compute the error between the last state of the previous phase and the first state of the next phase
        # based on the phase transition objective or constraint function. That is why we need to concatenate
        # twice the last state
        x = PenaltyHelpers.states(penalty, 0, lambda p, n, sn: integrated_states[-1])

        u = PenaltyHelpers.controls(
            penalty,
            0,
            lambda p, n, sn: (
                decision_controls[p][n][:, sn.index()] if n < len(decision_controls[p]) else np.ndarray((0, 1))
            ),
        )
        a = PenaltyHelpers.states(
            penalty,
            0,
            lambda p, n, sn: (
                decision_algebraic_states[p][n][:, sn.index()]
                if n < len(decision_algebraic_states[p])
                else np.ndarray((0, 1))
            ),
        )
        d_tp = PenaltyHelpers.numerical_timeseries(
            penalty,
            0,
            lambda p, n, sn: get_numerical_timeseries(self.ocp, p, n, sn),
        )
        d = np.array([]) if d_tp.shape == (0, 0) else np.array(d_tp)

        dx = penalty.function[-1](t0, dt, x, u, params, a, d)
        if dx.shape[0] != decision_states[phase_idx][0].shape[0]:
            raise RuntimeError(
                f"Phase transition must have the same number of states ({dx.shape[0]}) "
                f"when integrating with Shooting.SINGLE. If it is not possible, "
                f"please integrate with Shooting.SINGLE_DISCONTINUOUS_PHASE."
            )

        return [(integrated_states[-1] if shooting_type == Shooting.SINGLE else decision_states[phase_idx][0]) + dx]

    def _integrate_stepwise(self) -> None:
        """
        This method integrate to stepwise level the states. That is the states that are used in the dynamics and
        continuity constraints.

        Returns
        -------
        dict
            The integrated data structure similar in structure to the original _decision_states
        """
        from ...interfaces.interface_utils import get_numerical_timeseries

        params = self._parameters.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)[0][0]
        t_spans = self.t_span(time_alignment=TimeAlignment.CONTROLS)
        if len(self.ocp.nlp) == 1:
            t_spans = [t_spans]
        x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)
        a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=False)

        unscaled: list = [None] * len(self.ocp.nlp)
        for p, nlp in enumerate(self.ocp.nlp):
            d = []
            for n_idx in range(nlp.ns + 1):
                d_tp = get_numerical_timeseries(self.ocp, p, n_idx, 0)
                if d_tp.shape == (0, 0):
                    d += [np.array([])]
                else:
                    d += [np.array(d_tp)]

            integrated_sol = solve_ivp_interface(
                list_of_dynamics=[nlp.dynamics_func] * nlp.ns,
                shooting_type=Shooting.MULTIPLE,
                nlp=nlp,
                t=t_spans[p],
                x=x[p],
                u=u[p],
                a=a[p],
                p=params,
                d=d,
                method=SolutionIntegrator.OCP,
            )

            unscaled[p] = {}
            for key in nlp.states.keys():
                unscaled[p][key] = [None] * nlp.n_states_nodes
                for ns, sol_ns in enumerate(integrated_sol):
                    unscaled[p][key][ns] = sol_ns[nlp.states[key].index, :]

        self._stepwise_states = SolutionData.from_unscaled(self.ocp, unscaled, "x")

    def _return_time_vector(
        self, to_merge: SolutionMerge | list[SolutionMerge], duplicated_times: Bool
    ) -> AnyList | NpArray:
        """
        Returns the time vector at each node that matches stepwise_states or stepwise_controls
        Parameters
        ----------
        to_merge: SolutionMerge | list[SolutionMerge]
            The merge type to perform. If None, then no merge is performed.
        duplicated_times: bool
            If the times should be duplicated for each node.
            If False, then the returned time vector will not have any duplicated times.
        Returns
        -------
        The time vector at each node that matches stepwise_states or stepwise_controls
        """
        if to_merge is None:
            to_merge = []
        if isinstance(to_merge, SolutionMerge):
            to_merge = [to_merge]
        if SolutionMerge.NODES and SolutionMerge.PHASES in to_merge:
            time_vector = np.concatenate(self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times))
        elif SolutionMerge.NODES in to_merge:
            time_vector = self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times)
            for i in range(len(self.ocp.nlp)):
                time_vector[i] = np.concatenate(time_vector[i])
        else:
            time_vector = self.stepwise_time(to_merge=to_merge, duplicated_times=duplicated_times)
        return time_vector

    def interpolate(self, n_frames: Int | AnyIterable, scaled: Bool = False) -> AnyList | AnyDict:
        """
        Interpolate the states

        Parameters
        ----------
        n_frames: int | list | tuple
            If the value is an int, the Solution returns merges the phases,
            otherwise, it interpolates them independently
        scaled: bool
            If the states should be scaled or not (note that scaled is as Ipopt received them, while unscaled is as the
            model needs temps). If you don't know what it means, you probably want the unscaled version.

        Returns
        -------
        A Solution data structure with the states integrated. The controls are removed from this structure
        """

        if self._stepwise_states is None:
            self._integrate_stepwise()

        # Get the states, but do not bother the duplicates now
        if isinstance(n_frames, int):  # So merge phases
            t_all = [self.stepwise_time(to_merge=[SolutionMerge.ALL])]
            states = [self._stepwise_states.to_dict(scaled=scaled, to_merge=SolutionMerge.ALL)]
            n_frames = [n_frames]

        elif not isinstance(n_frames, (list, tuple)) or len(n_frames) != len(self._stepwise_states.unscaled):
            raise ValueError(
                "n_frames should either be an int to merge_phases phases "
                "or a list of int of the number of phases dimension"
            )

        else:
            t_all = self.stepwise_time(to_merge=[SolutionMerge.NODES])
            if len(self.ocp.nlp) == 1:
                t_all = [t_all]
            states = self._stepwise_states.to_dict(scaled=scaled, to_merge=[SolutionMerge.KEYS, SolutionMerge.NODES])

        data = []
        for p in range(len(states)):
            data.append({})

            nlp = self.ocp.nlp[p]

            # Now remove the duplicates
            t_round = np.round(t_all[p], decimals=8)  # Otherwise, there are some numerical issues with np.unique
            t, idx = np.unique(t_round, return_index=True)
            x = states[p][:, idx]

            x_interpolated = np.ndarray((x.shape[0], n_frames[p]))
            t_interpolated = np.linspace(t_round[0], t_round[-1], n_frames[p])
            for j in range(x.shape[0]):
                s = sci_interp.splrep(t, x[j, :], k=1)
                x_interpolated[j, :] = sci_interp.splev(t_interpolated, s)[:, 0]

            for key in nlp.states.keys():
                data[p][key] = x_interpolated[nlp.states[key].index, :]

        return data if len(data) > 1 else data[0]

    def graphs(
        self,
        automatically_organize: Bool = True,
        show_bounds: Bool = False,
        show_now: Bool = True,
        shooting_type: Shooting = Shooting.MULTIPLE,
        integrator: SolutionIntegrator = SolutionIntegrator.OCP,
        save_name: StrOptional = None,
        show_gcom_plot: Bool = False,
        show_interactive_stability_plot: Bool = False,
    ) -> list[plt.figure]:
        """
        Show the graphs of the simulation

        Parameters
        ----------
        automatically_organize: bool
            If the figures should be spread on the screen automatically
        show_bounds: bool
            If the plot should adapt to bounds (True) or to data (False)
        show_now: bool
            If the show method should be called. This is blocking
        shooting_type: Shooting
            The type of interpolation
        integrator: SolutionIntegrator
            Use the scipy solve_ivp integrator for RungeKutta 45 instead of currently defined integrator
        save_name: str
            If a name is provided, the figures will be saved with this name
        show_gcom_plot: bool
            If a ground-projected center of mass plot should be added as a separate figure
        show_interactive_stability_plot: bool
            If an interactive stability plot with a node slider should be added as a separate figure
        """

        plot_ocp = self.ocp.prepare_plots(automatically_organize, show_bounds, shooting_type, integrator)
        self.ocp.plot_ipopt_outputs = False  # This plot is not possible on solutions (only in live plots)
        plot_ocp.update_data(*plot_ocp.parse_data(**{"x": self.vector}))
        if show_gcom_plot:
            self._plot_ground_projected_com()
        if show_interactive_stability_plot:
            self._plot_interactive_stability()
        if save_name:
            if save_name.endswith(".png"):
                save_name = save_name[:-4]
            for i_fig, name_fig in enumerate(plt.get_figlabels()):
                fig = plt.figure(i_fig + 1)
                fig.savefig(f"{save_name}_{name_fig}.png", format="png")
        if show_now:
            plt.show()

        # Returning the figures for the tests
        fig_list = [plt.figure(i_fig + 1) for i_fig in range(len(plt.get_figlabels()))]
        return fig_list

    def _ground_projected_com(self) -> list[tuple[np.ndarray, np.ndarray, str]]:
        params = self.parameters
        if isinstance(params, dict):
            if params:
                params = np.concatenate([np.asarray(value, dtype=float).reshape(-1, 1) for value in params.values()])
            else:
                params = np.array([])
        else:
            params = np.asarray(params, dtype=float)

        states = self.decision_states(scaled=False, to_merge=SolutionMerge.NODES)
        if not isinstance(states, list):
            states = [states]

        com_data: list[tuple[np.ndarray, np.ndarray, str]] = []
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            phase_states = states[phase_idx]

            if "q" in phase_states:
                q = np.asarray(phase_states["q"], dtype=float)
            elif "q_u" in phase_states and hasattr(nlp.model, "compute_q_from_u_iterative"):
                q_v_init = getattr(nlp.model, "q_v_init_guess", None)
                if q_v_init is None:
                    q_v_init = np.zeros((getattr(nlp.model, "nb_dependent_joints", 0), 1))
                q = np.asarray(
                    nlp.model.compute_q_from_u_iterative(
                        np.asarray(phase_states["q_u"], dtype=float),
                        np.asarray(q_v_init, dtype=float),
                    ),
                    dtype=float,
                )
            else:
                raise RuntimeError(
                    "Unable to build the ground-projected CoM plot because the solution does not expose q or q_u."
                )

            if q.ndim == 1:
                q = q[:, np.newaxis]

            com_fun = nlp.model.center_of_mass()
            com_xy = np.zeros((q.shape[1], 2))
            for node_idx in range(q.shape[1]):
                com = np.asarray(com_fun(q[:, node_idx][:, np.newaxis], params), dtype=float).squeeze()
                com_xy[node_idx, :] = com[:2]

            com_data.append((com_xy[:, 0], com_xy[:, 1], f"phase {phase_idx + 1}"))

        return com_data

    def _plot_ground_projected_com(self) -> None:
        com_data = self._ground_projected_com()
        fig = plt.figure("ground_projected_com")
        ax = fig.gca()
        for x, y, label in com_data:
            ax.plot(x, y, label=label if len(com_data) > 1 else None)

        x_values = np.concatenate([x for x, _, _ in com_data])
        y_values = np.concatenate([y for _, y, _ in com_data])
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_span = x_max - x_min
        y_span = y_max - y_min
        target_span = max(x_span, y_span)
        min_span = max(target_span * 0.5, 1e-6)

        def _expand_range(min_value: float, max_value: float, required_span: float) -> tuple[float, float]:
            span = max_value - min_value
            if span >= required_span:
                return min_value, max_value
            padding = (required_span - span) / 2
            return min_value - padding, max_value + padding

        x_min, x_max = _expand_range(x_min, x_max, min_span)
        y_min, y_max = _expand_range(y_min, y_max, min_span)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title("Ground-projected center of mass")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True)
        if len(com_data) > 1:
            ax.legend()

    def _plot_interactive_stability(self) -> None:
        frames = self._interactive_stability_frames()
        if not frames:
            raise RuntimeError("Unable to build the interactive stability plot because no nodes were found.")

        fig, ax = plt.subplots(num="interactive_stability_plot")
        fig.subplots_adjust(bottom=0.24, right=0.82)

        support_line, = ax.plot(
            [], [], "-o", color="tab:blue", lw=1.8, ms=4, label="selected support polygon", zorder=6
        )
        com_point = ax.scatter([], [], s=55, color="tab:blue", label="CoM", zorder=7)
        zmp_point = ax.scatter([], [], s=75, color="red", label="ZMP", zorder=8)
        history_support_lines = [
            ax.plot([], [], color="0.7", lw=1.0, alpha=0.7, zorder=2)[0] for _ in frames
        ]
        history_zmp_points = ax.scatter([], [], s=12, color="red", alpha=0.55, label="_nolegend_", zorder=3)
        info_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left")
        all_state = {"enabled": False}

        def _set_scatter_offsets(scatter, x: float | None, y: float | None) -> None:
            if x is None or y is None:
                scatter.set_offsets(np.empty((0, 2)))
            else:
                scatter.set_offsets(np.array([[x, y]], dtype=float))

        def _update(frame_index: int) -> None:
            frame = frames[int(frame_index)]
            polygon_center = frame["support_center"] if all_state["enabled"] else np.zeros(2)
            polygon = frame["support_polygon"] - polygon_center if frame["support_polygon"].size else frame["support_polygon"]
            if polygon.size == 0:
                support_line.set_data([], [])
            else:
                closed_polygon = np.vstack([polygon, polygon[0]]) if len(polygon) > 1 else polygon
                support_line.set_data(closed_polygon[:, 0], closed_polygon[:, 1])

            _set_scatter_offsets(com_point, frame["com"][0] - polygon_center[0], frame["com"][1] - polygon_center[1])
            if frame["zmp"] is None:
                _set_scatter_offsets(zmp_point, None, None)
            else:
                _set_scatter_offsets(
                    zmp_point,
                    frame["zmp"][0] - polygon_center[0],
                    frame["zmp"][1] - polygon_center[1],
                )

            if all_state["enabled"]:
                for i_frame, history_line in enumerate(history_support_lines):
                    history_frame = frames[i_frame]
                    history_polygon = history_frame["support_polygon"]
                    history_center = history_frame["support_center"]
                    if history_polygon.size == 0:
                        history_line.set_data([], [])
                        history_line.set_visible(False)
                        continue
                    centered_history_polygon = history_polygon - history_center
                    closed_history_polygon = (
                        np.vstack([centered_history_polygon, centered_history_polygon[0]])
                        if len(centered_history_polygon) > 1
                        else centered_history_polygon
                    )
                    history_line.set_data(closed_history_polygon[:, 0], closed_history_polygon[:, 1])
                    history_line.set_visible(True)

                history_zmp_points.set_offsets(
                    np.array(
                        [
                            history_frame["zmp"] - history_frame["support_center"]
                            for history_frame in frames
                            if history_frame["zmp"] is not None
                        ],
                        dtype=float,
                    )
                    if any(history_frame["zmp"] is not None for history_frame in frames)
                    else np.empty((0, 2))
                )
                history_zmp_points.set_visible(True)
                support_line.set_color("tab:blue")
                support_line.set_linewidth(1.8)
                com_point.set_color("tab:blue")
                zmp_point.set_color("red")
            else:
                for history_line in history_support_lines:
                    history_line.set_visible(False)
                history_zmp_points.set_offsets(np.empty((0, 2)))
                history_zmp_points.set_visible(False)

            info_text.set_text(frame["label"])
            ax.set_title("Interactive stability plot" + (" - All" if all_state["enabled"] else ""))
            fig.canvas.draw_idle()

        global_points: list[np.ndarray] = []
        for frame in frames:
            if frame["support_polygon"].size:
                global_points.append(frame["support_polygon"])
            global_points.append(np.asarray(frame["com"], dtype=float)[np.newaxis, :])
            if frame["zmp"] is not None:
                global_points.append(np.asarray(frame["zmp"], dtype=float)[np.newaxis, :])

        points = np.concatenate(global_points, axis=0)
        x_min, x_max = float(np.min(points[:, 0])), float(np.max(points[:, 0]))
        y_min, y_max = float(np.min(points[:, 1])), float(np.max(points[:, 1]))
        x_min, x_max, y_min, y_max = self._balanced_xy_limits(x_min, x_max, y_min, y_max)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True)

        ax_check = fig.add_axes([0.85, 0.62, 0.12, 0.1])
        check = CheckButtons(ax_check, ["All"], [False])

        def _toggle_all(_label: str) -> None:
            all_state["enabled"] = not all_state["enabled"]
            _update(int(slider.val) if len(frames) > 1 else 0)

        check.on_clicked(_toggle_all)
        fig._bioptim_checkbuttons = check  # Keep the widget alive for the lifetime of the figure

        if len(frames) > 1:
            ax_slider = fig.add_axes([0.15, 0.08, 0.7, 0.04])
            slider = Slider(ax_slider, "node", 0, len(frames) - 1, valinit=0, valstep=1)
            slider.on_changed(_update)
            fig._bioptim_slider = slider  # Keep the widget alive for the lifetime of the figure
        else:
            slider = None

        _update(0)

    def _interactive_stability_frames(self) -> list[dict[str, Any]]:
        params = self.parameters
        if isinstance(params, dict):
            if params:
                params = np.concatenate([np.asarray(value, dtype=float).reshape(-1, 1) for value in params.values()])
            else:
                params = np.array([])
        else:
            params = np.asarray(params, dtype=float)

        states = self.decision_states(scaled=False, to_merge=SolutionMerge.NODES)
        controls = self.decision_controls(scaled=False, to_merge=SolutionMerge.NODES)
        algebraic_states = self.decision_algebraic_states(scaled=False, to_merge=SolutionMerge.NODES)
        times = self.stepwise_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES)

        if not isinstance(states, list):
            states = [states]
        if not isinstance(controls, list):
            controls = [controls]
        if not isinstance(algebraic_states, list):
            algebraic_states = [algebraic_states]
        if not isinstance(times, list):
            times = [times]

        frames: list[dict[str, Any]] = []
        for phase_idx, nlp in enumerate(self.ocp.nlp):
            phase_states = states[phase_idx]
            phase_controls = controls[phase_idx]
            phase_algebraic_states = algebraic_states[phase_idx]
            phase_times = np.asarray(times[phase_idx], dtype=float).reshape(-1)

            q, qdot = self._phase_q_and_qdot_from_states(phase_states, nlp)
            com_fun = nlp.model.center_of_mass()
            n_contacts = getattr(nlp.model, "nb_rigid_contacts", 0)

            for node_idx in range(q.shape[1]):
                q_node = q[:, node_idx][:, np.newaxis]
                com_xy = np.asarray(com_fun(q_node, params), dtype=float).squeeze()[:2]

                contact_positions = np.zeros((n_contacts, 3))
                for i_contact in range(n_contacts):
                    contact_position = np.asarray(nlp.model.rigid_contact_position(i_contact)(q_node, params), dtype=float).squeeze()
                    contact_positions[i_contact, :] = contact_position[:3]

                contact_forces = self._contact_forces_at_node(
                    nlp=nlp,
                    phase_states=phase_states,
                    phase_controls=phase_controls,
                    phase_algebraic_states=phase_algebraic_states,
                    params=params,
                    q=q,
                    qdot=qdot,
                    phase_times=phase_times,
                    node_idx=node_idx,
                    n_contacts=n_contacts,
                )
                support_points = contact_positions[:, :2]
                zmp = None
                if contact_forces is not None:
                    active_contacts = np.linalg.norm(contact_forces, axis=1) > 1e-8
                    if np.any(active_contacts):
                        support_points = support_points[active_contacts]
                        zmp = self._compute_zmp(contact_positions[active_contacts], contact_forces[active_contacts])

                frames.append(
                    {
                        "label": f"phase {phase_idx + 1}, node {node_idx}",
                        "com": com_xy,
                        "support_polygon": self._convex_hull_2d(support_points),
                        "support_center": self._polygon_center(support_points),
                        "zmp": zmp,
                    }
                )

        return frames

    @staticmethod
    def _phase_q_and_qdot_from_states(phase_states: AnyDict, nlp: Any) -> tuple[np.ndarray, np.ndarray]:
        if "q" in phase_states:
            q = np.asarray(phase_states["q"], dtype=float)
        elif "q_u" in phase_states and hasattr(nlp.model, "compute_q_from_u_iterative"):
            q_v_init = getattr(nlp.model, "q_v_init_guess", None)
            if q_v_init is None:
                q_v_init = np.zeros((getattr(nlp.model, "nb_dependent_joints", 0), 1))
            q = np.asarray(
                nlp.model.compute_q_from_u_iterative(
                    np.asarray(phase_states["q_u"], dtype=float),
                    np.asarray(q_v_init, dtype=float),
                ),
                dtype=float,
            )
        else:
            raise RuntimeError("Unable to build the stability plot because the solution does not expose q or q_u.")

        q = q if q.ndim > 1 else q[:, np.newaxis]

        if "qdot" in phase_states:
            qdot = np.asarray(phase_states["qdot"], dtype=float)
        elif "qdot_u" in phase_states and hasattr(nlp.model, "compute_qdot"):
            qdot_u = np.asarray(phase_states["qdot_u"], dtype=float)
            qdot = np.asarray(nlp.model.compute_qdot(q, qdot_u), dtype=float)
        elif "qdot_u" in phase_states:
            qdot = np.asarray(phase_states["qdot_u"], dtype=float)
        else:
            raise RuntimeError("Unable to build the stability plot because the solution does not expose qdot or qdot_u.")

        return q, qdot if qdot.ndim > 1 else qdot[:, np.newaxis]

    @staticmethod
    def _contact_forces_at_node(
        nlp: Any,
        phase_states: AnyDict,
        phase_controls: AnyDict,
        phase_algebraic_states: AnyDict,
        params: np.ndarray,
        q: np.ndarray,
        qdot: np.ndarray,
        phase_times: np.ndarray,
        node_idx: int,
        n_contacts: int,
    ) -> np.ndarray | None:
        if n_contacts == 0:
            return None

        contact_axes_per_point: list[list[int]] = []
        for i_contact in range(n_contacts):
            try:
                axes = list(nlp.model.rigid_contact_axes_index(i_contact))
            except Exception:
                axes = [2]
            contact_axes_per_point.append(axes if axes else [2])

        if hasattr(nlp, "rigid_contact_forces_func") and nlp.rigid_contact_forces_func is not None:
            control_idx = 0
            if phase_controls:
                first_control_key = next(iter(phase_controls))
                n_control_nodes = phase_controls[first_control_key].shape[1]
                if n_control_nodes > 0:
                    interval_size = max(1, (q.shape[1] - 1) // n_control_nodes)
                    control_idx = min(node_idx // interval_size, n_control_nodes - 1)

            x_node = Solution._stack_node_vector(phase_states, node_idx, nlp.states.keys())
            u_node = Solution._stack_node_vector(phase_controls, control_idx, nlp.controls.keys())
            a_node = Solution._stack_node_vector(phase_algebraic_states, node_idx, nlp.algebraic_states.keys())

            if phase_times.size == 0:
                t_span = np.array([0.0, 0.0])
            else:
                t0 = phase_times[min(node_idx, phase_times.size - 1)]
                t1 = phase_times[min(node_idx + 1, phase_times.size - 1)]
                t_span = np.array([t0, t1], dtype=float)

            try:
                force = nlp.rigid_contact_forces_func(t_span, x_node, u_node, params, a_node, np.array([]))
                force = np.asarray(force, dtype=float).reshape(-1)
                mapped = Solution._map_contact_forces_to_xyz(force, n_contacts, contact_axes_per_point)
                if mapped is not None:
                    return mapped
            except Exception:
                pass

        for key in ("contact_forces", "rigid_contact_forces"):
            for phase_data, node_idx_data in (
                (phase_states, node_idx),
                (phase_controls, control_idx if "control_idx" in locals() else node_idx),
                (phase_algebraic_states, node_idx),
            ):
                if key in phase_data:
                    force_data = np.asarray(phase_data[key], dtype=float)
                    if force_data.ndim == 1:
                        node_forces = force_data
                    else:
                        if node_idx_data >= force_data.shape[1]:
                            continue
                        node_forces = force_data[:, node_idx_data]

                    node_forces = np.asarray(node_forces, dtype=float).reshape(-1)
                    mapped = Solution._map_contact_forces_to_xyz(node_forces, n_contacts, contact_axes_per_point)
                    if mapped is not None:
                        return mapped

        return None

    @staticmethod
    def _map_contact_forces_to_xyz(
        force_vector: np.ndarray, n_contacts: int, contact_axes_per_point: list[list[int]]
    ) -> np.ndarray | None:
        force_vector = np.asarray(force_vector, dtype=float).reshape(-1)
        if force_vector.size == 0:
            return None

        total_contact_axes = sum(len(axes) for axes in contact_axes_per_point)

        # Most general case: the vector is packed per contact using available contact axes.
        if force_vector.size == total_contact_axes:
            contact_forces = np.zeros((n_contacts, 3))
            idx = 0
            for i_contact, axes in enumerate(contact_axes_per_point):
                for axis in axes:
                    if axis in (0, 1, 2):
                        contact_forces[i_contact, axis] = force_vector[idx]
                    idx += 1
            return contact_forces

        # Common case for single-axis contacts: one force scalar per contact.
        if force_vector.size == n_contacts:
            contact_forces = np.zeros((n_contacts, 3))
            for i_contact, axes in enumerate(contact_axes_per_point):
                normal_axis = axes[-1] if axes else 2
                if normal_axis not in (0, 1, 2):
                    normal_axis = 2
                contact_forces[i_contact, normal_axis] = force_vector[i_contact]
            return contact_forces

        # Fallback for already-expanded 3D forces. Try both common flattening conventions.
        if force_vector.size == 3 * n_contacts:
            by_contact = np.reshape(force_vector, (3, n_contacts), order="F").T
            by_axis_block = np.reshape(force_vector, (3, n_contacts), order="C").T

            def off_axis_score(candidate: np.ndarray) -> float:
                score = 0.0
                for i_contact, axes in enumerate(contact_axes_per_point):
                    inactive_axes = [i for i in (0, 1, 2) if i not in axes]
                    if inactive_axes:
                        score += float(np.sum(np.abs(candidate[i_contact, inactive_axes])))
                return score

            return by_contact if off_axis_score(by_contact) <= off_axis_score(by_axis_block) else by_axis_block

        return None

    @staticmethod
    def _stack_node_vector(data: AnyDict, node_idx: int, keys) -> np.ndarray:
        if not data or not keys:
            return np.array([])

        values = []
        for key in keys:
            if key not in data:
                continue
            key_values = np.asarray(data[key], dtype=float)
            if key_values.ndim == 1:
                values.append(key_values.reshape(-1, 1))
            else:
                if node_idx >= key_values.shape[1]:
                    values.append(key_values[:, -1].reshape(-1, 1))
                else:
                    values.append(key_values[:, node_idx].reshape(-1, 1))

        return np.concatenate(values, axis=0) if values else np.array([])

    @staticmethod
    def _compute_zmp(contact_positions: np.ndarray, contact_forces: np.ndarray, tol: float = 1e-8) -> np.ndarray | None:
        net_vertical_force = float(np.sum(contact_forces[:, 2]))
        if abs(net_vertical_force) < tol:
            return None

        net_moment = np.sum(np.cross(contact_positions, contact_forces), axis=0)
        return np.array([-net_moment[1] / net_vertical_force, net_moment[0] / net_vertical_force])

    @staticmethod
    def _convex_hull_2d(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.zeros((0, 2))

        points = np.unique(points.reshape(-1, 2), axis=0)
        if len(points) <= 2:
            return points

        points = points[np.lexsort((points[:, 1], points[:, 0]))]

        def _cross(origin: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
            return float((a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]))

        lower: list[np.ndarray] = []
        for point in points:
            while len(lower) >= 2 and _cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)

        upper: list[np.ndarray] = []
        for point in reversed(points):
            while len(upper) >= 2 and _cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)

        return np.asarray(lower[:-1] + upper[:-1], dtype=float)

    @staticmethod
    def _balanced_xy_limits(x_min: float, x_max: float, y_min: float, y_max: float) -> tuple[float, float, float, float]:
        x_span = x_max - x_min
        y_span = y_max - y_min
        target_span = max(x_span, y_span)
        minimum_span = max(target_span * 0.5, 1e-6)

        def _expand_range(min_value: float, max_value: float, required_span: float) -> tuple[float, float]:
            span = max_value - min_value
            if span >= required_span:
                return min_value, max_value
            padding = (required_span - span) / 2
            return min_value - padding, max_value + padding

        x_min, x_max = _expand_range(x_min, x_max, minimum_span)
        y_min, y_max = _expand_range(y_min, y_max, minimum_span)
        return x_min, x_max, y_min, y_max

    @staticmethod
    def _polygon_center(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.size == 0:
            return np.zeros(2)
        return np.mean(points[:, :2], axis=0)

    def animate(
        self,
        n_frames: Int = 0,
        shooting_type: Shooting = None,
        show_now: Bool = True,
        show_tracked_markers: Bool = False,
        viewer: Str = "bioviz",
        **kwargs: Any,
    ) -> AnyListOptional:
        """
        Animate the simulation

        Parameters
        ----------
        n_frames: int
            The number of frames to interpolate to. If the value is 0, the data are merged to a one phase if possible.
            If the value is -1, the data is not merge in one phase
        shooting_type: Shooting
            The Shooting type to animate
        show_now: bool
            If the bioviz exec() function should be called automatically. This is blocking method
        show_tracked_markers: bool
            If the tracked markers should be displayed
        viewer: str
            The viewer to use. Currently, bioviz or pyorerun
        kwargs: Any
            Any parameters to pass to bioviz

        Returns
        -------
            A list of bioviz structures (one for each phase). So one can call exec() by hand
        """

        from ...models.viewer_utils import _check_models_comes_from_same_super_class

        if shooting_type:
            self.integrate(shooting_type=shooting_type)

        _check_models_comes_from_same_super_class(self.ocp.nlp)

        type(self.ocp.nlp[0].model).animate(
            ocp=self.ocp,
            solution=self,
            show_now=show_now,
            show_tracked_markers=show_tracked_markers,
            viewer=viewer,
            n_frames=n_frames,
            **kwargs,
        )

    @staticmethod
    def _dispatch_params(params: AnyDict) -> NpArray:
        values = [params[key][0] for key in params.keys()]
        if values:
            return np.concatenate(values)
        else:
            return np.ndarray((0, 1))

    def _get_penalty_cost(self, penalty: PenaltyOption) -> FloatTuple:
        from ...interfaces.interface_utils import get_numerical_timeseries

        val = []
        val_weighted = []

        phases_dt = PenaltyHelpers.phases_dt(penalty, self.ocp, lambda p: np.array([self.phases_dt[idx] for idx in p]))
        params = PenaltyHelpers.parameters(
            penalty, 0, lambda p_idx, n_idx, sn_idx: self._dispatch_params(self._parameters.scaled[0])
        )

        merged_x = self._decision_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_u = self._stepwise_controls.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        merged_a = self._decision_algebraic_states.to_dict(to_merge=SolutionMerge.KEYS, scaled=True)
        for idx in range(len(penalty.node_idx)):
            t0 = PenaltyHelpers.t0(penalty, idx, lambda p_idx, n_idx: self._stepwise_times[p_idx][n_idx][0])
            x = PenaltyHelpers.states(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_x(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_x),
            )
            u = PenaltyHelpers.controls(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_u(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_u),
            )
            a = PenaltyHelpers.states(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: self._get_x(self.ocp, penalty, p_idx, n_idx, sn_idx, merged_a),
            )
            d_tp = PenaltyHelpers.numerical_timeseries(
                penalty,
                idx,
                lambda p_idx, n_idx, sn_idx: get_numerical_timeseries(self.ocp, p_idx, n_idx, sn_idx),
            )
            d = np.array([]) if d_tp.shape == (0, 0) else np.array(d_tp)

            weight = PenaltyHelpers.weight(penalty, idx)
            target = PenaltyHelpers.target(penalty, idx)

            node_idx = penalty.node_idx[idx]
            val.append(penalty.function_non_threaded[node_idx](t0, phases_dt, x, u, params, a, d))
            val_weighted.append(
                penalty.weighted_function_non_threaded[node_idx](t0, phases_dt, x, u, params, a, d, weight, target)
            )

        if self.ocp.n_threads > 1:
            val = [v[:, 0] for v in val]
            val_weighted = [v[:, 0] for v in val_weighted]

        val = np.nansum(val)
        val_weighted = np.nansum(val_weighted)

        return val, val_weighted

    @staticmethod
    def _get_x(ocp, penalty, phase_idx, node_idx, subnodes_idx, merged_x):
        values = merged_x[phase_idx]
        x = PenaltyHelpers.get_states(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
        return x

    @staticmethod
    def _get_u(ocp, penalty, phase_idx, node_idx, subnodes_idx, merged_u):
        values = merged_u[phase_idx]
        u = PenaltyHelpers.get_controls(ocp, penalty, phase_idx, node_idx, subnodes_idx, values)
        return u

    @property
    def cost(self) -> Float | DM:
        if self._cost is None:
            self._cost = 0
            for J in self.ocp.J:
                _, val_weighted = self._get_penalty_cost(J)
                self._cost += val_weighted

            for idx_phase, nlp in enumerate(self.ocp.nlp):
                for J in nlp.J:
                    _, val_weighted = self._get_penalty_cost(J)
                    self._cost += val_weighted
            self._cost = DM(self._cost)
        return self._cost

    @property
    def detailed_cost(self) -> AnyList:
        if self._detailed_cost is None:
            self._compute_detailed_cost()
        return self._detailed_cost

    def _compute_detailed_cost(self) -> None:
        """
        Adds the detailed objective functions and/or constraints values to sol

        Parameters
        ----------
        """
        self._detailed_cost = []

        for nlp in self.ocp.nlp:
            for penalty in nlp.J_internal + nlp.J:
                if not penalty:
                    continue
                val, val_weighted = self._get_penalty_cost(penalty)
                self._detailed_cost += [
                    {"name": penalty.type.__str__(), "cost_value_weighted": val_weighted, "cost_value": val}
                ]
        for penalty in self.ocp.J:
            val, val_weighted = self._get_penalty_cost(penalty)
            self._detailed_cost += [
                {"name": penalty.type.__str__(), "cost_value_weighted": val_weighted, "cost_value": val}
            ]
        return

    def print_cost(self, cost_type: CostType = CostType.ALL) -> Float:
        """
        Print the objective functions and/or constraints to the console

        Parameters
        ----------
        cost_type: CostType
            The type of cost to console print
        """

        def print_penalty_list(penalties, print_only_weighted):
            running_total = 0

            for penalty in penalties:
                if not penalty:
                    continue

                val, val_weighted = self._get_penalty_cost(penalty)
                running_total += val_weighted

                if penalty.node in [Node.MULTINODES, Node.TRANSITION]:
                    node_name = penalty.node.name
                else:
                    node_name = f"{penalty.node[0]}" if isinstance(penalty.node[0], int) else penalty.node[0].name

                if self._detailed_cost is not None:
                    self._detailed_cost += [
                        {
                            "name": penalty.type.__str__(),
                            "penalty": penalty.type.__str__().split(".")[0],
                            "function": penalty.name,
                            "cost_value_weighted": val_weighted,
                            "cost_value": val,
                            "params": penalty.extra_parameters,
                            "derivative": penalty.derivative,
                            "explicit_derivative": penalty.explicit_derivative,
                            "integration_rule": penalty.integration_rule.name,
                            "weight": penalty.weight,
                            "expand": penalty.expand,
                            "node": node_name,
                        }
                    ]
                if print_only_weighted:
                    print(f"{penalty.type}: {val_weighted}")
                else:
                    print(f"{penalty.type}: {val_weighted} (non weighted  {val})")

            return running_total

        def print_objective_functions(ocp: "OptimalControlProgram") -> None:
            """
            Print the values of each objective function to the console
            """
            print(f"\n---- COST FUNCTION VALUES ----")
            running_total = print_penalty_list(ocp.J_internal, False)
            running_total += print_penalty_list(ocp.J, False)
            if running_total:
                print("")

            for nlp in ocp.nlp:
                print(f"PHASE {nlp.phase_idx}")
                running_total += print_penalty_list(nlp.J_internal, False)
                running_total += print_penalty_list(nlp.J, False)
                print("")

            print(f"Sum cost functions: {running_total}")
            print(f"------------------------------")

        def print_constraints(ocp: "OptimalControlProgram", sol: AnyDict) -> None:
            """
            Print the values of each constraint with its lagrange multiplier to the console
            """

            if sol.constraints is None:
                return

            # Todo, min/mean/max
            print(f"\n--------- CONSTRAINTS ---------")
            if print_penalty_list(ocp.g_internal, True) + print_penalty_list(ocp.g, True):
                print("")

            for idx_phase, nlp in enumerate(ocp.nlp):
                print(f"PHASE {idx_phase}")
                print_penalty_list(nlp.g_internal, True)
                print_penalty_list(nlp.g, True)
                print("")
            print(f"------------------------------")

        if cost_type == CostType.OBJECTIVES:
            print_objective_functions(self.ocp)
        elif cost_type == CostType.CONSTRAINTS:
            print_constraints(self.ocp, self)
        elif cost_type == CostType.ALL:
            print(
                f"Solver reported time: {self.solver_time_to_optimize} sec\n"
                f"Real time: {self.real_time_to_optimize} sec"
            )
            self.print_cost(CostType.OBJECTIVES)
            self.print_cost(CostType.CONSTRAINTS)
        else:
            raise ValueError("print can only be called with CostType.OBJECTIVES or CostType.CONSTRAINTS")
