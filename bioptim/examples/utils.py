import os
import pickle
import platform
import signal
import socket
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

from bioptim.gui.online_callback_abstract import OnlineCallbackAbstract


def bioviz_snapshot_worker(payload_path: str) -> None:
    """
    Entry point of the detached viewer process spawned by `ExampleUtils.animate_q`.

    It runs in a fresh interpreter, so nothing of the optimization is alive here: the whole state is the
    npz payload, which is removed as soon as it has been read. Kept at module level (and not inside
    `ExampleUtils`) so that the child can reach it with a one-line `-c` import.
    """
    import bioviz

    with np.load(payload_path, allow_pickle=False) as payload:
        model_path = str(payload["model_path"].item())
        q = np.asarray(payload["q"], dtype=float)
    try:
        os.remove(payload_path)
    except OSError:
        pass

    viz = bioviz.Viz(model_path)
    viz.load_movement(q)
    viz.exec()


class _KeyPoller:
    """
    Whatever was typed since the previous call, never blocking.

    On Windows the console buffer is read key by key, so a single keystroke is enough. On POSIX the
    terminal is left in its usual line mode (switching it to raw would fight with the blocking `input()`
    prompt the iteration callback also uses), so a key is only seen once Enter has been pressed.
    """

    def __init__(self):
        self._is_windows = platform.system() == "Windows"
        try:
            self.enabled = sys.stdin is not None and sys.stdin.isatty()
        except ValueError:  # stdin was closed
            self.enabled = False

    @property
    def needs_enter(self) -> bool:
        return not self._is_windows

    def poll(self) -> str:
        if not self.enabled:
            return ""
        try:
            return self._poll_windows() if self._is_windows else self._poll_posix()
        except Exception:
            self.enabled = False  # no usable console after all, e.g. the script runs detached
            return ""

    def _poll_windows(self) -> str:
        import msvcrt

        pressed = ""
        while msvcrt.kbhit():
            char = msvcrt.getwch()
            if char in ("\x00", "\xe0"):  # arrow/function key, the second half carries no text
                msvcrt.getwch()
                continue
            pressed += char
        return pressed.lower()

    def _poll_posix(self) -> str:
        import select

        pressed = ""
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if not line:
                break
            pressed += line
        return pressed.lower()


class _static_property:
    def __init__(self, func):
        self.func = func

    def __get__(self, obj, objtype=None):
        return self.func()


class ExampleUtils:
    @_static_property
    def folder() -> str:
        """Returns the path to the examples folder."""
        return ExampleUtils._capitalize_folder_drive(str(Path(__file__).parent))

    @staticmethod
    def _capitalize_folder_drive(folder: str) -> str:
        if platform.system() == "Windows" and folder[1] == ":":
            # Capitalize the drive letter if it is windows
            folder = folder[0].upper() + folder[1:]
        return folder

    @staticmethod
    def _full_q_from_solution(ocp, sol) -> np.ndarray | None:
        """
        The full-dof trajectory (n_q x n_nodes), ready for bioviz without rebuilding anything.

        A holonomic transcription only carries the independent coordinates `q_u`, so the dependent ones
        are recovered by the model. Returns None if the ocp declares neither `q` nor `q_u`.

        Every column is recovered by a Newton solve warm-started on the previous column, so the columns
        are deliberately kept contiguous: subsampling here would lengthen the jump between consecutive
        solves and weaken that warm start, which matters when the iterate is still far from feasible.
        Drop frames afterwards instead, which is what `animate_q(..., stride=...)` does.
        """
        from bioptim import SolutionMerge

        to_merge = SolutionMerge.NODES if len(ocp.nlp) == 1 else [SolutionMerge.NODES, SolutionMerge.PHASES]
        states = sol.decision_states(to_merge=to_merge)

        if "q" in states:
            return np.asarray(states["q"], dtype=float)

        model = ocp.nlp[0].model
        if "q_u" in states and hasattr(model, "compute_q_from_u_iterative"):
            q_v_init = getattr(model, "q_v_init_guess", None)
            q_v_init = None if q_v_init is None else np.array(q_v_init, dtype=float)
            return np.asarray(model.compute_q_from_u_iterative(np.array(states["q_u"], dtype=float), q_v_init))

        return None

    @staticmethod
    def animate_q(model_path: str | Path, q: np.ndarray, stride: int = 1, log_dir: str | Path = None) -> subprocess.Popen | None:
        """
        Open a bioviz window on `q` in a detached process and return immediately.

        bioviz owns a Qt event loop and `viz.exec()` only returns when the window is closed, so a viewer
        started in-process would freeze whatever called it. A brand new interpreter is launched instead,
        which also keeps VTK away from an interpreter that already holds the solver; the trajectory travels
        through an npz file that the child deletes once loaded. The caller is free to open as many viewers
        as it likes, they are independent of each other and outlive the caller.

        Parameters
        ----------
        model_path: str | Path
            The .bioMod to display, typically `ocp.nlp[0].model.path`
        q: np.ndarray
            The full-dof trajectory (n_q x n_frames)
        stride: int
            Keep only every `stride`-th frame, e.g. to drop the collocation points between shooting nodes
        log_dir: str | Path
            Where to write the child's output. Defaults to a `bioptim_viewers` folder in the temp directory

        Returns
        -------
        The viewer process, or None if it could not be started
        """
        q = np.asarray(q, dtype=float)
        if q.ndim != 2 or q.shape[1] == 0:
            print(f">>> cannot animate a trajectory of shape {q.shape}")
            return None
        if stride > 1:
            q = q[:, ::stride]

        log_dir = Path(tempfile.gettempdir()) / "bioptim_viewers" if log_dir is None else Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%H_%M_%S_%f")

        payload_path = log_dir / f"snapshot_{stamp}.npz"
        np.savez(payload_path, model_path=np.array(str(model_path)), q=q)

        log_path = log_dir / f"snapshot_{stamp}.log"
        command = [
            sys.executable,
            "-c",
            "import sys; from bioptim.examples.utils import bioviz_snapshot_worker; "
            "bioviz_snapshot_worker(sys.argv[1])",
            str(payload_path),
        ]
        # The child writes to its own log instead of the console, otherwise its startup chatter would be
        # interleaved with the IPOPT iteration table of a solve still in progress.
        try:
            log_file = open(log_path, "w")
            return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT)
        except OSError as error:
            print(f">>> could not start the viewer: {error}")
            return None

    @staticmethod
    def save_solution(
        ocp,
        sol,
        out_path: str | Path = None,
        folder: str | Path = "solutions",
        extra: dict = None,
    ) -> Path:
        """
        Pickle a solved solution so that it can be inspected later on.

        The OCP itself cannot be pickled (it holds CasADi MX symbols and the Python
        closures of the custom penalties), but the solution without its ocp attribute is pure data
        (numpy arrays + a CasADi DM). The ocp is therefore detached before dumping and reattached
        afterwards, so the solution handed in stays fully usable. On load, rebuild an unsolved ocp with
        the very same prepare function.

        Parameters
        ----------
        ocp: OptimalControlProgram
            The ocp that produced the solution
        sol: Solution
            The solution to store
        out_path: str | Path
            The file to write. Defaults to `<folder>/<dd_MM_yyyy_HH_MM>.pkl`
        folder: str | Path
            The folder of the default file name; created if it does not exist. Ignored if `out_path` is given
        extra: dict
            Additional entries to store in the payload, e.g. the variant of a parametrized prepare function

        Returns
        -------
        The path the solution was written to
        """
        import bioptim

        if out_path is None:
            output_dir = Path(folder)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_path = output_dir / f"{datetime.now().strftime('%d_%m_%Y_%H_%M')}.pkl"
        else:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

        q = ExampleUtils._full_q_from_solution(ocp, sol)
        meta = {
            "bioptim_version": bioptim.__version__,
            "solved_on": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "status": None if sol.status is None else int(sol.status),
            "iterations": sol.iterations,
            "cost": None if sol.cost is None else float(sol.cost),
            "solver_time_to_optimize": sol.solver_time_to_optimize,
            "real_time_to_optimize": sol.real_time_to_optimize,
        }

        payload = {
            "model_path": [getattr(nlp.model, "path", None) for nlp in ocp.nlp],
            "n_phases": len(ocp.nlp),
            "vector_size": int(sol.vector.shape[0]),  # sanity check against the rebuilt ocp
            "q": q,
            "meta": meta,
            "solution": sol,
        }
        if extra is not None:
            payload.update(extra)

        ocp_backup = sol.ocp
        del sol.ocp
        try:
            with open(out_path, "wb") as file:
                pickle.dump(payload, file, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            sol.ocp = ocp_backup

        cost = "None" if meta["cost"] is None else f"{meta['cost']:.6g}"
        print(f"\nSaved solution to {out_path}")
        print(f"  phases={payload['n_phases']}  status={meta['status']} (0 = converged)  "
              f"iterations={meta['iterations']}  cost={cost}")
        print(f"  q shape={None if q is None else q.shape}  bioptim={meta['bioptim_version']}")

        return out_path

    @staticmethod
    def load_solution(
        path: str | Path,
        ocp=None,
        rebuild_q: bool = False,
        add_constraint_plots: bool = True,
    ) -> tuple:
        """
        Load a solution pickled by `save_solution` and reattach an ocp to it.

        The pickled `Solution` arrives without its `ocp` (CasADi MX symbols are not picklable), so almost
        everything it can do (graphs(), print_cost(), integrate(), decision_states(), ...) is dead until an
        ocp is put back. Rebuild an *unsolved* ocp with exactly the same prepare function the solving side
        used and hand it in here; rebuilding costs a few seconds, no optimization is run.

        Parameters
        ----------
        path: str | Path
            The .pkl file written by `save_solution`
        ocp: OptimalControlProgram
            A freshly built, unsolved ocp, structurally identical to the one that produced the solution.
            If None, the raw payload is returned and the Solution stays inert
        rebuild_q: bool
            Recompute the full-dof trajectory locally instead of using the one stored at save time.
            Requires `ocp`
        add_constraint_plots: bool
            Register the constraint penalties on the ocp, which has to happen before `sol.graphs()`

        Returns
        -------
        (sol, payload), where `payload` is the stored dictionary with `payload["q"]` up to date
        """
        import bioptim
        from bioptim import CostType

        with open(path, "rb") as file:
            payload = pickle.load(file)

        sol = payload["solution"]
        meta = payload["meta"]

        if meta["bioptim_version"] != bioptim.__version__:
            print(
                f"WARNING: the solution was produced with bioptim {meta['bioptim_version']} but this "
                f"machine runs {bioptim.__version__}. The unpickled Solution is interpreted by the "
                f"*local* bioptim, so results may be wrong or loading may fail."
            )

        cost = "None" if meta["cost"] is None else f"{meta['cost']:.6g}"
        solve_time = "None" if meta["real_time_to_optimize"] is None else f"{meta['real_time_to_optimize']:.1f} s"
        print(f"Loaded {path}")
        print(f"  solved on {meta['host']} at {meta['solved_on']}")
        print(f"  phases={payload['n_phases']}  status={meta['status']} (0 = converged)  "
              f"iterations={meta['iterations']}")
        print(f"  cost={cost}  solve time={solve_time}")

        if ocp is None:
            print("  no ocp given, the Solution is returned without one and stays inert")
            return sol, payload

        # The stored quantities are read out of the solution vector using the ocp layout, so a mismatch
        # here means the two sides do not agree on the transcription.
        rebuilt_size = int(ocp.variables_vector.shape[0])
        if rebuilt_size != payload["vector_size"]:
            raise RuntimeError(
                f"The rebuilt ocp has {rebuilt_size} decision variables but the pickled solution has "
                f"{payload['vector_size']}. The saving and loading sides build a different ocp."
            )

        if add_constraint_plots:
            ocp.add_plot_penalty(CostType.CONSTRAINTS)  # must be registered on the ocp *before* graphs()
        sol.ocp = ocp

        if rebuild_q:
            payload["q"] = ExampleUtils._full_q_from_solution(ocp, sol)

        return sol, payload

    @staticmethod
    def _parameters_vector(sol) -> np.ndarray:
        """The casadi model functions all take the (possibly empty) parameter vector as their last argument."""
        params = sol.parameters
        if not isinstance(params, dict):
            return np.asarray(params, dtype=float).reshape(-1, 1)
        if not params:
            return np.zeros((0, 1))
        return np.concatenate([np.asarray(value, dtype=float).reshape(-1, 1) for value in params.values()])

    @staticmethod
    def _as_phase_list(data, n_phases: int) -> list:
        """decision_*() collapses the outer list when the ocp has a single phase; undo that."""
        return data if isinstance(data, list) else [data] * n_phases

    @staticmethod
    def _shooting_node_indices(n_columns: int, n_shooting: int) -> np.ndarray:
        """
        Column indices of the shooting nodes inside a merged decision-state block.

        With COLLOCATION the block holds `n_shooting * (polynomial_degree + 2) + 1` columns and only every
        (polynomial_degree + 2)-th one is a shooting node. With a single-step integrator the block already
        is the ns + 1 shooting nodes and the stride is 1.
        """
        stride, remainder = divmod(n_columns - 1, n_shooting)
        if remainder or stride < 1:
            raise RuntimeError(f"Cannot map {n_columns} decision-state columns onto {n_shooting} shooting intervals.")
        return np.arange(n_shooting + 1) * stride

    @staticmethod
    def _control_names(model, control_keys: list, phase_controls: dict) -> list:
        """
        Row labels for the stacked control vector, e.g. ['tendons_thumb_t', ..., 'non_tendon_tau_thumb_proxy_RotY'].

        The naming mirrors what ConfigureVariables uses when it declares the controls; anything that cannot
        be matched to a name of the model falls back to the row number.
        """
        dof_names = list(model.name_dofs)
        names = []
        for key in control_keys:
            n_rows = np.asarray(phase_controls[key]).shape[0]
            if key == "tendons":
                elements = list(getattr(model, "tendon_names", ()))
            elif key == "non_tendon_tau":
                elements = [dof_names[idx] for idx in model.non_tendon_tau_indices]
            elif n_rows == len(dof_names):  # tau, qddot, ... on every dof
                elements = dof_names
            elif n_rows == len(getattr(model, "muscle_names", ())):
                elements = list(model.muscle_names)
            else:
                elements = []
            if len(elements) != n_rows:
                elements = [str(i) for i in range(n_rows)]
            names += [f"{key}_{element}" for element in elements]
        return names

    @staticmethod
    def _q_qdot_at_nodes(model, phase_states: dict, columns: np.ndarray) -> tuple:
        """The full-dof (q, qdot) at the given columns, rebuilt from the independent ones if holonomic."""
        if "q_u" in phase_states:
            q_u = np.asarray(phase_states["q_u"], dtype=float)[:, columns]
            qdot_u = np.asarray(phase_states["qdot_u"], dtype=float)[:, columns]
            q_v_init = getattr(model, "q_v_init_guess", None)
            q_v_init = None if q_v_init is None else np.array(q_v_init, dtype=float)
            q = np.asarray(model.compute_q_from_u_iterative(q_u, q_v_init), dtype=float)
            compute_qdot = model.compute_qdot()
            qdot = np.concatenate(
                [np.asarray(compute_qdot(q[:, i], qdot_u[:, i]), dtype=float) for i in range(q.shape[1])], axis=1
            )
            return q, qdot

        q = np.asarray(phase_states["q"], dtype=float)[:, columns]
        if "qdot" in phase_states:
            qdot = np.asarray(phase_states["qdot"], dtype=float)[:, columns]
        else:
            qdot = np.zeros_like(q)
        return q, qdot

    @staticmethod
    def control_data_from_solution(ocp, sol) -> dict:
        """
        Controls and total tendon lengths at every shooting node, phase after phase.

        The lengths are the *total* length of each tendon (origin -> routing points -> insertion), not the
        per-section lengths. They are evaluated on the full-dof (q, qdot), rebuilt from the independent
        (q_u, qdot_u) when the transcription is holonomic. A model without tendon simply yields empty
        `tendon_names` and a (0 x nb_nodes) `tendon_lengths`.

        The last shooting node of a phase carries no control, so the control columns are padded with NaN
        there, which keeps `tendon_lengths[:, i]` and `u[:, i]` on the very same node grid.
        """
        from bioptim import SolutionMerge, TimeAlignment

        models = [nlp.model for nlp in ocp.nlp]
        n_phases = len(models)
        params = ExampleUtils._parameters_vector(sol)

        states = ExampleUtils._as_phase_list(sol.decision_states(to_merge=SolutionMerge.NODES), n_phases)
        controls = ExampleUtils._as_phase_list(sol.decision_controls(to_merge=SolutionMerge.NODES), n_phases)
        times = ExampleUtils._as_phase_list(
            sol.decision_time(to_merge=SolutionMerge.NODES, time_alignment=TimeAlignment.STATES), n_phases
        )

        control_keys = list(controls[0].keys())
        control_names = ExampleUtils._control_names(models[0], control_keys, controls[0])
        control_rows = {}  # where each control key sits in the stacked u
        first_row = 0
        for key in control_keys:
            n_rows = np.asarray(controls[0][key]).shape[0]
            control_rows[key] = slice(first_row, first_row + n_rows)
            first_row += n_rows

        lengths_per_phase, u_per_phase, time_per_phase, phase_idx, node_idx = [], [], [], [], []
        for i_phase, model in enumerate(models):
            phase_states = states[i_phase]
            n_shooting = ocp.nlp[i_phase].ns
            first_state = phase_states["q_u" if "q_u" in phase_states else "q"]
            columns = ExampleUtils._shooting_node_indices(np.asarray(first_state).shape[1], n_shooting)

            q, qdot = ExampleUtils._q_qdot_at_nodes(model, phase_states, columns)

            if getattr(model, "nb_tendons", 0):
                tendon_lengths = model.tendon_lengths()
                lengths_per_phase.append(
                    np.concatenate(
                        [
                            np.asarray(tendon_lengths(q[:, i], qdot[:, i], params), dtype=float)
                            for i in range(q.shape[1])
                        ],
                        axis=1,
                    )
                )
            else:
                lengths_per_phase.append(np.zeros((0, columns.size)))

            phase_controls = controls[i_phase]
            if list(phase_controls.keys()) != control_keys:
                raise RuntimeError(f"Phase {i_phase} does not declare the same controls as phase 0.")
            u = np.concatenate([np.asarray(phase_controls[key], dtype=float) for key in control_keys], axis=0)
            if u.shape[1] < columns.size:  # no control on the last shooting node
                u = np.concatenate([u, np.full((u.shape[0], columns.size - u.shape[1]), np.nan)], axis=1)
            u_per_phase.append(u[:, : columns.size])

            time_per_phase.append(np.asarray(times[i_phase], dtype=float).reshape(-1)[columns])
            phase_idx.append(np.full(columns.size, i_phase, dtype=int))
            node_idx.append(np.arange(columns.size, dtype=int))

        u = np.concatenate(u_per_phase, axis=1)
        return {
            "tendon_names": np.asarray(getattr(models[0], "tendon_names", ()), dtype=object),
            "tendon_lengths": np.concatenate(lengths_per_phase, axis=1),
            "u": u,
            "u_names": np.asarray(control_names, dtype=object),
            "time": np.concatenate(time_per_phase),
            "phase_index": np.concatenate(phase_idx),
            "node_index": np.concatenate(node_idx),
            "controls_per_key": {key: u[rows, :] for key, rows in control_rows.items()},
        }

    @staticmethod
    def save_control_data(ocp, sol, path: str | Path, data: dict = None) -> Path:
        """
        Write the controls and the tendon lengths of every shooting node to a .npz archive.

        The archive holds `tendon_lengths` (nb_tendons x nb_nodes), `u` (nb_controls x nb_nodes) plus one
        `u_<key>` array per control key, `time`, `phase_index`, `node_index` and the matching name arrays.

        Parameters
        ----------
        ocp: OptimalControlProgram
            The ocp of the solution; `sol.ocp` if the solution was just solved, the rebuilt one after a load
        sol: Solution
            The solution to extract from
        path: str | Path
            The .npz file to write; the suffix is appended if missing
        data: dict
            An already extracted `control_data_from_solution` dictionary, to avoid computing it twice

        Returns
        -------
        The path the archive was written to
        """
        if data is None:
            data = ExampleUtils.control_data_from_solution(ocp, sol)

        arrays = {
            "time": data["time"],
            "phase_index": data["phase_index"],
            "node_index": data["node_index"],
            "tendon_lengths": data["tendon_lengths"],
            "tendon_names": np.asarray([str(name) for name in data["tendon_names"]]),
            "u": data["u"],
            "u_names": np.asarray([str(name) for name in data["u_names"]]),
        }
        for key, value in data["controls_per_key"].items():
            arrays[f"u_{key}"] = value

        path = Path(path)
        if path.suffix.lower() != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

        print(f"\nSaved control data to {path}")
        print(f"  tendon_lengths {arrays['tendon_lengths'].shape} for {[str(name) for name in arrays['tendon_names']]}")
        print(f"  u {arrays['u'].shape} for {[str(name) for name in arrays['u_names']]}")
        print(f"  time {arrays['time'].shape} (the last node of each phase carries no control -> NaN in u)")

        return path


class IterationsControllerCallback(OnlineCallbackAbstract):
    """
    Ipopt iteration callback that gives control over when the solve ends.

    - First Ctrl+C: Stop now and return current state as is (i.e., non-converged solution)
    - Second Ctrl+C: Program exit, as normal Ctrl+C
    - Every `budget` iterations: asks whether to keep going.
        On enter, "y" or "yes, run `default_extension` more iterations and ask again.
        On "n" or "no", stop now and return current state as is.
        When typing a number, run that many more iterations.
    - Pressing `animate_key` (Enter is needed as well outside of Windows): opens a bioviz window on the
        iterate IPOPT is currently at, in its own process, and the solve carries on.
    """

    def __init__(self, ocp, budget=1000, default_extension=500, animate_key="q", animate_stride=1):
        super().__init__(ocp)
        self.budget = budget
        self.next_stop = budget
        self.default_extension = default_extension
        self.animate_key = animate_key.lower() if animate_key else None
        self.animate_stride = max(1, int(animate_stride))
        self.n_iter = 0
        self._stop = False
        self._keys = _KeyPoller()
        self._viewers = []
        signal.signal(signal.SIGINT, self._on_sigint)

        if self.animate_key and self._keys.enabled:
            enter = " then Enter" if self._keys.needs_enter else ""
            print(f">>> press '{self.animate_key}'{enter} at any time to animate the current iterate")

    def _on_sigint(self, signum, frame):
        if self._stop:
            print("\n>>> interrupted twice, exiting")
            sys.exit(130)
        print("\n>>> stopping, IPOPT will return the current iterate")
        self._stop = True

    def _ask_for_more_iterations(self) -> int:
        try:
            answer = input(f"{self.n_iter} iterations reached - how many more? [{self.default_extension}/n] ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            # Ctrl+C at the prompt, or no console to ask on
            return 0

        if answer in ("", "y", "yes"):
            return self.default_extension

        try:
            extra = int(answer)
        except ValueError:
            return 0
        return extra if extra > 0 else 0

    def _handle_keys(self, arg) -> None:
        """Act on whatever was typed since the previous iteration; anything unrecognised is dropped."""
        if not self.animate_key:
            return
        if self.animate_key not in self._keys.poll():
            return
        self._animate_current_iterate(arg)

    def _animate_current_iterate(self, arg) -> None:
        """
        Open a viewer on the iterate IPOPT just handed us, then let it carry on.

        `arg[0]` is the (scaled) optimization vector, which `Solution` unscales and unstacks without
        touching the ocp, so reading it mid-solve is harmless. Rebuilding the trajectory does happen here,
        in the solver thread, and on a holonomic ocp that is one Newton solve per column, so a press costs
        the solve a short pause; only the animation itself is offloaded. The iterate is generally not
        feasible yet, so the dependent coordinates are whatever the reconstruction converges to, and the
        constraints they are supposed to satisfy are visibly violated. That is the point of looking.
        """
        from casadi import DM

        from bioptim.optimization.solution.solution import Solution

        model_path = getattr(self.ocp.nlp[0].model, "path", None)
        if not isinstance(model_path, str):
            print(">>> the model does not expose a single .bioMod path, animation disabled")
            self.animate_key = None
            return

        try:
            sol = Solution.from_vector(self.ocp, DM(arg[0]))
            q = ExampleUtils._full_q_from_solution(self.ocp, sol)
        except Exception as error:
            # A Newton solve that does not converge on a wildly infeasible iterate must not take the
            # optimization down with it; the next press will hit a better iterate.
            print(f">>> could not rebuild the current iterate: {type(error).__name__}: {error}")
            return

        if q is None:
            print(">>> the ocp declares neither q nor q_u, animation disabled")
            self.animate_key = None
            return

        viewer = ExampleUtils.animate_q(model_path, q, stride=self.animate_stride)
        if viewer is not None:
            self._viewers = [each for each in self._viewers if each.poll() is None] + [viewer]
            print(f">>> iteration {self.n_iter}: animating in pid {viewer.pid}, the solve keeps running")

    def close(self):
        pass

    def eval(self, arg, enforce=False):
        if enforce:
            return [0]
        self.n_iter += 1
        self._handle_keys(arg)
        if self._stop:
            return [1]
        if self.n_iter >= self.next_stop:
            extra = self._ask_for_more_iterations()
            if extra == 0:
                return [1]
            self.next_stop = self.n_iter + extra
        return [0]
