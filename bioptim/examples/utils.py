import platform
import signal
import sys
from pathlib import Path

from bioptim.gui.online_callback_abstract import OnlineCallbackAbstract


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

class IterationsControllerCallback(OnlineCallbackAbstract):
    """
    Ipopt iteration callback that gives control over when the solve ends.

    - First Ctrl+C: Stop now and return current state as is (i.e., non-converged solution)
    - Second Ctrl+C: Program exit, as normal Ctrl+C
    - Every `budget` iterations: asks whether to keep going.
        On enter, "y" or "yes, run `default_extension` more iterations and ask again.
        On "n" or "no", stop now and return current state as is.
        When typing a number, run that many more iterations.
    """

    def __init__(self, ocp, budget=1000, default_extension=500):
        super().__init__(ocp)
        self.budget = budget
        self.next_stop = budget
        self.default_extension = default_extension
        self.n_iter = 0
        self._stop = False
        signal.signal(signal.SIGINT, self._on_sigint)

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

    def close(self):
        pass

    def eval(self, arg, enforce=False):
        if enforce:
            return [0]
        self.n_iter += 1
        if self._stop:
            return [1]
        if self.n_iter >= self.next_stop:
            extra = self._ask_for_more_iterations()
            if extra == 0:
                return [1]
            self.next_stop = self.n_iter + extra
        return [0]
