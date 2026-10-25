from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from atst_tools.calculators import factory
from atst_tools.calculators.constant_potential import _backend_log_paths, _backend_scf_converged
from atst_tools.utils.config_schema import ATSTConfig
from atst_tools.workflows.md import _ScfValidatedCalculator


class FakeProfile:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeAbacus:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeSocketIO:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = 0
        FakeSocketIO.instances.append(self)

    def close(self):
        self.closed += 1


@pytest.fixture
def fakes(monkeypatch):
    FakeSocketIO.instances = []
    monkeypatch.setattr(factory, "ATSTAbacusProfile", FakeProfile)
    monkeypatch.setattr(factory, "Abacus", FakeAbacus)
    monkeypatch.setattr(factory, "AbacusSocketIO", FakeSocketIO)
    monkeypatch.setattr(factory, "_SOCKET_CALCULATORS", [])


def _config(socket, workflow="relax", **calculation):
    return {
        "calculation": {"type": workflow, **calculation},
        "calculator": {
            "name": "abacus",
            "abacus": {
                "mpi": 2,
                "socket": socket,
                "parameters": {"calculation": "scf", "cal_force": 1, "ecutwfc": 50},
            },
        },
    }


def test_socket_disabled_builds_fileio_calculator(fakes):
    calc = factory.CalculatorFactory.get_calculator("abacus", _config(False), directory="run")

    assert isinstance(calc, FakeAbacus)
    assert "socket" not in calc.kwargs


def test_socket_true_builds_socketio_with_unique_unix_names(fakes):
    first = factory.CalculatorFactory.get_calculator("abacus", _config(True), directory="a")
    second = factory.CalculatorFactory.get_calculator("abacus", _config(True), directory="b")

    assert isinstance(first, FakeSocketIO)
    assert first.kwargs["inp"] == {"calculation": "scf", "cal_force": 1}
    assert first.kwargs["ecutwfc"] == 50
    assert "port" not in first.kwargs
    assert first.kwargs["unixsocket"].startswith("atst_")
    assert first.kwargs["unixsocket"] != second.kwargs["unixsocket"]
    assert first.kwargs["profile"].kwargs["command"] == "mpirun -np 2 abacus"


def test_socket_mapping_passes_port_and_timeout(fakes):
    calc = factory.CalculatorFactory.get_calculator(
        "abacus", _config({"port": 31500, "timeout": 30.0}), directory="run"
    )

    assert calc.kwargs["port"] == 31500
    assert calc.kwargs["timeout"] == 30.0
    assert "unixsocket" not in calc.kwargs


def test_socket_mapping_disabled_falls_back_to_fileio(fakes):
    calc = factory.CalculatorFactory.get_calculator(
        "abacus", _config({"enabled": False}), directory="run"
    )

    assert isinstance(calc, FakeAbacus)


def test_close_socket_calculators_closes_each_once(fakes):
    factory.CalculatorFactory.get_calculator("abacus", _config(True), directory="run")

    factory.close_socket_calculators()
    factory.close_socket_calculators()

    assert [calc.closed for calc in FakeSocketIO.instances] == [1]


@pytest.mark.parametrize("workflow", ["neb", "autoneb", "irc", "dmf"])
def test_socket_rejects_multi_calculator_workflows(fakes, workflow):
    with pytest.raises(ValueError, match="calculator.abacus.socket is supported only"):
        factory.CalculatorFactory.get_calculator("abacus", _config(True, workflow), directory="run")


def test_socket_rejects_native_md_driver(fakes):
    with pytest.raises(ValueError, match="driver=ase"):
        factory.CalculatorFactory.get_calculator(
            "abacus", _config(True, "md", driver="abacus_native"), directory="run"
        )


def test_socket_rejects_port_and_unixsocket_together(fakes):
    with pytest.raises(ValueError, match="either port or unixsocket"):
        factory.CalculatorFactory.get_calculator(
            "abacus", _config({"port": 1, "unixsocket": "x"}), directory="run"
        )


def _schema_config(socket, workflow="relax", **calculation):
    calc = {"type": workflow, "init_structure": "init.stru", **calculation}
    return {
        "calculation": calc,
        "calculator": {"name": "abacus", "abacus": {"socket": socket}},
    }


def test_schema_accepts_socket_bool_and_mapping():
    assert ATSTConfig.model_validate(_schema_config(True)).calculator.abacus.socket_enabled()
    mapped = ATSTConfig.model_validate(_schema_config({"unixsocket": "h2au", "timeout": 60}))
    assert mapped.calculator.abacus.socket.unixsocket == "h2au"
    assert not ATSTConfig.model_validate(_schema_config(False)).calculator.abacus.socket_enabled()


def test_schema_rejects_socket_for_unsupported_workflow():
    config = _schema_config(True, "vibration")
    ATSTConfig.model_validate(config)
    config["calculation"] = {"type": "irc", "init_structure": "ts.stru"}
    with pytest.raises(ValueError, match="calculator.abacus.socket is supported only"):
        ATSTConfig.model_validate(config)


def test_schema_rejects_unknown_socket_key():
    with pytest.raises(ValueError):
        ATSTConfig.model_validate(_schema_config({"host": "node1"}))


def test_socket_rejects_variable_cell_md_ensemble(fakes):
    with pytest.raises(ValueError, match="fixed-cell md ensemble"):
        factory.CalculatorFactory.get_calculator(
            "abacus", _config(True, "md", ensemble="npt"), directory="run"
        )


def test_socket_accepts_fixed_cell_md(fakes):
    calc = factory.CalculatorFactory.get_calculator(
        "abacus", _config(True, "md", ensemble="nve"), directory="run"
    )

    assert isinstance(calc, FakeSocketIO)


def test_schema_rejects_socket_for_npt_md():
    config = _schema_config(True, "md", ensemble="nvt", algorithm="bussi")
    ATSTConfig.model_validate(config)
    config["calculation"].update(ensemble="npt", algorithm="nptberendsen")
    with pytest.raises(ValueError, match="fixed-cell md ensemble"):
        ATSTConfig.model_validate(config)


class _SocketLikeBackend(Calculator):
    """Socket-shaped backend: SCF state is an attribute, ABACUS runs in ``abacus``."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, converged, abacus_directory="."):
        super().__init__()
        self.converged = converged
        self.abacus = SimpleNamespace(directory=abacus_directory, parameters={})

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.last_scf_converged = self.converged
        self.results = {"energy": -1.0, "forces": np.zeros((len(atoms), 3))}


@pytest.mark.parametrize("converged", [True, False])
def test_scf_evidence_reads_socket_attribute(converged):
    backend = _SocketLikeBackend(converged)
    backend.last_scf_converged = converged

    assert _backend_scf_converged(backend, {"energy": -1.0}) is converged


def test_scf_log_paths_follow_wrapped_abacus_directory(tmp_path):
    backend = _SocketLikeBackend(None, abacus_directory=str(tmp_path / "md_run"))

    assert _backend_log_paths(backend) == (tmp_path / "md_run" / "OUT.ABACUS" / "running_scf.log",)


def test_md_scf_guard_accepts_converged_socket_and_rejects_unconverged():
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]], cell=[5, 5, 5], pbc=True)

    atoms.calc = _ScfValidatedCalculator(_SocketLikeBackend(True))
    assert atoms.get_forces().shape == (2, 3)

    atoms.calc = _ScfValidatedCalculator(_SocketLikeBackend(False))
    with pytest.raises(RuntimeError, match="SCF not converged"):
        atoms.get_forces()


def test_socket_accepts_ccqn(fakes):
    calc = factory.CalculatorFactory.get_calculator("abacus", _config(True, "ccqn"), directory="run")

    assert isinstance(calc, FakeSocketIO)


def test_schema_accepts_socket_for_ccqn():
    config = _schema_config(True, "ccqn", reactive_bonds="1-2")

    assert ATSTConfig.model_validate(config).calculator.abacus.socket_enabled()
