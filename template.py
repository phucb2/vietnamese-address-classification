from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

from address_classification import AddressCorrector, build_corrector, normalize_text
from beam_search import BeamSearchAddressCorrector, build_beam_corrector


class AddressSolution(ABC):
    """
    Simple interface for running interchangeable address solutions.
    """

    solution_id: str

    @abstractmethod
    def correct(self, raw_text: str) -> dict:
        raise NotImplementedError


class CorrectorSolution(AddressSolution):
    solution_id = "corrector"

    def __init__(self, corrector: AddressCorrector):
        self.corrector = corrector

    def correct(self, raw_text: str) -> dict:
        return self.corrector.correct(raw_text)


class CorrectorNoNumberSolution(AddressSolution):
    """
    Variant that removes standalone numbers before correction.
    Useful as a second benchmarkable strategy.
    """

    solution_id = "corrector_no_number"

    def __init__(self, corrector: AddressCorrector):
        self.corrector = corrector

    def correct(self, raw_text: str) -> dict:
        preprocessed = normalize_text(raw_text, remove_numbers=True)["spaced"]
        return self.corrector.correct(preprocessed)


class BeamSearchSolution(AddressSolution):
    solution_id = "beam_search"

    def __init__(self, corrector: BeamSearchAddressCorrector):
        self.corrector = corrector

    def correct(self, raw_text: str) -> dict:
        return self.corrector.correct(raw_text)


def available_solution_ids() -> List[str]:
    return ["corrector", "corrector_no_number", "beam_search"]


def build_solution(solution_id: str, data_dir: str) -> AddressSolution:
    base = build_corrector(data_dir)
    if solution_id == "corrector":
        return CorrectorSolution(base)
    if solution_id == "corrector_no_number":
        return CorrectorNoNumberSolution(base)
    if solution_id == "beam_search":
        return BeamSearchSolution(build_beam_corrector(data_dir))
    raise ValueError(
        f"Unknown solution '{solution_id}'. Available: {', '.join(available_solution_ids())}"
    )


def parse_solution_ids(raw: str) -> List[str]:
    values = [v.strip() for v in raw.split(",") if v.strip()]
    if not values:
        raise ValueError("No solution ids provided.")
    return values
