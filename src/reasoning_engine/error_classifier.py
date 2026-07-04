import json
from pathlib import Path

# Maps error label → programming concept to teach
ERROR_TO_CONCEPT = {
    "syntax_error": "Python/Language Syntax Rules",
    "logical_error": "Program Logic & Control Flow",
    "type_error": "Data Types & Type Casting",
    "off_by_one": "Loop Indexing & Array Boundaries",
    "infinite_loop": "Loop Termination Conditions",
    "null_reference": "Null Safety & Object Initialization",
    "division_by_zero": "Input Validation & Edge Cases",
    "bad_practice": "Clean Code & Readability Principles",
    "structural_issue": "Function Design & Code Decomposition",
}


class ErrorClassifier:
    """Maps raw error labels to human-readable types and concepts."""

    def __init__(self, error_types_path: str = "data/labels/error_types.json"):
        with open(error_types_path, "r") as f:
            self.error_types = json.load(f)
        # Reverse map: index → name. The repository stores label groups as lists,
        # so we flatten them into a stable index-to-label mapping.
        flattened = []
        for category, specifics in self.error_types.items():
            flattened.extend(
                [f"{category}/{specific}" for specific in specifics]
            )
        self.idx_to_label = {idx: label for idx, label in enumerate(flattened)}

    def decode(self, binary_vector: list) -> list:
        """Convert a binary prediction vector to a list of error label strings."""
        labels = []
        for i, val in enumerate(binary_vector):
            if val == 1:
                label = self.idx_to_label[i]
                labels.append(label.split("/", 1)[0])
        return labels

    def get_concepts(self, error_labels: list) -> list:
        """Map error labels to the concepts they teach."""
        return [
            ERROR_TO_CONCEPT.get(label, "General Programming") for label in error_labels
        ]

    def get_primary_error(self, error_labels: list) -> str:
        """Return the most important error to address first."""
        priority = [
            "syntax_error",
            "type_error",
            "null_reference",
            "division_by_zero",
            "off_by_one",
            "infinite_loop",
            "logical_error",
            "structural_issue",
            "bad_practice",
        ]
        for p in priority:
            if p in error_labels:
                return p
        return error_labels[0] if error_labels else "unknown"
