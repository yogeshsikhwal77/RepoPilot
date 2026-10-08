from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional


@dataclass(frozen=True)
class TaskInstance:
    """Represents a single benchmark task instance (e.g., SWE-bench Lite)."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    hints_text: Optional[str] = ""
    test_patch: Optional[str] = ""
    patch: Optional[str] = ""  # Ground truth patch for eval reference


class TaskDatasetLoader:
    """Loads and filters task instances based on task_ids.txt."""

    def __init__(
        self,
        task_ids_path: Path | str = Path(__file__).parent / "task_ids.txt",
        dataset_path: Path | str = Path(__file__).parent / "swebench_lite_subset.jsonl",
    ) -> None:
        self.task_ids_path = Path(task_ids_path)
        self.dataset_path = Path(dataset_path)

    def load_task_ids(self) -> List[str]:
        """Reads the fixed, deduplicated list of task IDs."""
        if not self.task_ids_path.exists():
            raise FileNotFoundError(f"Task IDs file missing at: {self.task_ids_path}")

        task_ids: List[str] = []
        with open(self.task_ids_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    task_ids.append(line)
        return task_ids

    def load_tasks(self) -> List[TaskInstance]:
        """Loads task instances matching the IDs in task_ids.txt."""
        target_ids = set(self.load_task_ids())
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file missing at: {self.dataset_path}")

        tasks: List[TaskInstance] = []
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                iid = record.get("instance_id")
                if iid in target_ids:
                    tasks.append(
                        TaskInstance(
                            instance_id=iid,
                            repo=record.get("repo", ""),
                            base_commit=record.get("base_commit", ""),
                            problem_statement=record.get("problem_statement", ""),
                            hints_text=record.get("hints_text", ""),
                            test_patch=record.get("test_patch", ""),
                            patch=record.get("patch", ""),
                        )
                    )

        # Verify all target IDs were found
        found_ids = {t.instance_id for t in tasks}
        missing = target_ids - found_ids
        if missing:
            print(f"[Warning] {len(missing)} task IDs from task_ids.txt were not found in JSONL dataset.")

        return tasks

    def __iter__(self) -> Iterator[TaskInstance]:
        return iter(self.load_tasks())


# Example default list creator
def create_sample_task_ids_file(path: Path) -> None:
    """Generates a default task_ids.txt file if one does not exist."""
    sample_ids = [
        "astropy__astropy-12907",
        "django__django-11099",
        "django__django-11179",
        "matplotlib__matplotlib-23913",
        "mwaskom__seaborn-3069",
        "pallets__flask-4045",
        "psf__requests-2674",
        "pytest-dev__pytest-5221",
        "scikit-learn__scikit-learn-13439",
        "sympy__sympy-14774",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Fixed SWE-bench Lite Subset Task IDs\n")
        f.write("\n".join(sample_ids) + "\n")