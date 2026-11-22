class ProjectOpsEnv:
    """模拟一次科研任务中的数据准备状态和实验配置。"""

    def __init__(self):
        self.dataset_ready = {"A": False, "B": False}
        self.configured_dataset = None
        self.experiment_runs: dict[str, str] = {}

    def prepare_dataset(self, dataset_id: str) -> dict:
        """将指定数据集标记为已准备。"""
        if dataset_id not in self.dataset_ready:
            return {"ok": False, "code": "unknown_dataset", "dataset_id": dataset_id}

        self.dataset_ready[dataset_id] = True
        return {"ok": True, "code": "dataset_ready", "dataset_id": dataset_id}

    def configure_experiment(self, dataset_id: str) -> dict:
        """使用一个已准备的数据集配置当前实验。"""
        if dataset_id not in self.dataset_ready:
            return {"ok": False, "code": "unknown_dataset", "dataset_id": dataset_id}

        if not self.dataset_ready[dataset_id]:
            return {"ok": False, "code": "dataset_not_ready", "dataset_id": dataset_id}

        self.configured_dataset = dataset_id
        return {"ok": True, "code": "experiment_configured", "dataset_id": dataset_id}

    def run_experiment(self) -> dict:
        """按当前配置执行一次模拟实验，记录本次运行使用的数据集。"""
        dataset_id = self.configured_dataset

        if dataset_id is None:
            return {"ok": False, "code": "experiment_not_configured"}

        run_id = f"run_{len(self.experiment_runs) + 1}"
        self.experiment_runs[run_id] = dataset_id

        return {
            "ok": True,
            "code": "experiment_completed",
            "run_id": run_id,
            "dataset_id": dataset_id,
        }
