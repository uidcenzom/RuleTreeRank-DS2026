import re
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union


class RankLibAlgorithm(Enum):
    MART = 0
    RankNet = 1
    RankBoost = 2
    AdaRank = 3
    CoordinateAscent = 4
    LambdaMART = 6
    ListNet = 7
    RandomForests = 8
    LinearRegression = 9


AlgorithmInput = Union[RankLibAlgorithm, int, str]
PathInput = Union[str, Path]
_ZERO_NDCG_SCORES = {
    "ndcg_training": 0.0,
    "ndcg_valid": 0.0,
    "ndcg_test": 0.0,
}
_LISTNET_RECOVERABLE_ERROR_MARKERS = (
    "RankLibError",
    "restoreBestModelOnValidation",
    "IndexOutOfBoundsException",
)


def zero_ndcg_scores() -> Dict[str, float]:
    return dict(_ZERO_NDCG_SCORES)


def is_recoverable_listnet_error(error: BaseException) -> bool:
    message = str(error)
    return any(marker in message for marker in _LISTNET_RECOVERABLE_ERROR_MARKERS)


class RankLibTrainer:
    """
    Wrapper minimale per allenare RankLib da Python e leggere le metriche NDCG.

    Esempio
    -------
    trainer = RankLibTrainer(
        algorithm="ListNet",
        metric="NDCG@10",
        epoch=50,
        lr=0.00001,
    )
    scores = trainer.fit_predict(
        train_file="datasets/MQ2008/train.txt",
        validation_file="datasets/MQ2008/vali.txt",
        test_file="datasets/MQ2008/test.txt",
    )
    """

    _METRIC_RE = re.compile(
        r"(?P<metric>NDCG(?:@\d+)?)\s+on\s+"
        r"(?P<split>training|validation|test)\s+data:\s+"
        r"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        re.IGNORECASE,
    )
    _PARAM_ALIASES = {
        "missing_zero": "missingZero",
        "linear_regression_l2": "L2",
        "l2": "L2",
        "learning_rate": "lr",
        "num_trees": "tree",
        "min_leaf_support": "mls",
        "early_stop": "estop",
    }

    def __init__(
        self,
        algorithm: AlgorithmInput,
        *,
        jar_path: Optional[PathInput] = None,
        metric: str = "NDCG@10",
        memory: str = "2G",
        java_bin: str = "java",
        model_path: Optional[PathInput] = None,
        keep_model: bool = False,
        timeout: Optional[int] = None,
        log_path: Optional[PathInput] = None,
        silent: bool = False,
        **ranklib_params: Any,
    ) -> None:
        """
        Parameters
        ----------
        algorithm:
            Algoritmo RankLib da usare. Accetta enum, id numerico o stringa
            (es. "ListNet", "LambdaMART", 7).
        jar_path:
            Path al jar di RankLib. Se omesso usa `other/listnet/ranklib.jar`.
        metric:
            Metrica NDCG da leggere, ad esempio "NDCG@10".
        memory:
            Heap Java massimo, ad esempio "2G" o "512M".
        model_path:
            Dove salvare il modello RankLib. Se omesso viene creato un file
            temporaneo.
        keep_model:
            Se `False` e `model_path` non e' stato fornito, il modello
            temporaneo viene eliminato a fine chiamata.
        timeout:
            Timeout in secondi per ogni processo Java.
        log_path:
            File opzionale in cui salvare stdout/stderr prodotti da RankLib.
        silent:
            Passa `-silent` a RankLib.
        **ranklib_params:
            Parametri RankLib aggiuntivi, senza trattino iniziale
            (es. `epoch=50`, `lr=0.00001`, `tree=200`, `leaf=10`).
        """
        self.algorithm = self._normalize_algorithm(algorithm)
        self.metric = metric.upper()
        if not self.metric.startswith("NDCG"):
            raise ValueError("Questo wrapper restituisce metriche NDCG: usa metric='NDCG' o 'NDCG@k'.")

        default_jar = Path(__file__).resolve().parent / "ranklib.jar"
        legacy_jar = Path(__file__).resolve().parents[1] / "other" / "listnet" / "ranklib.jar"
        if jar_path is None and not default_jar.exists() and legacy_jar.exists():
            default_jar = legacy_jar
        self.jar_path = Path(jar_path) if jar_path is not None else default_jar
        if not self.jar_path.exists():
            raise FileNotFoundError(f"RankLib jar non trovato: {self.jar_path}")

        self.memory = memory
        self.java_bin = java_bin
        self.model_path = Path(model_path) if model_path is not None else None
        self.keep_model = keep_model
        self.timeout = timeout
        self.log_path = Path(log_path) if log_path is not None else None
        self.silent = silent
        self.ranklib_params = dict(ranklib_params)

        self.last_model_path: Optional[Path] = None
        self.last_output: str = ""
        self.last_commands: List[List[str]] = []

    def fit_predict(
        self,
        train_file: PathInput,
        validation_file: PathInput,
        test_file: PathInput,
    ) -> Dict[str, float]:
        """
        Allena RankLib e restituisce le metriche NDCG sui tre split.

        I file devono essere gia' nel formato LTR/LETOR accettato da RankLib.
        """
        train_path = self._existing_file(train_file, "training")
        validation_path = self._existing_file(validation_file, "validation")
        test_path = self._existing_file(test_file, "test")
        self._evaluation_logs: List[str] = []

        if self.model_path is not None:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path = self.model_path
            temp_dir = None
        else:
            temp_dir = tempfile.TemporaryDirectory(prefix="ranklib_")
            model_path = Path(temp_dir.name) / "ranklib_model.txt"

        try:
            train_args = [
                "-train",
                train_path,
                "-ranker",
                self.algorithm,
                "-validate",
                validation_path,
                "-metric2t",
                self.metric,
                "-metric2T",
                self.metric,
                "-save",
                model_path,
            ]
            if self.silent:
                train_args.append("-silent")
            train_args.extend(self._params_to_args(self.ranklib_params))

            try:
                logs = [self._run_ranklib(train_args)]
                scores = {
                    "ndcg_training": self._evaluate_model(model_path, train_path),
                    "ndcg_valid": self._evaluate_model(model_path, validation_path),
                    "ndcg_test": self._evaluate_model(model_path, test_path),
                }
                logs.extend(self._evaluation_logs)
                self.last_model_path = model_path
                self.last_output = "\n".join(logs)
                self._write_log(self.last_output)
                return scores
            except RuntimeError as exc:
                self.last_output = str(exc)
                self._write_log(self.last_output)
                if self.algorithm == RankLibAlgorithm.ListNet.value and is_recoverable_listnet_error(exc):
                    return zero_ndcg_scores()
                raise
        finally:
            if temp_dir is not None and not self.keep_model:
                temp_dir.cleanup()
                self.last_model_path = None

    def _evaluate_model(self, model_path: Path, data_file: Path) -> float:
        output = self._run_ranklib(["-load", model_path, "-test", data_file, "-metric2T", self.metric])
        self._evaluation_logs.append(output)
        return self._extract_ndcg(output)

    def _run_ranklib(self, args: Iterable[Any]) -> str:
        cmd = [
            self.java_bin,
            f"-Xmx{self.memory}",
            "-jar",
            str(self.jar_path),
            *[str(arg) for arg in args],
        ]
        self.last_commands.append(cmd)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Runtime Java non trovato. Installa Java o configura `java_bin`.") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"RankLib ha superato il timeout di {self.timeout} secondi.") from exc

        output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode != 0:
            output = self._truncate_output(output)
            raise RuntimeError(
                "Esecuzione RankLib fallita.\n"
                f"Comando: {' '.join(cmd)}\n"
                f"Exit code: {result.returncode}\n"
                f"Output:\n{output}"
            )
        return output

    @staticmethod
    def _truncate_output(output: str, max_chars: int = 12000) -> str:
        if len(output) <= max_chars:
            return output
        return "... output RankLib troncato ...\n" + output[-max_chars:]

    def _write_log(self, output: str) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text(output, encoding="utf-8")

    def _extract_ndcg(self, output: str) -> float:
        matches = [
            match
            for match in self._METRIC_RE.finditer(output)
            if match.group("metric").upper() == self.metric
        ]
        if not matches:
            raise ValueError(
                f"Impossibile trovare '{self.metric}' nell'output RankLib.\n"
                f"Output ricevuto:\n{output}"
            )
        return float(matches[-1].group("value"))

    @classmethod
    def _normalize_algorithm(cls, algorithm: AlgorithmInput) -> int:
        if isinstance(algorithm, RankLibAlgorithm):
            return algorithm.value
        if isinstance(algorithm, int):
            known_ids = {item.value for item in RankLibAlgorithm}
            if algorithm in known_ids:
                return algorithm
            raise ValueError(f"Id algoritmo RankLib non supportato: {algorithm}")
        if isinstance(algorithm, str):
            normalized = re.sub(r"[\s_-]+", "", algorithm).lower()
            for item in RankLibAlgorithm:
                if re.sub(r"[\s_-]+", "", item.name).lower() == normalized:
                    return item.value
        allowed = ", ".join(item.name for item in RankLibAlgorithm)
        raise ValueError(f"Algoritmo RankLib non supportato: {algorithm}. Valori ammessi: {allowed}.")

    @staticmethod
    def _existing_file(path: PathInput, label: str) -> Path:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {label} non trovato: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Il path {label} non e' un file: {file_path}")
        return file_path

    @classmethod
    def _params_to_args(cls, params: Dict[str, Any]) -> List[str]:
        args: List[str] = []
        for key, value in params.items():
            if value is None or value is False:
                continue
            flag = cls._PARAM_ALIASES.get(key, key)
            args.append(f"-{flag}")
            if value is not True:
                args.append(str(value))
        return args
