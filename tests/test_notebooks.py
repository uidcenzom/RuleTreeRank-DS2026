"""Guards for the notebook bugs fixed in this repo. Run: python tests/test_notebooks.py

These are content checks, not execution checks: the retrain notebooks cannot run
end to end until the model-selection stage exists, so the invariants they used to
violate are asserted statically instead.
"""
import ast
import inspect
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def code_source(nb_path):
    nb = json.loads(nb_path.read_text())
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n".join(
        "\n".join("" if re.match(r"\s*[%!]", l) else l for l in "".join(c["source"]).split("\n"))
        for c in cells
    )


def test_retrain_exports_a_defined_name():
    """The final to_json cell used to reference `results`, which nothing defined."""
    for nb in sorted(ROOT.glob("experiments/query_based/*/retrain.ipynb")):
        tree = ast.parse(code_source(nb))
        assigned = {n.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
                    for n in ast.walk(node.targets[0]) if isinstance(n, ast.Name)}
        used_in_export = {
            n.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "to_json"
            for n in ast.walk(node) if isinstance(n, ast.Name)
        }
        missing = used_in_export - assigned - {"pd", "base_result"}
        assert not missing, f"{nb.parent.name}: exports undefined name(s) {missing}"
    print("ok  retrain notebooks export only names they define")


def test_retrain_result_schema_matches_listnet():
    """<DS>_result.json concatenates our table with ListNet's; columns must agree."""
    from listnet.listnet_model_selection import make_listnet_result
    expected = '["model", "fold", "qxm", "k", "mean", "std", "median"]'
    assert expected in inspect.getsource(make_listnet_result)
    for nb in sorted(ROOT.glob("experiments/query_based/*/retrain.ipynb")):
        assert expected in code_source(nb), f"{nb.parent.name}: aggregation schema drifted"
    print("ok  retrain aggregation schema matches make_listnet_result")


def test_clustering_writes_where_retrain_reads():
    """4 of 7 save paths used to point at directories no consumer reads."""
    src = code_source(ROOT / "clustering/query_similarity.ipynb")
    tree = ast.parse(src)
    out_dirs = {node.args[2].value
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "sweep"}
    assert len(out_dirs) >= 7, f"expected one sweep per dataset, got {sorted(out_dirs)}"
    for d in out_dirs:
        assert (ROOT / "experiments/query_based" / d).is_dir(), \
            f"clustering writes to {d}/, which is not an experiment directory"
    saved = re.search(r'save_dict_to_json\(sim_queries, (.+?)\)', src).group(1)
    assert 'results/query_similarity.json' in saved, saved
    print(f"ok  clustering writes into {len(out_dirs)} real experiment dirs")


def test_min_exp_extract_returns_lower_bound():
    """It used to return split(" ")[2], i.e. the interval's upper bound."""
    src = code_source(ROOT / "datasets/FindHR/makeFitness.ipynb")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "min_exp_extract")
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<nb>", "exec"), ns)
    for interval, lower in [("1 - 2 years", 1), ("2 - 3 years", 2), ("5 - 10 years", 5)]:
        got = ns["min_exp_extract"](interval)
        assert got == lower, f"{interval!r} -> {got}, expected {lower}"
    print("ok  min_exp_extract returns the lower bound")


def test_findhr_score_is_seeded_and_name_aligned():
    src = code_source(ROOT / "datasets/FindHR/makeFitness.ipynb")
    assert "default_rng(" in src, "ground-truth noise must be seeded to be reproducible"
    assert ".dot(" not in src, "positional .dot() silently mis-weights on column reorder"
    print("ok  FindHR score is seeded and computed by column name")


def test_ablation_hatches_mark_euclidean_variants():
    """The Mix pair was hatched backwards, contradicting the figure caption."""
    src = code_source(ROOT / "experiments/overall_plots/overall_results.ipynb")
    tree = ast.parse(src)
    consts = {t.id: node.value
              for node in ast.walk(tree) if isinstance(node, ast.Assign)
              for t in node.targets if isinstance(t, ast.Name)}
    models = [k.value for k in consts["abl_models_rtr"].keys]
    hatches = ast.literal_eval(consts["hatches"])
    assert len(models) == len(hatches), (models, hatches)
    for m, h in zip(models, hatches):
        assert m.endswith("_eucl") == bool(h), f"{m}: hatched={bool(h)} but caption says Euclidean-only"
    print("ok  ablation hatches mark exactly the Euclidean variants")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} checks passed")
