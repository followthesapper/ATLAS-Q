
def test_period_head_training_script_imports():
    import importlib.util, sys, pathlib
    p = pathlib.Path("scripts/qih_period_head_train.py")
    spec = importlib.util.spec_from_file_location("qih_period_head_train", str(p))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert hasattr(mod, "train_period_head")
