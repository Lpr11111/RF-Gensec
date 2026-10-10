"""Protect the method-only package and its refactored configuration paths."""
import ast
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_only_rf_gensec_is_registered():
    from rift.cbv.planning import CBV_POLICY_LIST
    assert set(CBV_POLICY_LIST)=={'rlft_rf_gensec'}
    for removed in ['rl','fine_tuner/sft','fine_tuner/rlft/rift_pluto','fine_tuner/rlft/ppo_pluto']:
        assert not (ROOT/'rift/cbv/planning'/removed).exists()


def test_main_configuration_resolves_after_refactor():
    cfg=yaml.safe_load((ROOT/'rift/cbv/planning/config/rlft_rf_gensec.yaml').read_text())
    caller=ROOT/'rift/cbv/planning/fine_tuner/rlft/rlft_rf_gensec/base'
    config_dir=(caller/cfg['rlft_config_path']).resolve()
    training=yaml.safe_load((config_dir/cfg['rlft_config_name']).read_text())
    for entry in training['defaults']:
        for group,name in entry.items():
            assert (config_dir/group/(name+'.yaml')).is_file()
    data=yaml.safe_load((config_dir/'datamodule/rf_gensec_datamodule.yaml').read_text())
    assert data['type']=='rlft-rf-gensec'


def test_refactored_internal_imports_resolve():
    folder=ROOT/'rift/cbv/planning/fine_tuner/rlft/rlft_rf_gensec'
    for path in folder.rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module and ('rlft_rf_gensec.base.' in node.module or node.module=='rift.cbv.planning.pluto.backbone'):
                imported=ROOT.joinpath(*node.module.split('.')).with_suffix('.py')
                assert imported.is_file(), (path,node.module)
                classes={n.name for n in ast.parse(imported.read_text()).body if isinstance(n,ast.ClassDef)}
                assert all(alias.name in classes for alias in node.names)
