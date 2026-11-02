"""Versioned, offline requirements -> evidence -> dimension/custom metrics.

Extension scores are reported beside the unchanged core profile. A development
plugin cannot silently replace a core score or become a formal study result.
"""
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import pkgutil

DIMENSIONS = ('Execution', 'Tooling', 'Context', 'Lifecycle', 'Observability', 'Verification', 'Governance')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def discover():
    import process_plugins
    registry = {}
    for item in sorted(pkgutil.iter_modules(process_plugins.__path__), key=lambda item: item.name):
        if item.name.startswith('_'):
            continue
        module = importlib.import_module('process_plugins.' + item.name)
        if hasattr(module, 'PLUGIN'):
            register(registry, module.PLUGIN)
    return registry


def discover_profiles():
    """Mandatory seven-dimensional contracts share the existing plugin namespace."""
    import process_plugins
    registry = {}
    for item in sorted(pkgutil.iter_modules(process_plugins.__path__), key=lambda item: item.name):
        if item.name.startswith('_'):
            continue
        profile = getattr(importlib.import_module('process_plugins.' + item.name), 'SEVEN_PROFILE', None)
        if profile is not None:
            register_profile(registry, profile)
    return registry


def register_profile(registry, profile):
    if not all(profile.get(k) for k in ('id', 'benchmark', 'version', 'dimensions')):
        raise ValueError('Seven-dimension profile identity missing')
    if set(profile['dimensions']) != set(DIMENSIONS):
        raise ValueError('Every admitted plugin must define all seven dimensions')
    for name, spec in profile['dimensions'].items():
        if spec.get('method') not in ('core', 'judge') or not spec.get('obligations'):
            raise ValueError('Every dimension needs an implementation and task obligations: ' + name)
        if not all(isinstance(v, str) and v.strip() for v in spec['obligations'].values()):
            raise ValueError('Empty obligation')
    if profile['benchmark'] in registry:
        raise ValueError('Duplicate seven-dimension benchmark profile')
    registry[profile['benchmark']] = profile


def register(registry, plugin):
    for key in ('id', 'version', 'benchmark', 'metrics', 'requirements', 'extract', 'evaluate', 'source_files'):
        if not plugin.get(key):
            raise ValueError('Process plugin missing ' + key)
    for key in ('requirements', 'extract', 'evaluate'):
        if not callable(plugin[key]):
            raise TypeError(key + ' must be callable')
    if plugin['id'] in registry:
        raise ValueError('Duplicate process plugin: ' + plugin['id'])
    for name, spec in plugin['metrics'].items():
        if spec.get('kind') not in ('dimension', 'custom'):
            raise ValueError('Invalid metric kind: ' + name)
        if spec['kind'] == 'dimension' and spec.get('dimension') not in DIMENSIONS:
            raise ValueError('Unknown core dimension: ' + name)
        if spec['kind'] == 'custom' and spec.get('dimension') is not None:
            raise ValueError('Custom metrics cannot silently contribute to core dimensions')
        if not spec.get('definition'):
            raise ValueError('Metric definition required: ' + name)
    registry[plugin['id']] = plugin


def identity(plugin):
    # Include helper modules explicitly. Persist this manifest before a formal
    # study; matching hashes alone do not certify independent validation.
    root = Path(__file__).resolve().parents[2]
    files = {}
    for value in plugin['source_files']:
        path = Path(value).resolve()
        name = str(path.relative_to(root)) if root in path.parents else str(path)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    contract = {'id': plugin['id'], 'version': plugin['version'], 'benchmark': plugin['benchmark'],
                'metrics': plugin['metrics'], 'files': files,
                'framework_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    return dict(contract, rule_sha256=digest(contract), validation_status='development',
                rules_frozen=False, formal_analysis_eligible=False)


class Bundle:
    """Read-only artifact access. No model, browser, mutation or runtime hooks."""
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.task = self.read('task.json')
        self.trace = self.read('trajectory.jsonl', lines=True)

    def read(self, name, lines=False, optional=False):
        path = (self.folder / name).resolve()
        if self.folder not in path.parents:
            raise ValueError('Artifact path escapes bundle')
        if optional and not path.exists():
            return None
        text = path.read_text()
        return [json.loads(row) for row in text.splitlines() if row.strip()] if lines else json.loads(text)


def metric(score, applicable, coverage, evidence, reason):
    return dict(score=score, applicable=applicable, coverage=coverage, evidence=evidence, reason=reason)


def _validate(row, evidence):
    if not isinstance(row, dict) or set(row) != {'score', 'applicable', 'coverage', 'evidence', 'reason'}:
        raise ValueError('Metric result must contain exactly the five result fields')
    if row.get('applicable') not in (True, False, None) or type(row.get('applicable')) not in (bool, type(None)):
        raise ValueError('applicable must be true, false or unknown/null')
    for key in ('score', 'coverage'):
        value = row.get(key)
        if key == 'score' and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(key + ' must be finite in [0,1]')
    if not isinstance(row.get('evidence'), list) or any(ref not in evidence for ref in row['evidence']):
        raise ValueError('Unknown evidence reference')
    if not isinstance(row.get('reason'), str) or not row['reason']:
        raise ValueError('Metric requires a reason')
    if row['score'] is not None and (row['applicable'] is not True or not row['evidence'] or row['coverage'] <= 0):
        raise ValueError('Numeric scores require applicability and sufficient cited evidence')
    if row['applicable'] is False and row['coverage'] != 0:
        raise ValueError('Inapplicable metrics have zero coverage')


def evaluate_bundle(folder, registry=None):
    bundle = Bundle(folder)
    registry = discover() if registry is None else registry
    outputs = []
    for plugin in registry.values():
        if plugin['benchmark'] != bundle.task.get('source_benchmark'):
            continue
        ident = identity(plugin)
        evidence, requirements, errors = {}, {}, []
        try:
            requirements = plugin['requirements'](copy.deepcopy(bundle.task))
            evidence = plugin['extract'](bundle, copy.deepcopy(requirements))
            if not isinstance(evidence, dict):
                raise ValueError('Extractor must return an evidence dictionary')
            rows = plugin['evaluate'](copy.deepcopy(requirements), copy.deepcopy(evidence))
            if set(rows) != set(plugin['metrics']):
                raise ValueError('Evaluator must return every declared metric exactly once')
            for row in rows.values():
                _validate(row, evidence)
        except Exception as error:
            errors = [{'stage': 'process_plugin', 'type': type(error).__name__, 'reason': str(error)[:250]}]
            rows = {name: metric(None, None, 0, [], 'Plugin evaluation failed; see errors')
                    for name in plugin['metrics']}
        output = {'plugin': ident, 'requirements': requirements, 'evidence': evidence, 'errors': errors, 'metrics': []}
        for name, row in rows.items():
            output['metrics'].append(dict(row, metric=name, **plugin['metrics'][name],
                                          plugin_id=plugin['id'], plugin_version=plugin['version'],
                                          rule_sha256=ident['rule_sha256'], formal_analysis_eligible=False))
        outputs.append(output)
    return {'task_id': bundle.task['task_id'], 'plugins': outputs}


def aggregate(agent_dir, benchmark, task_ids=None):
    root = Path(agent_dir)
    registry = discover()
    ids = sorted(set(task_ids)) if task_ids is not None else sorted(p.parent.name for p in root.glob('*/result.json'))
    if not any(p['benchmark'] == benchmark for p in registry.values()):
        return {'schema_version': '1.0', 'phase': 'offline_only', 'tasks': [],
                'dimension_metrics': [], 'custom_metrics': [],
                'validity': {'errors': [], 'status': 'no_process_extension', 'core_scores_changed': False}}
    tasks, errors, groups = [], [], {}
    for tid in ids:
        try:
            if (root / tid).resolve().parent != root.resolve():
                raise ValueError('Task bundle must be a direct child')
            bundle = evaluate_bundle(root / tid, registry)
            if json.loads((root / tid / 'task.json').read_text()).get('source_benchmark') != benchmark:
                raise ValueError('Bundle benchmark mismatch')
            tasks.append(bundle)
            for plugin in bundle['plugins']:
                errors.extend(dict(error, task_id=tid) for error in plugin['errors'])
                for row in plugin['metrics']:
                    key = (row['plugin_id'], row['plugin_version'], row['rule_sha256'], row['metric'])
                    groups.setdefault(key, []).append(row)
        except Exception as error:
            errors.append({'task_id': tid, 'type': type(error).__name__, 'reason': str(error)[:250]})
    metrics = []
    for key, rows in sorted(groups.items()):
        numbers = [r['score'] for r in rows if r['score'] is not None]
        applicable = sum(r['applicable'] is True for r in rows)
        unknown = sum(r['applicable'] is None for r in rows)
        missing = len(ids) - len(rows)
        denominator = applicable + unknown + missing
        metrics.append(dict(plugin_id=key[0], plugin_version=key[1], rule_sha256=key[2], metric=key[3],
                            kind=rows[0]['kind'], dimension=rows[0].get('dimension'),
                            score=sum(numbers)/len(numbers) if numbers else None,
                            coverage=sum(r['coverage'] for r in rows)/denominator if denominator else 0,
                            n_tasks=len(ids), n_applicable=applicable, n_unknown=unknown + missing,
                            n_scored=len(numbers), formal_analysis_eligible=False))
    return {'schema_version': '1.0', 'phase': 'offline_only', 'tasks': tasks,
            'dimension_metrics': [m for m in metrics if m['kind'] == 'dimension'],
            'custom_metrics': [m for m in metrics if m['kind'] == 'custom'],
            'validity': {'errors': errors, 'rules_frozen': False, 'formal_analysis_eligible': False,
                         'core_scores_changed': False,
                         'note': 'Development extensions. No implicit averaging into the core seven; custom metrics remain separate.'}}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent-dir', type=Path, required=True)
    parser.add_argument('--bench', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = aggregate(args.agent_dir, args.bench)
    # Exclusive creation protects earlier evaluation versions and source bundles.
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
