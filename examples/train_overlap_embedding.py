"""Small differentiable overlap-metric learner, not attention and not a benchmark.

A shared encoder learns state geometry using pairwise fidelity targets. Both
executors train the same model from the same initialization and data. PennyLane
QNodes participate in the second training run, not only a post-hoc check.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pennylane as qml
from pennylane import numpy as anp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qvm import Program, QVM, __version__


def make_program():
    p = Program('overlap_metric_embedding')
    p.input('angles', 'angles')
    p.emit('states', 'encode_ryrz', 'angles', role='state', result_type='ket')
    p.emit('collection', 'as_collection', 'states', result_type='state_collection')
    p.emit('gram', 'gram_matrix', 'collection', result_type='gram')
    p.emit('similarity', 'absolute_square', 'gram', result_type='real')
    return p.returns('similarity')


def data(rng, n):
    labels = np.arange(n) % 2
    inputs = rng.normal(scale=.22, size=(n, 2))
    inputs[:, 0] += 2*labels-1
    targets = (labels[:, None] == labels[None, :]).astype(float)
    return inputs, labels, targets


def run_demo(steps=40):
    vm, program = QVM(), make_program()
    rng = np.random.default_rng(404)
    x, labels, target = data(rng, 16)
    test_x, test_labels, test_target = data(rng, 16)
    initial = rng.normal(scale=.3, size=12)
    mask = 1-np.eye(len(x))

    def similarity(theta, inputs, backend):
        angles = anp.reshape(anp.dot(inputs, anp.reshape(theta[:8], (2, 4))) + theta[8:], (-1, 2, 2))
        return vm.run(program, {'angles': angles}, backend=backend)

    def loss(theta, inputs, targets, backend):
        return anp.sum(mask*(similarity(theta, inputs, backend)-targets)**2)/anp.sum(mask)

    rows, checkpoints, predictions = [], [], []
    for backend in ('numpy_overlap', 'pennylane_overlap'):
        theta = anp.array(initial, requires_grad=True)
        objective = lambda t: loss(t, x, target, backend)
        gradient = qml.grad(objective)
        history = [float(objective(theta))]
        for _ in range(steps):
            theta = theta - .3*gradient(theta)
            history.append(float(objective(theta)))
        checkpoints.append(np.asarray(theta))
        predictions.append(np.asarray(similarity(theta, test_x, backend)))
        rows.append({'backend': backend, 'steps': steps, 'initial_loss': history[0],
                     'final_loss': history[-1], 'heldout_pair_loss': float(loss(theta, test_x, test_target, backend)),
                     'history': history, 'finite_gradient': bool(np.isfinite(gradient(theta)).all())})
    # Frozen-parameter finite difference, independent of update trajectories.
    theta = anp.array(initial, requires_grad=True)
    direction = rng.normal(size=12)
    f = lambda t: loss(t, x, target, 'pennylane_overlap')
    eps = 1e-6
    fd = (f(theta+eps*direction)-f(theta-eps*direction))/(2*eps)
    derivative = np.dot(np.asarray(qml.grad(f)(theta)), direction)
    report = {'qvm_version': __version__, 'program_sha256': program.sha256,
              'parameter_count': 12, 'seed': 404, 'runs': rows,
              'max_checkpoint_difference': float(np.max(abs(checkpoints[0]-checkpoints[1]))),
              'max_prediction_difference': float(np.max(abs(predictions[0]-predictions[1]))),
              'finite_difference_error': float(abs(fd-derivative)),
              'torch_imported': 'torch' in sys.modules,
              'interpretation': 'Engineering smoke example only; no superiority, attention comparison, or quantum advantage claim.'}
    arrays = {'initial': initial, 'numpy_theta': checkpoints[0], 'pennylane_theta': checkpoints[1],
              'train_x': x, 'train_labels': labels, 'test_x': test_x, 'test_labels': test_labels,
              'numpy_predictions': predictions[0], 'pennylane_predictions': predictions[1]}
    return report, arrays


def main():
    report, arrays = run_demo()
    out = Path(__file__).resolve().parents[1] / 'artifacts'
    out.mkdir(exist_ok=True)
    np.savez_compressed(out/'overlap_learning.npz', **arrays)
    report['arrays_sha256'] = hashlib.sha256((out/'overlap_learning.npz').read_bytes()).hexdigest()
    (out/'overlap_learning.json').write_text(json.dumps(report, indent=2))
    compact = {**report, 'runs': [{k:v for k,v in row.items() if k != 'history'} for row in report['runs']]}
    print(json.dumps(compact, indent=2))
    assert all(row['final_loss'] < row['initial_loss'] and row['finite_gradient'] for row in report['runs'])
    assert report['max_checkpoint_difference'] < 1e-10
    assert report['finite_difference_error'] < 1e-7
    assert not report['torch_imported']


if __name__ == '__main__':
    main()
