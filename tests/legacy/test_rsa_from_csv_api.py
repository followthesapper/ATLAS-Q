
import csv
from pathlib import Path

def test_batch_factor_api_like():
    # Prepare a transient CSV similar to the example notebook
    csv_path = Path('tmp/tmp_rsa_targets.csv')
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N'])
        w.writerows([[221],[391],[899]])

    from atlas_q import QuantumClassicalHybrid
    h = QuantumClassicalHybrid(verbose=False)
    # Just verify we can factor each and the results multiply back
    for N in [221, 391, 899]:
        f = h.factor_number(N)
        assert f[0] * f[1] == N
