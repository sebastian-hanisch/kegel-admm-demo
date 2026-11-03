"""Projektion auf den Exponentialkegel links vom Gitter der Wurzelsuche (x/y unter etwa -700): der Randpunkt (x, y, y e^(x/y)) ist exakt; Vorlage: kegel-projektion-demo."""

import numpy as np
import pytest

import kad_projection as P


@pytest.mark.parametrize("v,dist", [((-800.0, 1.0, -1.0), 1.0), ((-1500.0, 2.0, -0.5), 0.5), ((-3000.0, 1.0, -2.0), 2.0)])
def test_the_projection_left_of_the_grid_is_the_boundary_point(v, dist):
    V = np.array([v])
    p = P.proj_exp(V)
    assert np.linalg.norm(V - p) == pytest.approx(dist, abs=1e-12)          # Abstand = |z|: der Punkt (x, y, 0) liegt im Kegel
    assert P.exp_member(p, 1e-12)[0]
