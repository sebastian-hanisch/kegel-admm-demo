"""Die vier Kugel-Programme (kad_forms): Handwerte, Gleichheit der Radien, Zulässigkeit des Starts."""

import math

import numpy as np
import pytest

import kegel_kern as K
import kad_barrier as B
import kad_forms as F
from kad_scenario import make_cloud


def test_two_antipodal_points_give_the_norm_of_one_point_in_every_cone():
    pts = np.array([[1.0, 2.0], [-1.0, -2.0]])
    assert K.solve_barrier(F.build_soc_ball(pts), tol=1e-11).value == pytest.approx(math.sqrt(5.0), abs=1e-7)
    assert K.solve_barrier(F.build_psd_ball(pts), tol=1e-11).value == pytest.approx(math.sqrt(5.0), abs=1e-7)
    assert B.solve(F.build_lp_ball(pts, 3.0), tol=1e-11).value == pytest.approx(9.0 ** (1.0 / 3.0), abs=1e-7)
    assert B.solve(F.build_lse_ball(pts, 1.0), tol=1e-11).value == pytest.approx(math.log(math.exp(1) + math.exp(2) + math.exp(-1) + math.exp(-2)), abs=1e-7)


def test_the_lorentz_and_the_psd_ball_have_the_same_radius_and_the_p_two_power_ball_too():
    pts = make_cloud(10, 3.0, 30, 100000).points
    soc = K.solve_barrier(F.build_soc_ball(pts), tol=1e-11).value
    assert K.solve_barrier(F.build_psd_ball(pts), tol=1e-11).value == pytest.approx(soc, abs=1e-7)
    assert B.solve(F.build_lp_ball(pts, 2.0), tol=1e-11).value == pytest.approx(soc, abs=1e-6)


def test_the_programs_have_the_documented_shapes_and_cones():
    pts = make_cloud(6, 2.0, 10, 1).points
    soc, psd, lp, lse = F.build_soc_ball(pts), F.build_psd_ball(pts), F.build_lp_ball(pts, 3.0), F.build_lse_ball(pts, 2.0)
    assert soc.A.shape == (18, 3) and K.barrier_parameter(soc.cones) == 12 and psd.A.shape == (36, 3) and K.barrier_parameter(psd.cones) == 18
    assert lp.A.shape == (3 * 12 + 6, 3 + 12) and lp.nu == 3 * 12 + 6 and lse.A.shape == (3 * 24 + 6, 3 + 24) and lse.nu == 3 * 24 + 6


def test_the_start_points_of_the_programs_are_interior():
    pts = make_cloud(6, 2.0, 10, 1).points
    for prob in (F.build_lp_ball(pts, 3.0), F.build_lse_ball(pts, 2.0)):
        assert B.barrier(prob, B.slack(prob, prob.x0), hess=False)[1] is not None
    for prob in (F.build_soc_ball(pts), F.build_psd_ball(pts)):
        assert K.check_solution(prob, prob.x0) == []
