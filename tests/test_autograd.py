import math

from my_microgpt.autograd import Value


def test_add_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    assert c.data == 5.0


def test_mul_forward():
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    assert c.data == 6.0


def test_backward_branching_graph():
    """L = a*b + a: a is used twice, so its gradient accumulates from both paths."""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    big_l = c + a
    big_l.backward()
    assert a.grad == 4.0  # dL/da = b + 1
    assert b.grad == 2.0  # dL/db = a


def test_pow_and_backward():
    a = Value(3.0)
    b = a**2
    b.backward()
    assert b.data == 9.0
    assert a.grad == 6.0  # d(a^2)/da = 2a = 6


def test_log_backward():
    a = Value(2.0)
    b = a.log()
    b.backward()
    assert abs(b.data - math.log(2.0)) < 1e-10
    assert abs(a.grad - 0.5) < 1e-10  # d(ln a)/da = 1/a


def test_exp_backward():
    a = Value(1.0)
    b = a.exp()
    b.backward()
    assert abs(b.data - math.e) < 1e-10
    assert abs(a.grad - math.e) < 1e-10  # d(e^a)/da = e^a


def test_relu_positive():
    a = Value(3.0)
    b = a.relu()
    b.backward()
    assert b.data == 3.0
    assert a.grad == 1.0


def test_relu_negative():
    a = Value(-2.0)
    b = a.relu()
    b.backward()
    assert b.data == 0.0
    assert a.grad == 0.0


def test_sub_and_div():
    a = Value(6.0)
    b = Value(2.0)
    c = a - b  # 4.0
    d = c / b  # 2.0
    assert d.data == 2.0


def test_composite_expression():
    """Test a more complex expression: L = (a + b) * (a - b) = a^2 - b^2."""
    a = Value(3.0)
    b = Value(2.0)
    big_l = (a + b) * (a - b)
    big_l.backward()
    assert big_l.data == 5.0  # 9 - 4
    assert abs(a.grad - 6.0) < 1e-10  # dL/da = 2a
    assert abs(b.grad - (-4.0)) < 1e-10  # dL/db = -2b


# --- Value.no_grad() ---


def test_no_grad_produces_correct_data():
    """Operations under no_grad() compute the same .data as normal."""
    with Value.no_grad():
        a = Value(2.0)
        b = Value(3.0)
        c = a * b + a  # same expression as test_backward_branching_graph
    assert c.data == 8.0


def test_no_grad_skips_graph():
    """Operations under no_grad() produce Values with no children."""
    with Value.no_grad():
        a = Value(2.0)
        b = Value(3.0)
        c = a * b
        d = c + a
    assert c._children == ()
    assert c._local_grads == ()
    assert d._children == ()
    assert d._local_grads == ()


def test_no_grad_all_ops_skip_graph():
    """Every operation type skips graph construction under no_grad()."""
    with Value.no_grad():
        a = Value(2.0)
        results = [
            a + Value(1.0),
            a * Value(3.0),
            a ** 2,
            a.log(),
            a.exp(),
            a.relu(),
            -a,
            a - Value(1.0),
            a / Value(2.0),
        ]
    for r in results:
        assert r._children == (), f"expected no children for {r.data}"


def test_no_grad_context_restores():
    """After exiting no_grad(), operations build graphs again."""
    with Value.no_grad():
        a = Value(2.0)
        b = a * Value(3.0)
    assert b._children == ()

    # Outside the context, graph construction resumes
    c = Value(2.0)
    d = c * Value(3.0)
    assert len(d._children) == 2


def test_backward_still_works_after_no_grad():
    """Training is unaffected by having used no_grad() earlier."""
    with Value.no_grad():
        _ = Value(1.0) + Value(2.0)

    # Normal autograd should still work
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    big_l = c + a
    big_l.backward()
    assert a.grad == 4.0
    assert b.grad == 2.0
