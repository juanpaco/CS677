from nodes import (Fixed, Normal, Power)

def test_power():
    node1 = Normal(
            'test node 1',
            mean=1,
            var=2,
            val=2
            )

    node2 = Fixed(
            'fixed value',
            val=2
            )

    max_powers = Power(node1, node2)

    assert max_powers.value() == 4
