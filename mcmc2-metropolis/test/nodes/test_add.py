from nodes import (Add, Normal)

def test_add():
    node1 = Normal(
            'test node 1',
            mean=1,
            var=2,
            val=1
            )

    node2 = Normal(
            'test node 2',
            mean=3,
            var=4,
            val=2
            )

    add = Add(node1, node2)

    assert add.value() == node1.value() + node2.value()
