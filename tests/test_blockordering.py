from .FlowControl import ControlBlock as cb
from .BlockOrdering import compute_dominanators, post_order_nodes


def test_graph(G, expected):
    # G should be a mapping of blocks -> blocks
    pass


def test_graph_1():
    """
    irreducible graph from Figure 2 of A Simple Fast Dominance Algorithm

    """

    blocks = [cb() for i in range(5)]
    nums = {b: i for (i, b) in enumerate(blocks)}

    blocks[0].add_child(blocks[1])
    blocks[1].add_child(blocks[0])
    blocks[2].add_child(blocks[1])
    blocks[3].add_child(blocks[0])
    blocks[4].add_child(blocks[3])
    blocks[4].add_child(blocks[2])

    
    print(", ".join(str(n) for n in nums.values()))
    print('\n\n')

    print('\n'.join('check parents'), 'expect:' '4: None,', '3: 4,',  '2: 4,', '1: {0, 2},', '0: {3, 1},')
    for b in blocks:
        print(nums[b], ': ', ', '.join(str(nums[p]) for p in b.parents))

    order = post_order_nodes(blocks[4])
    print('expected: ', '4 after 2,3 after 0,1')

    print(', '.join(str(nums[o]) for o in order))

    
    doms = compute_dominanators(blocks[0])

    for d in doms:
        print(d)
