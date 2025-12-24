# Test: Binary tree sum
class Leaf:
    def __init__(self, value):
        self.value = value

class Node:
    def __init__(self, left, right):
        self.left = left
        self.right = right

def tree_sum(tree):
    if isinstance(tree, Leaf):
        return tree.value
    return tree_sum(tree.left) + tree_sum(tree.right)

def main():
    tree = Node(
        Node(Leaf(1), Leaf(2)),
        Node(Leaf(3), Node(Leaf(4), Leaf(5)))
    )
    return tree_sum(tree)

print(main())
