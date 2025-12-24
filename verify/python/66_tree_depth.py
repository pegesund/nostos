# Test: Binary tree depth
class Leaf:
    def __init__(self, value):
        self.value = value

class Node:
    def __init__(self, left, right):
        self.left = left
        self.right = right

def tree_depth(tree):
    if isinstance(tree, Leaf):
        return 1
    return 1 + max(tree_depth(tree.left), tree_depth(tree.right))

def main():
    tree = Node(
        Leaf(1),
        Node(Leaf(2), Node(Leaf(3), Leaf(4)))
    )
    return tree_depth(tree)

print(main())
