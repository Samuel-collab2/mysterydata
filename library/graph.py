class Graph:

    def __init__(self, nodes, edges=None):
        self._nodes = nodes
        self._edges = edges or []

    @property
    def nodes(self):
        return self._nodes

    @property
    def edges(self):
        return self._edges

    def connect(self, node1, node2, data):
        self._edges.append((node1, node2, data))
