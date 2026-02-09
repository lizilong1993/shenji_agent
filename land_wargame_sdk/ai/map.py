"""
Hi there, this is Hu Jian from CASIA. I am excited to release this module under
open source license. It contains some basic but really useful functions for
developing your own agent.
The implementation of path finding in the module is inspired by this awesome
blog which I strongly recommend you to take a look at.
  https://www.redblobgames.com/pathfinding/a-star/introduction.html
Feel free to add or modify anything you want.
Have fun!

NOTE: After this release, the use of `core.utils` is discouraged and will be
deprecated. To migrate your code, just change every appearance of
`from core.utils.map import Map`
inside you agent module to
`from .map import Map`.

P.S. If you dislike these lengthy comments as much as I do, delete them.
That's fine. :stuck_out_tongue_winking_eye:
"""
import heapq
# import json
# import os
# import pickle
import random

# import numpy


class Map:
    def __init__(self, basic_data, cost_data, see_data):
        """
        Load basic map data, move cost data and see data.

        You could do a lot more funny stuff using these three kind of data, e.g.
        get the whole see matrix of a given position via
        `self.see[mode][row, col]` or dynamically modify `self.cost` according
        to the observation of obstacles to generate customized move path.

        :param scenario: int
        """
        self.basic = basic_data["map_data"]

        self.max_row = len(self.basic)
        self.max_col = len(self.basic[0])

        self.cost = cost_data
        self.see = see_data

    def is_valid(self, pos):
        """
        Check if `pos` is inside the map.

        :param pos: int
        :return: bool
        """
        row, col = divmod(pos, 100)
        return 0 <= row < self.max_row and 0 <= col < self.max_col

    def get_map_data(self):
        """Not very useful. Kept for backward compatibility."""
        return self.basic

    def get_neighbors(self, pos):
        """
        Get neighbors of `pos` in six directions.

        :param pos: int
        :return: List[int]
        """
        if not self.is_valid(pos):
            return []
        row, col = divmod(pos, 100)
        return self.basic[row][col]["neighbors"]

    def can_see(self, pos1, pos2, mode):
        """
        Check if `pos1` can see `pos2` with given `mode`.

        `self.see[mode]` is a `numpy.ndarray` that supports multidimensional
        indexing. So you could get the whole see matrix of a given position via
        `self.see[mode][row, col]`.

        By bit-masking see matrices of different positions, you could easily get
        all desired positions. For example,
        `np.argwhere((self.see[0][8, 24] & self.see[0][24, 8]) == True)` returns
        all positions that can be seen from both `0824` and `2408`. Tweak the
        condition and you could create a lot more interesting stuff.

        :param pos1: int
        :param pos2: int
        :param mode: int
        :return: bool
        """
        if (
            not self.is_valid(pos1)
            or not self.is_valid(pos2)
            or not 0 <= mode < len(self.see)
        ):
            return False
        row1, col1 = divmod(pos1, 100)
        row2, col2 = divmod(pos2, 100)
        return self.see[mode][row1, col1, row2, col2]

    def get_distance(self, pos1, pos2):
        """
        Get distance between `pos1` and `pos2`.

        :param pos1: int
        :param pos2: int
        :return: int
        """
        if not self.is_valid(pos1) or not self.is_valid(pos2):
            return -1
        # convert position to cube coordinate
        row1, col1 = divmod(pos1, 100)
        q1 = col1 - (row1 - (row1 & 1)) // 2
        r1 = row1
        s1 = -q1 - r1
        row2, col2 = divmod(pos2, 100)
        q2 = col2 - (row2 - (row2 & 1)) // 2
        r2 = row2
        s2 = -q2 - r2
        # calculate Manhattan distance
        return (abs(q1 - q2) + abs(r1 - r2) + abs(s1 - s2)) // 2

    def gen_move_route(self, begin, end, mode):
        """
        Generate one of the fastest move path from `begin` to `end` with given
        `mode` using A* algorithm.

        Here I use python standard `heapq` package just for convenience. The
        idea of A* search algorithm is quite simple. It combines BFS with
        priority queue and adds some heuristic to accelerate the search.

        When an obstacle appears, you could modify the corresponding
        `self.cost[mode][row][col]` to some big number. Then this function is
        still able to give you one of the fastest path available.

        As you might have thought, it's ok to inject even more insights to the
        so-called `cost`. For example, when feeling a "threat", you could
        increase cost or add heuristic at some positions to avoid passing
        through these "dangerous" regions.

        For those who want to find all positions that can be reached within a
        given amount of time, Dijkstra is all you need. It's only BFS with
        priority queue. To find all positions that can be reached within a given
        amount of time, just modify the exit condition inside `search()` and
        save all valid positions during search. It's left for you brilliant
        developers to implement.

        :param begin: int
        :param end: int
        :param mode: int
        :return: List[int]
        """
        if (
            not self.is_valid(begin)
            or not self.is_valid(end)
            or not 0 <= mode < len(self.cost)
            or begin == end
        ):
            return []
        frontier = [(0, random.random(), begin)]
        cost_so_far = {begin: 0}
        came_from = {begin: None}

        def a_star_search():
            while frontier:
                _, _, cur = heapq.heappop(frontier)
                if cur == end:
                    break
                row, col = divmod(cur, 100)
                for neigh, edge_cost in self.cost[mode][row][col].items():
                    neigh_cost = cost_so_far[cur] + edge_cost
                    if neigh not in cost_so_far or neigh_cost < cost_so_far[neigh]:
                        cost_so_far[neigh] = neigh_cost
                        came_from[neigh] = cur
                        heuristic = self.get_distance(neigh, end)
                        heapq.heappush(frontier, (neigh_cost + heuristic, random.random(), neigh))

        def reconstruct_path():
            path = []
            if end in came_from:
                cur = end
                while cur != begin:
                    path.append(cur)
                    cur = came_from[cur]
                path.reverse()
            return path

        a_star_search()
        return reconstruct_path()
