# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html
"""
Capture-the-flag team.

Layout of this file:
  * Core      - maze analysis (distances, home border, dead-end depth), exact
                Bayesian tracking of hidden opponents, the one-ghost-vs-one-pacman
                chase game solved exactly at start-up, shared team context.
  * Planner   - goal-based agent: picks a target (food, capsule, home, invader,
                patrol point) and routes to it with a ghost-aware Dijkstra.

create_team(**params) builds the team; DEFAULT_PARAMS are what the contest plays,
since it passes no options.
"""

import heapq
import os
import random
from collections import deque
from itertools import combinations

from contest.capture_agents import CaptureAgent
from contest.game import Directions

DEBUG = bool(os.environ.get('PACMAN_AGENT_DEBUG'))

INF = float('inf')
SIGHT_RANGE = 5
SONAR_NOISE = 6
SCARED_HARMLESS = 2  # a scared ghost with more turns than this left cannot hurt us next move
CHASE_MAX_CELLS = 900  # the chase tables grow with the square of the maze size: skip them on huge mazes
CHASE_PRECOMPUTE_CAPSULES = 3  # build the tables for every capsule subset at start-up up to this many capsules

_VEC = {
    Directions.NORTH: (0, 1),
    Directions.SOUTH: (0, -1),
    Directions.EAST: (1, 0),
    Directions.WEST: (-1, 0),
    Directions.STOP: (0, 0),
}


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def action_towards(a, b):
    """Action that moves from cell a to the adjacent cell b (STOP if equal)."""
    d = (b[0] - a[0], b[1] - a[1])
    for act, v in _VEC.items():
        if v == d:
            return act
    return Directions.STOP


def as_cell(pos):
    return None if pos is None else (int(pos[0]), int(pos[1]))


#########################################################################
# Core: maze analysis                                                   #
#########################################################################

class Maze:
    def __init__(self, game_state, red, distances):
        walls = game_state.get_walls()
        self.width, self.height = walls.width, walls.height
        self.red = red
        self.mid = self.width // 2
        self.cells = [as_cell(c) for c in walls.as_list(False)]
        cellset = set(self.cells)
        self.nbrs = {}
        for (x, y) in self.cells:
            self.nbrs[(x, y)] = [n for n in ((x, y + 1), (x, y - 1), (x + 1, y), (x - 1, y)) if n in cellset]
        self._d = distances

        home_x = self.mid - 1 if red else self.mid
        enemy_x = self.mid if red else self.mid - 1
        self.home_cells = frozenset(c for c in self.cells if self.is_red_side(c) == red)
        self.enemy_cells = frozenset(c for c in self.cells if c not in self.home_cells)
        self.border = [c for c in self.cells if c[0] == home_x and (enemy_x, c[1]) in cellset]
        self.enemy_border = [c for c in self.cells if c[0] == enemy_x and (home_x, c[1]) in cellset]
        self._dead_ends()
        self.home_dist = {c: min(self.dist(c, b) for b in self.border) for c in self.cells}
        self.enemy_home_dist = {c: min(self.dist(c, b) for b in self.enemy_border) for c in self.cells}

    def is_red_side(self, c):
        return c[0] < self.mid

    def is_home(self, c):
        return c in self.home_cells

    def dist(self, a, b):
        return self._d[(a, b)]

    def _dead_ends(self):
        """Peel leaves off the maze graph. Cells removed this way sit in dead-end
        trees: depth = steps to the tree's exit, exit = first cell on the cyclic core."""
        deg = {c: len(n) for c, n in self.nbrs.items()}
        alive = set(self.cells)
        order, parent = [], {}
        queue = deque(c for c in self.cells if deg[c] <= 1)
        while queue:
            c = queue.popleft()
            if c not in alive:
                continue
            alive.discard(c)
            order.append(c)
            parent[c] = None
            for n in self.nbrs[c]:
                if n in alive:
                    parent[c] = n
                    deg[n] -= 1
                    if deg[n] <= 1:
                        queue.append(n)
        self.depth = {c: 0 for c in self.cells}
        self.exit = {c: c for c in self.cells}
        for c in reversed(order):
            p = parent[c]
            if p is not None:
                self.depth[c] = self.depth[p] + 1
                self.exit[c] = self.exit[p]


#########################################################################
# Core: opponent tracking (exact HMM over maze cells)                   #
#########################################################################

class Tracker:
    def __init__(self, game_state, maze, enemies):
        self.maze = maze
        self.enemies = enemies
        self.start = {e: as_cell(game_state.get_initial_agent_position(e)) for e in enemies}
        self.belief = {e: {self.start[e]: 1.0} for e in enemies}

    def elapse(self, e):
        nbrs = self.maze.nbrs
        new = {}
        for c, p in self.belief[e].items():
            opts = nbrs[c]
            share = p / (len(opts) + 1)
            new[c] = new.get(c, 0.0) + share
            for n in opts:
                new[n] = new.get(n, 0.0) + share
        self.belief[e] = new

    def set_position(self, e, cell):
        self.belief[e] = {cell: 1.0}

    def observe(self, game_state, observer, team):
        my_pos = as_cell(game_state.get_agent_position(observer))
        team_pos = [as_cell(game_state.get_agent_position(i)) for i in team]
        noisy = game_state.get_agent_distances()
        for e in self.enemies:
            st = game_state.get_agent_state(e)
            pos = as_cell(st.get_position())
            if pos is not None:
                self.belief[e] = {pos: 1.0}
                continue
            is_pac = st.is_pacman
            reading = noisy[e] if noisy else None

            def consistent(c):
                if self.maze.is_home(c) != is_pac:
                    return False
                for t in team_pos:
                    if manhattan(c, t) <= SIGHT_RANGE:
                        return False
                return reading is None or abs(manhattan(c, my_pos) - reading) <= SONAR_NOISE

            new = {c: p for c, p in self.belief[e].items() if consistent(c)}
            total = sum(new.values())
            if total <= 0:
                # The model broke: almost always because the enemy was eaten and respawned.
                start = self.start[e]
                if consistent(start):
                    new, total = {start: 1.0}, 1.0
                else:
                    new = {c: 1.0 for c in self.maze.cells if consistent(c)} or {start: 1.0}
                    total = float(len(new))
            self.belief[e] = {c: p / total for c, p in new.items()}

    def mode(self, e):
        b = self.belief[e]
        return max(b, key=b.get)

    def expected_dist(self, e, cell):
        d = self.maze.dist
        return sum(p * d(c, cell) for c, p in self.belief[e].items())


#########################################################################
# Core: the chase game (one ghost against one pacman), solved exactly   #
#########################################################################

class Chase:
    """One ghost chasing one pacman on the ghost's half of the maze, as a game of
    alternating moves, solved exactly by retrograde analysis at start-up. The
    pacman is safe as soon as it steps off this half (or onto an `escapes` cell,
    e.g. a capsule). For every pair of positions we know whether the ghost can
    force a capture and in how many of its moves:
      ghost_to_move(pacman, ghost) / pacman_to_move(pacman, ghost) -> moves or None.
    The defender uses it to finish off invaders, the attacker to stay out of
    positions that are already lost."""

    def __init__(self, maze, cells, escapes=()):
        cells = sorted(cells)
        n = len(cells)
        self.n = n
        self.index = {c: i for i, c in enumerate(cells)}
        cellset = set(cells)
        escapes = set(escapes)
        # pacman options: stay or step; free[i] = it can step to safety from cell i
        self.opts = []
        free = []
        for c in cells:
            nb = maze.nbrs[c]
            free.append(c in escapes or any(m not in cellset or m in escapes for m in nb))
            self.opts.append([self.index[m] for m in nb if m in cellset] + [self.index[c]])
        opts = self.opts
        g_win = [0] * (n * n)  # [pacman * n + ghost], ghost to move: moves to capture (0 = cannot force it)
        p_win = [-1] * (n * n)  # pacman to move (-1 = cannot force it)
        left = [0] * (n * n)  # pacman to move: options not yet known to lose
        for pi in range(n):
            k = len(opts[pi])
            base = pi * n
            for gi in range(n):
                left[base + gi] = k - (1 if gi in opts[pi] and gi != pi else 0)  # walking into the ghost loses at once
        level = []
        for pi in range(n):
            for gi in opts[pi]:
                if gi != pi:
                    g_win[pi * n + gi] = 1
                    level.append((pi, gi))
        t = 1
        while level:
            lost = []
            for pi, gi in level:  # the pacman may not move into (pi, gi): update the cells it could come from
                for qi in opts[pi]:
                    if qi == gi or free[qi]:
                        continue
                    k = qi * n + gi
                    if p_win[k] >= 0:
                        continue
                    left[k] -= 1
                    if left[k] == 0:
                        p_win[k] = t
                        lost.append((qi, gi))
            t += 1
            level = []
            for pi, gi in lost:  # the ghost may move into (pi, gi) from any neighbouring cell
                for hi in opts[gi]:
                    k = pi * n + hi
                    if g_win[k] == 0 and hi != pi:
                        g_win[k] = t
                        level.append((pi, hi))
        self.g_win, self.p_win = g_win, p_win
        self.cells = cells
        self._solve_escape(free)

    def _solve_escape(self, free):
        """Second question on the same board: from which positions can the pacman
        force its way to safety, whatever the ghost does? (Everywhere else the ghost
        can at least keep it locked in.) p_out / g_out: pacman / ghost to move."""
        n, opts = self.n, self.opts
        adjacent = [set(o) for o in opts]
        p_out = [False] * (n * n)
        g_out = [False] * (n * n)
        left = [0] * (n * n)  # ghost to move: ghost moves not yet known to let the pacman out
        for gi in range(n):
            k = len(opts[gi])
            for pi in range(n):
                left[pi * n + gi] = k
        queue = []
        for pi in range(n):
            if free[pi]:
                base = pi * n
                for gi in range(n):
                    if gi != pi:
                        p_out[base + gi] = True
                        queue.append((pi, gi))
        while queue:
            nxt = []
            for pi, gi in queue:  # pacman to move at (pi, gi) gets out: ghost moves that lead here are no good
                for hi in opts[gi]:
                    if hi == pi or pi in adjacent[hi]:
                        continue  # the ghost would rather capture from hi
                    k = pi * n + hi
                    if g_out[k]:
                        continue
                    left[k] -= 1
                    if left[k] == 0:
                        g_out[k] = True
                        for qi in opts[pi]:  # the pacman may step (or wait) into this position
                            if qi != hi and not p_out[qi * n + hi]:
                                p_out[qi * n + hi] = True
                                nxt.append((qi, hi))
            queue = nxt
        self.p_out = p_out

    def sealing_steps(self, pac, ghost):
        """Ghost moves (cells) after which the pacman can no longer force its way out."""
        pi, gi = self.index.get(pac), self.index.get(ghost)
        if pi is None or gi is None:
            return []
        return [self.cells[hi] for hi in self.opts[gi] if hi != pi and not self.p_out[pi * self.n + hi]]

    def ghost_to_move(self, pac, ghost):
        """Moves the ghost needs to catch the pacman when the ghost moves first, or None."""
        pi, gi = self.index.get(pac), self.index.get(ghost)
        if pi is None or gi is None:
            return None
        return self.g_win[pi * self.n + gi] or None

    def pacman_to_move(self, pac, ghost):
        """Moves the ghost needs to catch the pacman when the pacman moves first, or None."""
        pi, gi = self.index.get(pac), self.index.get(ghost)
        if pi is None or gi is None:
            return None
        t = self.p_win[pi * self.n + gi]
        return t if t >= 0 else None

    def ghost_step(self, pac, ghost):
        """Best ghost move (a cell) when it can force a capture, else None."""
        pi, gi = self.index.get(pac), self.index.get(ghost)
        if pi is None or gi is None or not self.g_win[pi * self.n + gi]:
            return None
        best, best_t = None, None
        for hi in self.opts[gi]:
            if hi == pi:
                return self.cells[hi]
            t = self.p_win[pi * self.n + hi]
            if t >= 0 and (best_t is None or t < best_t):
                best, best_t = hi, t
        return None if best is None else self.cells[best]


#########################################################################
# Core: team context shared by both agents of a team                    #
#########################################################################

_TEAM_CONTEXT = {}


class TeamContext:
    def __init__(self, agent, game_state):
        self.layout_key = id(game_state.data.layout)
        self.red = agent.red
        self.maze = Maze(game_state, agent.red, agent.distancer._distances)
        self.team = agent.get_team(game_state)
        self.enemies = agent.get_opponents(game_state)
        self.tracker = Tracker(game_state, self.maze, self.enemies)
        self.defend_food = set(agent.get_food_you_are_defending(game_state).as_list())
        self.defend_capsules = set(agent.get_capsules_you_are_defending(game_state))
        # chase tables (see Chase): us as ghost on our half, where an invader is safe on
        # a capsule, and an enemy ghost against us on their half. A table depends on which
        # capsules are left, so it is built at start-up for every set of capsules that can
        # still occur (when there are few), not in the middle of a move.
        self.chase_home = self.chase_away = None
        self.chase_tables = {}
        self.use_chase = len(self.maze.cells) <= CHASE_MAX_CELLS
        if self.use_chase:
            home_caps = [as_cell(c) for c in self.defend_capsules]
            away_caps = [as_cell(c) for c in agent.get_capsules(game_state)]
            for side, caps in (('home', home_caps), ('away', away_caps)):
                if len(caps) <= CHASE_PRECOMPUTE_CAPSULES:
                    for k in range(len(caps) + 1):
                        for subset in combinations(caps, k):
                            self.chase_table(side, subset)
            self.update_chase(agent, game_state)
        self.total_food = len(agent.get_food(game_state).as_list())
        self.targets = {}  # agent index -> current attack target
        self.roles = {}  # agent index -> 'attack' / 'defend'
        self.last_eaten = None  # (cell, timeleft) of the most recent food the enemy ate
        self.updated_at = None
        self.progress_total = 0  # food our team carries or has banked
        self.last_progress = None  # timeleft when that last went up
        self.open_mode = False  # open-game mode: the defender attacks too
        self.last_enemy_invasion = None  # timeleft when an enemy was last on our side

    @staticmethod
    def get(agent, game_state):
        ctx = _TEAM_CONTEXT.get(agent.red)
        if ctx is None or ctx.layout_key != id(game_state.data.layout) or ctx.team != agent.get_team(game_state):
            ctx = TeamContext(agent, game_state)
            _TEAM_CONTEXT[agent.red] = ctx
        return ctx

    def update(self, agent, game_state):
        """Called at the start of every turn of either teammate."""
        timeleft = game_state.data.timeleft
        if self.updated_at == timeleft:
            return
        first_update = self.updated_at is None
        self.updated_at = timeleft
        # Exactly one opponent moved since the previous team update: the one just before us.
        mover = (agent.index - 1) % game_state.get_num_agents()
        if not first_update and mover in self.enemies:
            self.tracker.elapse(mover)

        food_now = set(agent.get_food_you_are_defending(game_state).as_list())
        eaten = self.defend_food - food_now
        caps_now = set(agent.get_capsules_you_are_defending(game_state))
        eaten |= (self.defend_capsules - caps_now)
        if eaten and mover in self.enemies:
            cell = min(eaten, key=lambda c: -self.tracker.belief[mover].get(c, 0.0))
            self.tracker.set_position(mover, as_cell(cell))
            self.last_eaten = (as_cell(cell), timeleft)
        self.defend_food = food_now
        self.defend_capsules = caps_now
        if self.use_chase:
            self.update_chase(agent, game_state)
        # Team progress: food we are carrying or have banked. A drop (a death) resets the baseline.
        total = sum(game_state.get_agent_state(i).num_carrying + game_state.get_agent_state(i).num_returned
                    for i in self.team)
        if self.last_progress is None:
            # the clock starts once one of us has reached the border (maps differ a lot in size)
            if any(self.maze.home_dist[as_cell(game_state.get_agent_position(i))] <= 1 for i in self.team):
                self.last_progress = timeleft
        elif total > self.progress_total:
            self.last_progress = timeleft
        self.progress_total = total
        hop = agent.p('quiet_hop', 0)
        for e in self.enemies:
            st = game_state.get_agent_state(e)
            if not st.is_pacman:
                continue
            p = as_cell(st.get_position())
            if hop and st.num_carrying == 0 and p is not None and self.maze.enemy_home_dist[p] <= hop:
                continue  # an empty-handed enemy hopping over the border line and back
            self.last_enemy_invasion = timeleft
            break
        self.tracker.observe(game_state, agent.index, self.team)

    def chase_table(self, side, capsules):
        key = (side, frozenset(capsules))
        table = self.chase_tables.get(key)
        if table is None:
            cells = self.maze.home_cells if side == 'home' else self.maze.enemy_cells
            table = self.chase_tables[key] = Chase(self.maze, cells, key[1])
        return table

    def update_chase(self, agent, game_state):
        self.chase_home = self.chase_table('home', [as_cell(c) for c in self.defend_capsules])
        self.chase_away = self.chase_table('away', [as_cell(c) for c in agent.get_capsules(game_state)])

    def stalled_moves(self, game_state):
        """Moves (per agent) since our team last ate or banked any food."""
        if self.last_progress is None:
            return 0
        return (self.last_progress - game_state.data.timeleft) // 4

    def quiet_moves(self, game_state):
        """Moves (per agent) since an enemy was last on our side (large if never)."""
        if self.last_enemy_invasion is None:
            return 10 ** 6
        return (self.last_enemy_invasion - game_state.data.timeleft) // 4


#########################################################################
# Base agent                                                            #
#########################################################################

class BaseAgent(CaptureAgent):
    """Shared plumbing: team context, safe action selection, common queries."""

    def __init__(self, index, role='attack', **params):
        super().__init__(index)
        self.role = role
        self.default_role = role
        self.params = params
        self.ctx = None
        self.maze = None

    def register_initial_state(self, game_state):
        CaptureAgent.register_initial_state(self, game_state)
        self.ctx = TeamContext.get(self, game_state)
        self.maze = self.ctx.maze
        self.start = as_cell(game_state.get_agent_position(self.index))
        self.teammate = [i for i in self.ctx.team if i != self.index][0]
        self.on_start(game_state)

    def on_start(self, game_state):
        self.history = []
        self.taboo = {}

    def choose_action(self, game_state):
        legal = game_state.get_legal_actions(self.index)
        try:
            self.ctx.update(self, game_state)
            action = self.decide(game_state, legal)
        except Exception:
            if DEBUG or self.params.get('strict'):
                raise
            action = None
        if action not in legal:
            non_stop = [a for a in legal if a != Directions.STOP]
            action = random.choice(non_stop or legal)
        return action

    def decide(self, game_state, legal):
        raise NotImplementedError

    # ---- common queries ------------------------------------------------
    def p(self, name, default):
        return self.params.get(name, default)

    def my_pos(self, gs):
        return as_cell(gs.get_agent_position(self.index))

    def moves_left(self, gs):
        """Moves this agent still gets (each agent moves once per 4 game steps)."""
        return gs.data.timeleft // 4

    def score(self, gs):
        return self.get_score(gs)

    def visible_enemies(self, gs):
        out = []
        for e in self.ctx.enemies:
            st = gs.get_agent_state(e)
            pos = as_cell(st.get_position())
            if pos is not None:
                out.append((e, pos, st))
        return out

    def dangerous_ghosts(self, gs):
        """Visible enemy ghosts that can eat us (on their side, not safely scared)."""
        return [(e, pos) for e, pos, st in self.visible_enemies(gs)
                if not st.is_pacman and st.scared_timer <= SCARED_HARMLESS]

    def scared_ghosts(self, gs):
        """Visible enemy ghosts that are still scared: (position, moves until they can eat us again)."""
        return [(pos, st.scared_timer) for e, pos, st in self.visible_enemies(gs)
                if not st.is_pacman and st.scared_timer > SCARED_HARMLESS]

    def invaders(self, gs):
        return [(e, pos) for e, pos, st in self.visible_enemies(gs) if st.is_pacman]


#########################################################################
# Planner: goal-based agent with ghost-aware Dijkstra                   #
#########################################################################

def dijkstra(nbrs, src, step_cost, max_cost=INF):
    """step_cost(cell, steps_to_cell) -> extra cost (INF to block). Returns (cost, steps, parent)."""
    cost = {src: 0.0}
    steps = {src: 0}
    parent = {src: None}
    heap = [(0.0, 0, src)]
    while heap:
        c_cost, c_steps, c = heapq.heappop(heap)
        if c_cost > cost[c]:
            continue
        for n in nbrs[c]:
            s = c_steps + 1
            extra = step_cost(n, s)
            if extra == INF:
                continue
            nc = c_cost + 1 + extra
            if nc < cost.get(n, INF) and nc <= max_cost:
                cost[n] = nc
                steps[n] = s
                parent[n] = c
                heapq.heappush(heap, (nc, s, n))
    return cost, steps, parent


def first_step(parent, src, target):
    c = target
    if c == src or c not in parent:
        return src
    while parent[c] != src:
        c = parent[c]
    return c


class PlannerAgent(BaseAgent):
    """
    Each turn: update roles, then
      attack: pick the best safe food (or capsule / home) and route with a cost map
              that blocks cells a ghost can reach first and dead ends it can seal.
      defend: catch a visible invader when the chase table says we can force it,
              else lock it in so it can never get home, else cut it off from what
              it can still reach (pursuit_step); for unseen invaders go where the
              tracker says they are, else guard the border cell closest to the
              most threatening enemy.
      roles:  one attacker and one defender by default; see update_role for when
              the attacker comes back and when the defender attacks too.
    """

    def decide(self, gs, legal):
        self.update_role(gs)
        pos = self.my_pos(gs)
        if self.p('stall_window', 0):
            self.check_stall(gs, pos)
        self.goal = None
        run = self.breakout_step(gs, pos)
        if run is not None:
            self.last_goal = self.goal
            self.history.append(pos)
            if len(self.history) > 8:
                self.history.pop(0)
            return action_towards(pos, run) if run != pos else self.fallback_action(gs, pos, legal)
        if self.role == 'attack':
            target = self.attack_target(gs, pos)
            if self.p('unstick', False) and self.oscillating(pos, target):
                # we are about to undo our last move again: forbid it for a while and replan
                self.taboo[target] = gs.data.timeleft - 4 * self.p('taboo_moves', 6)
                target = self.attack_target(gs, pos)
        elif pos not in self.maze.home_cells:
            # a defender that is still on the other side comes home the careful way
            self.force_home = True
            target = self.attack_target(gs, pos)
            self.force_home = False
        else:
            target = self.defend_target(gs, pos)
        if self.role == 'attack' or pos not in self.maze.home_cells:
            target = self.safe_step(gs, pos, target)
        self.last_goal = self.goal
        self.history.append(pos)
        if len(self.history) > 8:
            self.history.pop(0)
        if target is None or target == pos:
            return self.fallback_action(gs, pos, legal)
        return action_towards(pos, target)

    def breakout_step(self, gs, pos):
        """A pacman boxed in for a long time: it has stood within a few cells for
        `breakout` moves (typically a ghost parked at the only way out, which never
        comes in for it). Waiting gains nothing for the rest of the game, so it runs
        home along the route that stays farthest from the ghosts, even if that means
        being caught: respawning at home beats standing still. Sooner
        (`breakout_urgent` moves) when an invader is on our side and no teammate is
        at home to stop it. Returns the next cell, or None."""
        n = self.p('breakout', 0)
        maze = self.maze
        if not n or pos in maze.home_cells:
            self.box_trail, self.breaking = [], False
            return None
        if not hasattr(self, 'box_trail'):
            self.box_trail, self.breaking = [], False
        trail = self.box_trail
        trail.append(pos)
        urgent = self.p('breakout_urgent', 0)
        hurry = False
        if urgent:
            mate = as_cell(gs.get_agent_position(self.teammate))
            mate_home = mate in maze.home_cells and gs.get_agent_state(self.teammate).scared_timer == 0
            if not mate_home and any(gs.get_agent_state(e).is_pacman for e in self.ctx.enemies):
                n = min(n, urgent)
                hurry = True
        if len(trail) > self.p('breakout', 0):
            trail.pop(0)
        # Ahead, a standoff ties up one of theirs as much as one of ours and keeps the
        # lead; only an invader eating at home makes it worth breaking up.
        if self.score(gs) > self.p('breakout_lead', 1000) and not hurry:
            self.breaking = False
            return None
        if not self.breaking:
            recent = trail[-n:]
            if len(recent) < n or len(set(recent)) > self.p('breakout_cells', 3):
                return None
            self.breaking = True
        ghosts = [g for _, g in self.dangerous_ghosts(gs)]

        def cost(c, s):
            if c in maze.home_cells or not ghosts:
                return 0.0
            gd = min(maze.dist(g, c) for g in ghosts)
            if gd <= 1:
                return 30.0
            return 6.0 / gd if gd <= 3 else 0.0
        dist, _, parent = dijkstra(maze.nbrs, pos, cost)
        goal = min(maze.border, key=lambda b: (dist.get(b, INF), b))
        self.goal = goal
        return first_step(parent, pos, goal)

    def check_stall(self, gs, pos):
        """An attacker walking in a small loop (e.g. a ghost keeps blocking the way
        to its goal) bans that goal for a while and, if it is at home, goes to try
        another crossing. Our attacker being called back again and again to help at
        home counts as held up too: some teams keep it busy by hopping over the
        border and back."""
        t = gs.data.timeleft
        if not hasattr(self, 'trail'):
            self.trail, self.goal_taboo = [], {}
        helping = self.default_role == 'attack' and pos in self.maze.home_cells
        if self.role != 'attack' and not helping:
            self.trail = []
            return
        w = self.p('stall_window', 16)
        self.trail.append(pos)
        if len(self.trail) > w:
            self.trail.pop(0)
        if len(self.trail) == w and len(set(self.trail)) <= self.p('stall_unique', 6):
            self.trail = []
            last = getattr(self, 'last_goal', None)
            if last is not None:
                self.goal_taboo[last] = t - 4 * self.p('stall_ban', 30)
            if pos in self.maze.home_cells:
                self.start_relocation(gs, pos)

    def start_relocation(self, gs, pos):
        """Held up at the border: commit to the crossing farthest from the enemy
        ghosts (seen, or else where the tracker puts them) and walk there."""
        maze = self.maze
        ghosts = []
        for e in self.ctx.enemies:
            st = gs.get_agent_state(e)
            g = as_cell(st.get_position()) or self.ctx.tracker.mode(e)
            # guards: enemies on their side or hopping on the border line (not real invaders)
            if not st.is_pacman or maze.home_dist[g] <= 1:
                ghosts.append(g)
        if not ghosts:
            return
        cell = max(maze.border, key=lambda b: (min(maze.dist(g, b) for g in ghosts), -maze.dist(pos, b)))
        if maze.dist(pos, cell) > 3:
            self.relocating = (cell, gs.data.timeleft - 4 * self.p('relocate_moves', 30))

    def banned_goals(self, gs):
        taboo = getattr(self, 'goal_taboo', None)
        if not taboo:
            return set()
        t = gs.data.timeleft
        return {c for c, until in taboo.items() if until < t}

    def oscillating(self, pos, nxt):
        h = self.history
        return len(h) >= 3 and nxt == h[-1] and pos == h[-2] and nxt == h[-3]

    def taboo_cells(self, gs):
        t = gs.data.timeleft
        return {c for c, until in self.taboo.items() if until < t}

    # ---- roles ---------------------------------------------------------
    def update_role(self, gs):
        role = self.default_role
        if self.p('both_attack', False):
            role = 'attack'
        score = self.score(gs)
        moves_left = self.moves_left(gs)
        # Protect a lead at the end of the game.
        if score > 0 and moves_left < self.p('endgame_defend', 30):
            role = 'defend'
        if score >= self.p('lead_defend', 1000):
            role = 'defend'
        # An attacker that is still at home helps when an invader is close by.
        if role == 'attack' and self.p('help_defend', True) and getattr(self, 'relocating', None) is None:
            pos = self.my_pos(gs)
            invaders = self.invaders(gs)
            if self.maze.is_home(pos) and invaders and \
                    min(self.maze.dist(pos, p) for _, p in invaders) <= self.p('help_radius', 4) and \
                    gs.get_agent_state(self.index).scared_timer == 0:
                role = 'defend'
        # Invaders hold a big haul on our side: catching one returns all of it, and
        # two defenders can corner a pacman where one cannot, so the attacker comes
        # back (unless it carries more itself).
        heavy = self.p('heavy_defend', 0)
        if role == 'attack' and heavy:
            me = gs.get_agent_state(self.index)
            stolen = sum(gs.get_agent_state(e).num_carrying for e in self.ctx.enemies
                         if gs.get_agent_state(e).is_pacman)
            if stolen >= heavy and me.scared_timer == 0 and me.num_carrying < stolen:
                role = 'defend'
        # Open-game mode: we are not ahead and the team has made no progress
        # (nothing eaten or banked) for a long time. A 0-0 tie is worth 1 point and
        # an open game at even odds about 1.5, so the defender commits to attacking
        # too until we lead. At home it still answers an invader that is close by.
        open_after = self.p('open_stall', 0)
        if open_after and self.default_role == 'defend':
            ctx = self.ctx
            if score > 0:
                ctx.open_mode = False
            elif not ctx.open_mode and ctx.stalled_moves(gs) >= open_after and \
                    ctx.quiet_moves(gs) >= self.p('open_quiet', 20):
                ctx.open_mode = True
            if ctx.open_mode and role == 'defend' and moves_left > self.p('endgame_defend', 30):
                role = 'attack'
                pos = self.my_pos(gs)
                invaders = self.invaders(gs)
                if self.maze.is_home(pos) and invaders and gs.get_agent_state(self.index).scared_timer == 0 and \
                        min(self.maze.dist(pos, p) for _, p in invaders) <= self.p('open_help_radius', 6):
                    role = 'defend'
        # A scared ghost cannot defend, but on the other side it is an ordinary
        # pacman: raid until the scare is about to wear off.
        scared_raid = self.p('scared_raid', 0)
        if scared_raid and role == 'defend' and moves_left > self.p('endgame_defend', 30) and \
                gs.get_agent_state(self.index).scared_timer > scared_raid:
            role = 'attack'
        self.role = role
        self.ctx.roles[self.index] = role

    # ---- attack --------------------------------------------------------
    def attack_target(self, gs, pos):
        maze = self.maze
        me = gs.get_agent_state(self.index)
        carrying = me.num_carrying
        food = [as_cell(f) for f in self.get_food(gs).as_list()]
        capsules = [as_cell(c) for c in self.get_capsules(gs)]
        n_food = len(food)
        banned = self.banned_goals(gs)
        if banned:
            capsules = [c for c in capsules if c not in banned]
            food = [f for f in food if f not in banned] or food
        ghosts = self.dangerous_ghosts(gs)
        moves_left = self.moves_left(gs)
        home_d = maze.home_dist[pos]

        scared_left = min([gs.get_agent_state(e).scared_timer for e in self.ctx.enemies] or [0])
        ghost_pos = [g for _, g in ghosts]
        nearest_ghost = min([maze.dist(pos, g) for g in ghost_pos] or [INF])
        threatened = nearest_ghost <= self.p('threat_radius', 5)

        relocate = self.relocate_step(gs, pos, ghost_pos, carrying)
        if relocate is not None:
            return relocate

        late = self.scared_ghosts(gs)
        base_cost = self.danger_cost(ghost_pos, gs, pos, late)
        taboo = self.taboo_cells(gs)
        if taboo:
            step_cost = lambda c, s: INF if (s == 1 and c in taboo) else base_cost(c, s)
        else:
            step_cost = base_cost
        cost, steps, parent = dijkstra(maze.nbrs, pos, step_cost)

        def reachable(cells):
            return [c for c in cells if c in cost]

        # 1. Must go home?
        carry_limit = self.p('carry_limit', 6)
        if scared_left > home_d + 3:
            carry_limit = self.p('carry_limit_scared', 12)
        go_home = (
            n_food <= 2
            or (carrying > 0 and moves_left <= home_d + self.p('time_margin', 4))
            or carrying >= carry_limit
            or getattr(self, 'force_home', False)
        )
        home_targets = reachable(maze.border) if not maze.is_home(pos) else []

        # 2. Under threat, a reachable capsule is the best escape.
        if threatened and capsules:
            safe_caps = [c for c in reachable(capsules)
                         if all(steps[c] < maze.dist(g, c) for g in ghost_pos)]
            if safe_caps:
                return self.step_to(parent, pos, min(safe_caps, key=lambda c: cost[c]))

        # 2b. A ghost is guarding: grab a capsule we can reach first, then feast.
        if self.p('capsule_first', False) and capsules and ghost_pos and scared_left == 0 and not go_home:
            guard_r = self.p('capsule_guard_radius', 8)
            if nearest_ghost <= guard_r:
                safe_caps = [c for c in reachable(capsules)
                             if all(steps[c] < maze.dist(g, c) for g in ghost_pos)]
                if safe_caps:
                    return self.step_to(parent, pos, min(safe_caps, key=lambda c: cost[c]))

        if go_home and not maze.is_home(pos):
            if home_targets:
                return self.step_to(parent, pos, min(home_targets, key=lambda c: cost[c]))
            return self.escape_step(gs, pos, ghost_pos)

        # 3. Pick food: cheapest safe food, avoiding the teammate's target.
        mate_target = self.ctx.targets.get(self.teammate) if self.ctx.roles.get(self.teammate) == 'attack' else None
        prev_target = self.ctx.targets.get(self.index)
        commit = self.p('commit_bonus', 0)
        # Second attacker while the first is held up at the border: enter somewhere else.
        mate_held = None
        if self.p('wait_spread', 0) and self.default_role == 'defend' and \
                self.ctx.roles.get(self.teammate) == 'attack':
            mate_pos = as_cell(gs.get_agent_position(self.teammate))
            if mate_pos in maze.home_cells:
                mate_held = mate_pos
        candidates = reachable(food)
        if candidates:
            def food_score(f):
                s = cost[f]
                if f == prev_target:
                    s -= commit  # hysteresis: do not flip-flop between targets
                if mate_target is not None:
                    s += self.p('spread', 3) * max(0, 6 - maze.dist(f, mate_target))
                if mate_held is not None:
                    s += 2 * self.p('wait_spread', 0) * max(0, 10 - maze.dist(f, mate_held))
                # mild preference for food closer to home once carrying
                s += self.p('home_pull', 0.0) * carrying * maze.home_dist[f]
                return s
            best = min(candidates, key=food_score)
            self.ctx.targets[self.index] = best
            return self.step_to(parent, pos, best)

        # 4. No safe food: go home with what we have, else wait / escape.
        if not maze.is_home(pos):
            if home_targets:
                return self.step_to(parent, pos, min(home_targets, key=lambda c: cost[c]))
            return self.escape_step(gs, pos, ghost_pos)
        # At home and nothing is safe: hold near the border away from ghosts.
        return self.border_wait_target(gs, pos, ghost_pos)

    def risk_level(self, gs):
        """How many steps 'late' we accept arriving at a cell a ghost could also
        reach. 0 = strict. Grows when tied or behind late in the game: a tie is
        worth 1 point, a win 3, so a stalemate is worth gambling on."""
        if self.score(gs) > 0:
            return 0
        moves_left = self.moves_left(gs)
        level = 0
        for threshold in (self.p('risk_t1', 0), self.p('risk_t2', 0)):
            if moves_left < threshold:
                level += 1
        # ... and when the team has gained nothing for a long time (a guard parked
        # where we need to go): waiting will not get any better either.
        stall = self.p('stall_risk', 0)
        if stall:
            level = max(level, min(2, self.ctx.stalled_moves(gs) // stall))
        return level

    def danger_cost(self, ghost_pos, gs=None, pos=None, late=()):
        """late: scared ghosts as (position, timer). They cannot hurt us before the
        timer runs out, but they can walk anywhere meanwhile, so a scared ghost
        'arrives' at a cell after max(distance, timer) moves."""
        maze = self.maze
        if not ghost_pos and not late:
            return lambda c, s: 0.0
        risk = self.risk_level(gs) if gs is not None else 0
        horizon = self.p('danger_horizon', 6)
        dead_margin = self.p('dead_end_margin', 1)
        gd_cache = {}

        def gdist(c):
            v = gd_cache.get(c)
            if v is None:
                v = min([maze.dist(g, c) for g in ghost_pos] + [max(maze.dist(g, c), t) for g, t in late])
                gd_cache[c] = v
            return v

        def cost(c, s):
            if c in maze.home_cells:
                return 0.0
            gd = gdist(c)
            if gd <= 1:
                return INF
            if s <= horizon and gd + risk <= s:
                return INF  # a ghost can be there no later than us
            # dead end: we must be able to get back out before a ghost seals the exit
            d = maze.depth[c]
            if d > 0:
                ex = maze.exit[c]
                if s + d + dead_margin >= gdist(ex) + risk:
                    return INF
            if gd <= 3:
                return 6.0 / gd
            return 0.0
        return cost

    def step_to(self, parent, pos, target):
        self.goal = target
        return first_step(parent, pos, target)

    def escape_step(self, gs, pos, ghost_pos):
        """No safe plan: move to the neighbour that keeps us farthest from ghosts,
        preferring cells that are not dead ends and are closer to home."""
        maze = self.maze
        best, best_val = pos, -INF
        # Two of us fleeing the same ghost into the same corner get caught together:
        # keep away from a teammate that is also a pacman.
        spread = self.p('escape_spread', 0)
        mate = as_cell(gs.get_agent_position(self.teammate)) if spread else None
        if mate is not None and not gs.get_agent_state(self.teammate).is_pacman:
            mate = None
        for n in maze.nbrs[pos] + [pos]:
            gd = min([maze.dist(g, n) for g in ghost_pos] or [20])
            if gd <= 1 and n not in maze.home_cells:
                val = -1000 + gd
            else:
                val = min(gd, 6) * 10 - maze.depth[n] * 3 - maze.home_dist[n]
                if mate is not None:
                    val -= spread * max(0, 4 - maze.dist(n, mate))
            if n in maze.home_cells:
                val += 50
            if val > best_val:
                best, best_val = n, val
        return best

    def relocate_step(self, gs, pos, ghost_pos, carrying):
        """Second attacker (our usual defender) while the first is held up at the
        border by a ghost we can see too: one ghost covers both of us there. Commit
        to a crossing far from the teammate and that ghost, and walk to it along our
        side before looking at food again. Returns the next cell, or None."""
        spread = self.p('wait_spread', 0)
        maze = self.maze
        if pos not in maze.home_cells or carrying:
            self.relocating = None
            return None
        now = gs.data.timeleft
        target = getattr(self, 'relocating', None)
        if target is not None:
            cell, until = target
            if now > until and maze.dist(pos, cell) > 1:
                return self.route_home_side(pos, cell)
            self.relocating = None
            return None
        if not spread or self.default_role != 'defend':
            return None
        if self.ctx.roles.get(self.teammate) != 'attack' or not ghost_pos:
            return None
        mate = as_cell(gs.get_agent_position(self.teammate))
        if mate not in maze.home_cells or maze.dist(pos, mate) > self.p('relocate_near', 6):
            return None
        cell = max(maze.border, key=lambda b: min(maze.dist(g, b) for g in ghost_pos) +
                   spread * min(maze.dist(mate, b), 15))
        if maze.dist(pos, cell) <= 3:
            return None  # nowhere better to go
        self.relocating = (cell, now - 4 * self.p('relocate_moves', 30))
        return self.route_home_side(pos, cell)

    def border_wait_target(self, gs, pos, ghost_pos):
        maze = self.maze
        if not ghost_pos:
            return maze.border[len(maze.border) // 2]

        # Two attackers waiting at the same crossing are covered by one ghost: the
        # second one (our usual defender) waits far from its teammate instead, so the
        # ghost has to choose.
        spread, mate = self.p('wait_spread', 0), None
        if spread and self.default_role == 'defend' and self.ctx.roles.get(self.teammate) == 'attack':
            mate = as_cell(gs.get_agent_position(self.teammate))
            if mate not in maze.home_cells:
                mate = None

        def val(b):
            v = -min(maze.dist(g, b) for g in ghost_pos)
            if mate is not None:
                v -= spread * min(maze.dist(mate, b), 15)
            return v
        target = min(maze.border, key=val)
        cost, steps, parent = dijkstra(maze.nbrs, pos, lambda c, s: 0.0 if c in maze.home_cells else INF)
        if target in parent:
            return first_step(parent, pos, target)
        return None

    # ---- defend --------------------------------------------------------
    def defend_target(self, gs, pos):
        maze = self.maze
        me = gs.get_agent_state(self.index)
        invaders = self.invaders(gs)
        scared = me.scared_timer > 0

        if invaders:
            if scared:
                # keep 2 steps away from the nearest one until the scare wears off
                target_inv = min((p for _, p in invaders), key=lambda p: maze.dist(pos, p))
                options = [n for n in maze.nbrs[pos] + [pos] if n in maze.home_cells] or maze.nbrs[pos] + [pos]
                return min(options, key=lambda n: (abs(maze.dist(n, target_inv) - 2), maze.dist(n, target_inv) < 2))
            step = self.forced_capture_step(pos, invaders)
            if step is None:
                step = self.lock_in_step(gs, pos, invaders)
            if step is not None:
                return step
            return self.pursuit_step(gs, pos, invaders)

        # hidden invaders: go to the most likely position
        hidden = [e for e in self.ctx.enemies if gs.get_agent_state(e).is_pacman]
        if hidden:
            goal = None
            if self.ctx.last_eaten and self.ctx.last_eaten[1] - gs.data.timeleft <= 8:
                goal = self.ctx.last_eaten[0]
            if goal is None:
                tracker = self.ctx.tracker
                # chase the hidden invader we are most likely closest to
                e = min(hidden, key=lambda h: tracker.expected_dist(h, pos))
                goal = tracker.mode(e)
            return self.route_home_side(pos, goal)

        return self.route_home_side(pos, self.patrol_point(gs, pos))

    def forced_capture_step(self, pos, invaders):
        """If the chase table says we can force catching a visible invader on our
        own, play the fastest capture."""
        chase = self.ctx.chase_home
        if chase is None:
            return None
        best = None
        for _, ip in invaders:
            t = chase.ghost_to_move(ip, pos)
            if t is not None and (best is None or t < best[0]):
                best = (t, ip)
        if best is None:
            return None
        self.goal = best[1]
        return chase.ghost_step(best[1], pos)

    def lock_in_step(self, gs, pos, invaders):
        """Food only counts once it is carried home. If we can take a step after
        which an invader can never force its way out (to the border or to a
        capsule), take it and keep it locked in, closing in as far as that allows.
        With several invaders, lock in the one carrying the most."""
        chase = self.ctx.chase_home
        if chase is None:
            return None
        maze = self.maze
        best = None
        heaviest = 0
        for e, ip in invaders:
            carried = gs.get_agent_state(e).num_carrying
            heaviest = max(heaviest, carried)
            steps = chase.sealing_steps(ip, pos) if ip in maze.home_cells else []
            if steps and (best is None or carried > best[0]):
                best = (carried, ip, steps)
        if best is None or best[0] + self.p('lock_slack', 2) < heaviest:
            return None
        _, ip, steps = best

        def key(n):
            t = chase.pacman_to_move(ip, n)
            return (0, t, 0) if t is not None else (1, maze.dist(n, ip), n == pos)
        step = min(steps, key=key)
        self.goal = ip
        return step

    def safe_step(self, gs, pos, target):
        """Attack: never step into a position from which one visible ghost can force
        catching us (chase table). If the planned step is lost, take the safe step
        closest to the goal; if every step is lost, the one that lasts longest."""
        chase = self.ctx.chase_away
        if chase is None:
            return target
        maze = self.maze
        ghosts = [g for _, g in self.dangerous_ghosts(gs) if g in maze.enemy_cells]
        if not ghosts:
            return target

        def lost_in(c):
            if c in maze.home_cells:
                return None
            times = [chase.ghost_to_move(c, g) for g in ghosts]
            times = [t for t in times if t is not None]
            return min(times) if times else None

        nxt = pos if target is None else target
        if lost_in(nxt) is None:
            return target
        options = maze.nbrs[pos] + [pos]
        safe = [n for n in options if lost_in(n) is None]
        if safe:
            goal = self.goal
            if goal is None or goal == nxt or goal in maze.home_cells:
                return min(safe, key=lambda n: (maze.home_dist[n], n != pos))
            return min(safe, key=lambda n: (maze.dist(n, goal), maze.home_dist[n]))
        return max(options, key=lambda n: lost_in(n))

    def pursuit_step(self, gs, pos, invaders):
        """Region-cutting pursuit. Running straight at an invader only pushes it
        ahead of us, so instead take the step that leaves it the least to gain:
        our food, capsules and border cells that it still reaches strictly before
        every defender (whoever gets to a cell first owns it: a defender can go
        and stand there). Each defender minimises this on its own turn with the
        teammate's position fixed, which makes two defenders close in from two
        sides. Once nothing is left to protect, the tie-break closes the distance;
        in a dead end that walks in along the only way out, which is a sure catch."""
        maze = self.maze
        if pos not in maze.home_cells:
            # called back from the other side: head for the nearest invader, which leads home
            nearest = min((p for _, p in invaders), key=lambda p: maze.dist(pos, p))
            return min(maze.nbrs[pos], key=lambda n: maze.dist(n, nearest))
        mates = []
        if self.ctx.roles.get(self.teammate) == 'defend':
            mate = as_cell(gs.get_agent_position(self.teammate))
            if mate in maze.home_cells and gs.get_agent_state(self.teammate).scared_timer == 0:
                mates.append(mate)
        inv_pos = [p for _, p in invaders]
        carried = {p: gs.get_agent_state(e).num_carrying for e, p in invaders}
        food = [as_cell(f) for f in self.ctx.defend_food]
        caps = [as_cell(c) for c in self.ctx.defend_capsules]
        soft = float(self.p('pursuit_soft', 3))
        margin = self.p('pursuit_margin', 1)  # being only just first to a cell is not safe yet
        cap_w, exit_w, chase_w = self.p('pursuit_cap', 8.0), self.p('pursuit_exit', 1.5), self.p('pursuit_chase', 0.03)

        def lead(cell, pursuers, inv):
            """How much earlier the invader reaches the cell than our closest defender (0..1)."""
            d = min(maze.dist(q, cell) for q in pursuers) - maze.dist(inv, cell) + margin
            return 0.0 if d <= 0 else min(d, soft) / soft

        def value(cand):
            pursuers = mates + [cand]
            v = chase_w * min(maze.dist(cand, i) for i in inv_pos)
            for f in food:
                v += max(lead(f, pursuers, i) for i in inv_pos)
            for c in caps:
                v += cap_w * max(lead(c, pursuers, i) for i in inv_pos)
            for i in inv_pos:
                # banking what it carries: it needs one border cell it reaches first
                v += exit_w * (carried[i] + 0.3) * max(lead(b, pursuers, i) for b in maze.border)
            return v

        options = [n for n in maze.nbrs[pos] if n in maze.home_cells] + [pos]
        best = min(options, key=lambda n: (round(value(n), 6), min(maze.dist(n, i) for i in inv_pos), n != pos))
        self.goal = best
        return best

    def patrol_point(self, gs, pos):
        """Border cell nearest to where the most threatening enemy will cross."""
        maze = self.maze
        tracker = self.ctx.tracker
        best_e, best_d = None, INF
        for e in self.ctx.enemies:
            d = sum(p * maze.home_dist[c] for c, p in tracker.belief[e].items())
            if d < best_d:
                best_e, best_d = e, d
        mode = tracker.mode(best_e)
        defend_food = list(self.ctx.defend_food)

        def val(b):
            v = maze.dist(mode, b)
            if defend_food and self.p('guard_food', True):
                v += 0.3 * min(maze.dist(b, f) for f in defend_food)
            return v
        return min(maze.border, key=val)

    def route_home_side(self, pos, goal):
        maze = self.maze
        if goal is None:
            return None
        if goal not in maze.home_cells:
            goal = min(maze.border, key=lambda b: maze.dist(b, goal))
        self.goal = goal
        cost, steps, parent = dijkstra(maze.nbrs, pos, lambda c, s: 0.0 if c in maze.home_cells else INF)
        if goal in parent:
            return first_step(parent, pos, goal)
        return min(maze.nbrs[pos], key=lambda n: maze.dist(n, goal))

    def fallback_action(self, gs, pos, legal):
        return Directions.STOP if Directions.STOP in legal else random.choice(legal)


#########################################################################
# Team creation                                                         #
#########################################################################

STRATEGIES = {
    'planner': (PlannerAgent, PlannerAgent),
}


def _parse_value(v):
    if isinstance(v, str):
        low = v.lower()
        if low in ('true', 'false'):
            return low == 'true'
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    return v


# The team that plays when no options are given (the contest passes none).
DEFAULT_STRATEGY = 'planner'
DEFAULT_PARAMS = {'unstick': True, 'risk_t1': 150, 'risk_t2': 60, 'capsule_first': True, 'carry_limit': 12, 'stall_window': 16, 'open_stall': 60, 'heavy_defend': 8, 'scared_raid': 10,
                  'breakout': 12, 'breakout_urgent': 6, 'breakout_lead': 0, 'stall_risk': 60, 'quiet_hop': 2,
                  'escape_spread': 10}


def create_team(first_index, second_index, is_red, strategy=None, num_training=0, **params):
    merged = dict(DEFAULT_PARAMS)
    merged.update({k: _parse_value(v) for k, v in params.items()})
    params = merged
    first_cls, second_cls = STRATEGIES[strategy or DEFAULT_STRATEGY]
    return [first_cls(first_index, role='attack', **params),
            second_cls(second_index, role='defend', **params)]
