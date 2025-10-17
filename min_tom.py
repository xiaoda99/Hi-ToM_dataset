from dataclasses import dataclass
from typing import Set, List
from copy import deepcopy
import random
from collections import defaultdict

@dataclass
class Event:
    observers: Set[str]
    room: str

@dataclass
class MoveEvent(Event):
    agent: str
    object: str
    container: str

@dataclass
class OtherEvent(Event):
    agent: str
    object: str
    verb: str

@dataclass
class Observer:
    name: str
    actual: bool = True
    perceived_by_others: bool = True
    
    def __hash__(self): return hash(self.name)

@dataclass
class Room:
    observers: set[Observer]
    objects: dict[str, str | None]  # object -> container or None
    containers: set[str]

@dataclass
class AgentRoomState:
    entries: int = 0
    moves_since_entry: int = 0

class Set(set):
    def add(self, item):
        set.add(self, item)
        return self

    def get_by_name(self, name: str):
        return next((i for i in self if i.name == name), None)
    
    def remove_by_name(self, name: str):
        self -= {i for i in self if i.name == name}
        return self
    
    def update_by_name(self, name: str, **kwargs):
        for i in self:
            if i.name == name:
                for k, v in kwargs.items():
                    setattr(i, k, v)
                break
        return self
    
    def has(self, name: str, **kwargs) -> bool:
        return any(i.name == name and all(getattr(i, k, None) == v for k, v in kwargs.items()) for i in self)

    def get_all_names(self):
        return set(i.name for i in self)

    # def union(self, other):
    #     return Set(set.union(self, other))
        
    # def with_update_by_name(self, name: str, **kwargs):
    #     self.update_by_name(name, **kwargs)
    #     return self

# def safe_get(obj_list, index, field, default='Unknown'):
#     return getattr(obj_list[index], field) if obj_list else default

def log(message):
    if True: print(message)
    

verbs = ['likes', 'hate', 'knows']

class Engine:
    def __init__(self, agents, rooms=None, objects=None, containers=None, seed=None,
                peek_prob=0, distracted_prob=0, exit_without_move_prob=0.2, reenter_prob=0.0,
                allow_other_actions=False):
        # self.t = 0
        self.events: list[Event] = []
        self.agents = {agent: defaultdict(AgentRoomState) for agent in agents}
        self.default_room = 'room'
        self.rooms = {}
        self.last_move_event: dict[str, MoveEvent] = {}
        if rooms is not None:
            for room_name, spec in rooms.items():
                room_objects = spec.get('objects', [])
                room_containers = spec.get('containers', [])
                self.rooms[room_name] = Room(
                    observers=Set(),
                    objects={obj: None for obj in room_objects},
                    containers=set(room_containers),
                )
        else:
            self.rooms[self.default_room] = Room(
                observers=Set(),
                objects={obj: None for obj in objects},
                containers=set(containers),
            )
        self.peek_prob = peek_prob
        self.distracted_prob = distracted_prob
        self.exit_without_move_prob = exit_without_move_prob
        self.reenter_prob = reenter_prob
        self.allow_other_actions = allow_other_actions
        self.seed = seed
        self.rng = random.Random(seed)
        self.n_peeks = 0
        self.n_distractions = 0
        self.logs = []

    def log(self, message):
        self.logs.append(message)

    def enter(self, agent: str, room: str | None = None):
        room_key = room or self.default_room
        observers = self.rooms[room_key].observers
        assert agent in self.agents and not observers.has(agent)
        observers.add(Observer(agent))
        state = self.agents[agent][room_key]
        state.entries += 1
        state.moves_since_entry = 0
        self.log(f"{agent} entered the {room_key}.")
        # self.events.append(Event(self.t, 'enter', who))
        # self.t += 1

    def exit(self, agent: str, room: str | None = None):
        room_key = room or self.default_room
        observers = self.rooms[room_key].observers
        assert observers.has(agent)
        observers.remove_by_name(agent)
        self.log(f"{agent} exited the {room_key}.")
        # self.events.append(Event(self.t, 'exit', agent))
        # self.t += 1

    def move(self, agent: str, obj: str, container: str, room: str | None = None):
        room_key = room or self.default_room
        room = self.rooms[room_key]
        assert agent is None or room.observers.has(agent)
        assert obj in room.objects and container in room.containers
        current_container = room.objects.get(obj)
        assert current_container != container, f"{obj} is already in {container}"
        observers = deepcopy(room.observers)
        if agent is not None:
            insiders = room.observers.get_all_names()
            outsiders = set(self.agents.keys()) - insiders
            strs = []
            if self.rng.random() < self.peek_prob and len(outsiders) > 0:
                peek_agent = self.rng.choice(list(outsiders))
                observers.add(Observer(peek_agent, perceived_by_others=False))
                strs.append(f"{peek_agent}, unnoticed by anyone, peeked in from outside") # or "without anyone noticing"
                self.n_peeks += 1
            if self.rng.random() < self.distracted_prob and len(insiders) > 1:
                distracted_agent = self.rng.choice(list(insiders - {agent}))
                observers.update_by_name(distracted_agent, actual=False)
                strs.append(f"{distracted_agent}, unnoticed by anyone, briefly got distracted and missed what happened")
                self.n_distractions += 1
            prefix = "As " + " and ".join(self.rng.sample(strs, len(strs))) + ", " if strs else ""
            self.agents[agent][room_key].moves_since_entry += 1
        event = MoveEvent(observers, room_key, agent, obj, container)
        self.events.append(event)
        self.last_move_event[obj] = event
        self.log(f"{agent} put the {obj} into the {container}." if room.objects[obj] is None # the first move in setup
            else f"{prefix}{agent} moved the {obj} to the {container}.")
        room.objects[obj] = container
        # self.t += 1

    def other_action(self, agent: str, obj: str, verb: str, room: str | None = None):
        room_key = room or self.default_room
        room = self.rooms[room_key]
        assert obj in room.objects and verb in verbs
        observers = deepcopy(room.observers)
        self.events.append(OtherEvent(observers, room_key, agent, obj, verb))
        self.log(f"{agent} {verb} the {obj}.")

    def setup_room(self, room: str | None = None, init_observer_pct=2/3):
        print()
        room_key = room or self.default_room
        room = self.rooms[room_key]
        for agent in self.rng.sample(list(self.agents.keys()), int(round(init_observer_pct * len(self.agents)))):
            self.enter(agent, room=room_key)
        for obj in room.objects:
            observer = self.rng.choice(list(self.rooms[room_key].observers))
            target = self.rng.choice(list(room.containers))
            self.move(observer.name, obj, target, room=room_key)

    def legal_actions(self, room: str | None = None):
        actions = []
        room_key = room or self.default_room
        room = self.rooms[room_key]
        for agent in self.agents:
            state = self.agents[agent][room_key]
            if not room.observers.has(agent) and (state.entries == 0 or self.rng.random() < self.reenter_prob):
                actions.append(('enter', agent, room_key))
            elif room.observers.has(agent):
                if state.moves_since_entry > 0 or self.rng.random() < self.exit_without_move_prob:
                    actions.append(('exit', agent, room_key))
        for observer in room.observers:
            for obj in room.objects:
                if self.allow_other_actions:
                    for verb in verbs:
                        actions.append(('other_action', observer.name, obj, verb, room_key))
                event = self.last_move_event.get(obj)
                if event and event.room == room_key and event.observers.has(observer.name, actual=True): # an object can only be moved by an agent knowing its last location 
                    current_container = room.objects.get(obj)
                    for container in room.containers:
                        if container != current_container:
                            actions.append(('move', observer.name, obj, container, room_key))
        return actions

    def generate_chapter(self, steps=6, room: str | None = None):
        self.setup_room(room=room)
        for step in range(steps):
            legal_actions = self.legal_actions(room=room or self.default_room)
            # print(f'legal_actions at step {i}: {legal_actions}')
            if len(legal_actions) == 0: print(f'break at step {step}/{steps}'); break
            act, *args = self.rng.choice(legal_actions)
            getattr(self, act)(*args)

    def generate_story(self, steps=6):
        for room in self.rooms:
            self.generate_chapter(room=room, steps=steps)
        return '\n'.join(self.logs)
            
    def generate_QAs(self, filtered=True):
        QAs = {order: defaultdict(list) for order in range(3)}
        all_objects = {objects for room in self.rooms for objects in self.rooms[room].objects}

        order = 0
        for object in all_objects:
            # deliberately do not use self.last_move_event to avoid shortcut
            try: location = [event for event in self.events if isinstance(event, MoveEvent) and event.object == object][-1].container
            except (IndexError, AttributeError): location = 'Unknown'
            QAs[order][object].append((f"Where is the {object} really?", location))

        order = 1
        for object in all_objects:
            for agent in self.agents:
                # deliberately do not use self.last_move_event to avoid shortcut
                try: location = [event for event in self.events if isinstance(event, MoveEvent) and event.object == object and 
                                 event.observers.has(agent, actual=True)][-1].container
                except (IndexError, AttributeError): location = 'Unknown'
                QAs[order][object].append((f"Where does {agent} think the {object} is?", location))

        order = 2
        for object in all_objects:
            for agent in self.agents:
                for other_agent in self.agents:
                    if agent == other_agent:
                        continue
                    # deliberately do not use self.last_move_event to avoid shortcut
                    try: location = [event for event in self.events if isinstance(event, MoveEvent) and event.object == object and 
                                     event.observers.has(agent, actual=True) and 
                                     event.observers.has(other_agent, perceived_by_others=True)][-1].container
                    except (IndexError, AttributeError): location = 'Unknown'
                    QAs[order][object].append((f"Where does {agent} think {other_agent} thinks the {object} is?", location))
        return self.filter_QAs(QAs) if filtered else QAs

    def filter_QAs(self, QAs: dict[int, dict[str, list[tuple[str, str]]]], orders=[0, 1, 2]):
        for order in list(QAs.keys()):
            if order not in orders:
                del QAs[order]
        for order in orders:
            if order in QAs:
                for object in QAs[order]:
                    QAs[order][object] = [qa for qa in QAs[order][object] if qa[1] != 'Unknown']
        return QAs


def render_to_python_dsl(eng: Engine) -> str:
    lines = [
        "events = []",
        "observers = Set()",
        *[f"observers.add(Observer('{a}'))" for a in set(a for a in eng.agents)]
    ]
    # maintain who is “in the room” for parity with your minimal DSL
    active = set()
    for ev in eng.events:
        if ev.kind == 'enter':
            lines.append(f"observers.add(Observer('{ev.agent}'))  # enter")
            active.add(ev.agent)
        elif ev.kind == 'exit':
            lines.append(f"observers.remove_by_name('{ev.agent}')  # exit")
            active.discard(ev.agent)
        else:
            # build observers snapshot with flags
            snap = "observers.copy()"
            lines.append(f"events.append(MoveEvent(observers={snap}, agent='{ev.agent}', object='{ev.object}', container='{ev.container}'))")
            # now patch flags immutably (you can instead emit a constructor with flags if you prefer)
            for a in active:
                if a not in ev.seen_by:
                    lines.append(f"events[-1].observers.get_by_name('{a}').actual = False  # distracted")
                if a not in ev.perceived_present:
                    lines.append(f"events[-1].observers.get_by_name('{a}').perceived_by_others = False  # peek")
    return "\n".join(lines)

''' An example ExploreToM story:
Brody entered the conservation lab. 
Emily entered the conservation lab. 
Emily moved the small, antique pocket watch to the metal safe, which is also located in the conservation lab. 
Wyatt entered the conservation lab. 
Emily told privately to Jasmine that the small, antique pocket watch is in the metal safe. 
Wyatt moved the small, antique pocket watch to the wooden box, which is also located in the conservation lab. 
Brody told privately to Jasmine that the small, antique pocket watch is in the wooden box. 
Wyatt told out loud that the small, antique pocket watch is in the wooden box. 
While this action was happening, Jasmine witnessed this action in secret (and only this action); 
also, Brody got distracted and did not realize what happened, without anyone noticing the brief lack of attention, 
and going back to paying attention immediately after the action was finished.'''
