from world import World
import numpy as np
from min_tom1 import *
from eval1 import *
import json


@dataclass
class QAExample:
    order: int
    obj: str
    question: str
    expected: str


# 从文本文件中提取内容
world_paths = 'world_large.txt'
world = World()
world.load(world_paths)


# 随机提取角色、地点、容器、物品
def generate_world(seed, num_agents, num_locations):
    np_random = np.random.RandomState(seed)
    actors = world.get_actors()
    locations = world.get_locations()
    objects = world.get_objects()
    containers = world.get_containers()

    random_actors = np_random.choice(actors, size=num_agents, replace=False)
    random_locations = np_random.choice(locations, size=num_locations, replace=False)
    random_objects = np_random.choice(objects, size=num_locations, replace=False)
    random_containers = np_random.choice(containers, size=num_locations * 5, replace=False)
    return random_actors, random_locations, random_objects, random_containers


# 构建角色列表及地点包含容器及物品的字典
def generate_world_structure(seed, num_agents, num_locations):
    random_actors, random_locations, random_objects, random_containers = generate_world(seed, num_agents, num_locations)

    agents = list(random_actors)  # 角色列表

    rooms = {}
    num_locations = len(random_locations)
    object_chunks = np.array_split(random_objects, num_locations)
    container_chunks = np.array_split(random_containers, num_locations)

    for i, loc in enumerate(random_locations):
        objects_in_loc = list(object_chunks[i])  # 当前房间的物体列表
        containers_in_loc = list(container_chunks[i])  # 当前房间的容器列表

        rooms[loc] = {
            'objects': objects_in_loc,
            'containers': containers_in_loc
        }

    return agents, rooms


def flatten_qas(qas: Dict[int, Dict[str, List[tuple[str, str]]]]) -> List[QAExample]:
    """Convert the nested QA dictionary into a flat list."""
    flat: List[QAExample] = []
    for order, object_map in sorted(qas.items()):
        for obj, qa_list in object_map.items():
            for question, expected in qa_list:
                flat.append(QAExample(order=order, obj=obj, question=question, expected=expected))
    return flat


def gather_story_data(engine: Engine, steps: int, story_id: int) -> tuple[str, List[QAExample], Dict[str, int]]:
    story = engine.generate_story(steps=steps)
    qas_dict = engine.generate_QAs()
    qas = flatten_qas(qas_dict)
    metadata = {
        "story_id": story_id,
        "n_peeks": getattr(engine, "n_peeks", 0),
        "n_distractions": getattr(engine, "n_distractions", 0),
    }
    return story, qas, metadata


# 每种order的问题只保留一个
def select_random_per_order(seed, qas: List[QAExample]) -> List[QAExample]:
    np_random = np.random.RandomState(seed)
    order_0 = [qa for qa in qas if qa.order == 0]
    order_1 = [qa for qa in qas if qa.order == 1]
    order_2 = [qa for qa in qas if qa.order == 2]

    new_qas = []
    if order_0: new_qas.append(np_random.choice(order_0))
    if order_1: new_qas.append(np_random.choice(order_1))
    if order_2: new_qas.append(np_random.choice(order_2))
    return new_qas


# 生成故事
def generate_story(num_stories):
    flat_items: List[FlatResult] = []
    l = 1
    for story_id in range(1, num_stories + 1):
        agents, rooms = generate_world_structure(story_id + 100, 5, l)
        engine = Engine(agents, rooms, seed=story_id + 100, peek_prob=0., distracted_prob=0.,
                        exit_without_move_prob=0.2, allow_other_actions=True)
        story, qas, metadata = gather_story_data(engine, 10, story_id)
        print(
            f"Story {story_id}: {len(qas)} QA items, "
            f"{metadata['n_peeks']} peeks, {metadata['n_distractions']} distractions."
        )
        qas = select_random_per_order(story_id + 100, qas)
        for qa in qas:
            flat_items.append(
                FlatResult(
                    **metadata,
                    id=3 * (story_id - 1) + qa.order + 1,
                    story_length=l,
                    seed=engine.seed,
                    story_text=story,
                    model="",
                    qa_order=qa.order,
                    qa_object=qa.obj,
                    question=qa.question,
                    expected=qa.expected,
                )
            )
            # print(qa)
        if story_id % 20 == 0:
            l = l + 1
        print(story)
        # for item in flat_items:
        #     print(item)

    with open("ceshi.json", "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in flat_items], f, indent=2, ensure_ascii=False)


# r=get()
# for item in r:
#     print(item)
generate_story(20)
# a, b, c, d = create_world(1, 5, 3)
# print(a)
# print(b)
# print(c)
# print(d)
