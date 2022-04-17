import torch

NUM_ACTOR_CLASSES = 26
NUM_OBJECT_CLASSES = 125
NUM_REL_CLASSES = 19
NUM_ATT_CLASSES = 4
NUM_TA_CLASSES = 33
NUM_IA_CLASSES = 9

class bidict(dict):
  """
  A many-to-one bidirectional dictionary
  Reference: https://stackoverflow.com/questions/3318625/how-to-implement-an-efficient-bidirectional-hash-table
  """
  def __init__(self, *args, **kwargs):
    super(bidict, self).__init__(*args, **kwargs)
    self.inverse = {}
    for key, value in self.items():
      self.inverse.setdefault(value, set()).add(key)

  def __setitem__(self, key, value):
    if key in self:
      self.inverse[self[key]].remove(key)
    super(bidict, self).__setitem__(key, value)
    self.inverse.setdefault(value, set()).add(key)

  def __delitem__(self, key):
    self.inverse[self[key]].remove(key)
    if len(self.inverse[self[key]]) == 0:
      del self.inverse[self[key]]
    super(bidict, self).__delitem__(key)


class Metadatum:
  def __init__(self, ann):
    self.fname = ann['file_name']
    self.num_frames = ann['num_frames']
    self.width = ann['width']
    self.height = ann['height']
    self.duration = ann['duration']

  def get_fid(self, time):
    """ Get the frame ID given a timestamp in seconds
    """
    fps = (self.num_frames-1)/self.duration
    fid = time*fps
    return fid

  def get_time(self, fid):
    raise NotImplementedError

  def __repr__(self):
    return f'Metadatum(fname={self.fname}, size=({self.num_frames}, {self.height}, {self.width}, 3), ' \
           f'duration={self.duration}'


class Act:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.cname = ann['class_name']
    self.cid = taxonomy.index(self.cname)
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_sact = [x['id'] for x in ann['sub_activities']]

  def __repr__(self):
    return f'Act(id={self.id}, cname={self.cname}, time=[{self.start}, end={self.end}), num_sacts={len(self.ids_sact)}'


class SAct:
  def __init__(self, ann, taxonomy):
    self.id = ann['id']
    self.cname = ann['class_name']
    self.cid = taxonomy.index(self.cname)
    self.start = ann['start_time']
    self.end = ann['end_time']
    self.ids_hoi = [x['id'] for x in ann['higher_order_interactions']]

    # unique entity instances in this sub-activity
    info_actor = set([(y['id'], y['class_name']) for x in ann['higher_order_interactions'] for y in x['actors']])
    info_object = set([(y['id'], y['class_name']) for x in ann['higher_order_interactions'] for y in x['objects']])
    id_actor_to_cname_actor = dict(info_actor)
    id_object_to_cname_object = dict(info_object)
    self.ids_actor = sorted(id_actor_to_cname_actor.keys())
    self.ids_object = sorted(id_object_to_cname_object.keys(), key=int)
    self.__id_entity_to_cname_entity = id_actor_to_cname_actor|id_object_to_cname_object

  def get_cname_entity(self, id_entity):
    return self.__id_entity_to_cname_entity[id_entity]

  def __repr__(self):
    return f'SAct(id={self.id}, cname={self.cname}, time=[{self.start}, end={self.end}), num_hois={len(self.ids_hoi)})'


class HOI:
  def __init__(self, ann, taxonomy_actor, taxonomy_object, taxonomy_ia, taxonomy_ta, taxonomy_att, taxonomy_rel):
    self.id = ann['id']
    self.time = ann['time']
    self.actors = [Entity(x, 'actor', taxonomy_actor) for x in ann['actors']]
    self.objects = [Entity(x, 'object', taxonomy_object) for x in ann['objects']]
    self.ias = [Predicate(x, 'ia', taxonomy_ia) for x in ann['intransitive_actions']]
    self.tas = [Predicate(x, 'ta', taxonomy_ta) for x in ann['transitive_actions']]
    self.atts = [Predicate(x, 'att', taxonomy_att) for x in ann['attributes']]
    self.rels = [Predicate(x, 'rel', taxonomy_rel) for x in ann['relationships']]
    self.orc_node_attr, self.orc_edge_index, self.orc_edge_attr, self.num_nodes, self.node_map = self.make_graph()

  def make_graph(self):
    # orc_node_attr: [actor/object id one hot] length = NUM_ACTOR_CLASSES + NUM_OBJECT_CLASSES
    node_map = {}
    node_attr = []
    for actor in self.actors:
      node_map[actor.id] = len(node_attr)
      feat = torch.zeros(NUM_ACTOR_CLASSES + NUM_OBJECT_CLASSES)
      feat[actor.cid] = 1
      node_attr.append(feat)
    for object in self.objects:
      node_map[object.id] = len(node_attr)
      feat = torch.zeros(NUM_ACTOR_CLASSES + NUM_OBJECT_CLASSES)
      feat[NUM_ACTOR_CLASSES + object.cid] = 1
      node_attr.append(feat)

    # edge_index: dimension = [2, num_edges]
    # edge_attr: [ia/ta/att/rel one hot] length = NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + NUM_REL_CLASSES
    edge_index = []
    edge_attr = []
    for ia in self.ias:
      feat = torch.zeros(NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + NUM_REL_CLASSES)
      feat[ia.cid] = 1
      edge_index.append([node_map[ia.id_src], node_map[ia.id_src]])
      edge_attr.append(feat)
    for ta in self.tas:
      feat = torch.zeros(NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + NUM_REL_CLASSES)
      feat[NUM_IA_CLASSES + ta.cid] = 1
      edge_index.append([node_map[ta.id_src], node_map[ta.id_trg]])
      edge_attr.append(feat)
    for att in self.atts:
      feat = torch.zeros(NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + NUM_REL_CLASSES)
      feat[NUM_IA_CLASSES + NUM_TA_CLASSES + att.cid] = 1
      edge_index.append([node_map[att.id_src], node_map[att.id_src]])
      edge_attr.append(feat)
    for rel in self.rels:
      feat = torch.zeros(NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + NUM_REL_CLASSES)
      feat[NUM_IA_CLASSES + NUM_TA_CLASSES + NUM_ATT_CLASSES + rel.cid] = 1
      edge_index.append([node_map[rel.id_src], node_map[rel.id_trg]])
      edge_attr.append(feat)

    if len(node_attr) > 0:
      node_attr = torch.stack(node_attr)
    else:
      node_attr = torch.tensor(node_attr)
    if len(edge_attr) > 0:
      edge_attr = torch.stack(edge_attr)
    else:
      edge_attr = torch.tensor(edge_attr)

    return node_attr, torch.LongTensor(edge_index).reshape(2, -1), edge_attr, len(node_map), node_map

  @property
  def ids_actor(self):
    return sorted([actor.id for actor in self.actors])

  @property
  def ids_object(self):
    return sorted([object.id for object in self.objects], key=int)

  def __repr__(self):
    return f'HOI(id={self.id}, time={self.time}, ' \
           f'num_actors={len(self.actors)}, num_objects={len(self.objects)}, ' \
           f'num_ias={len(self.ias)}, num_tas={len(self.tas)}, ' \
           f'num_atts={len(self.atts)}, num_rels={len(self.rels)}, ' \
           f'ids_actor={self.ids_actor}, ids_object={self.ids_object})'


class BBox:
  def __init__(self, ann):
    self.x, self.y, self.width, self.height = ann

  @property
  def x1(self):
      return self.x

  @property
  def y1(self):
      return self.y

  @property
  def x2(self):
      return self.x+self.width

  @property
  def y2(self):
      return self.y+self.height

  def __repr__(self):
    return f'BBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})'


class Entity:
  def __init__(self, ann, kind, taxonomy):
    self.id = ann['id']  # local instance ID
    self.kind = kind
    self.cname = ann['class_name']
    self.cid = taxonomy.index(self.cname)
    self.bbox = BBox(ann['bbox'])

  def __repr__(self):
    name = ''.join(x.capitalize() for x in self.kind.split('_'))
    return f'{name}(id={self.id}, cname={self.cname})'


class Predicate:
  def __init__(self, ann, kind, taxonomy):
    is_binary = 'target_id' in ann
    self.kind = kind
    self.signature = {x[0]:(x[1:] if is_binary else x[1]) for x in taxonomy}[ann['class_name']]
    self.cname = ann['class_name']
    self.cid = [x[0] for x in taxonomy].index(self.cname)
    self.id_src = ann['source_id']
    self.id_trg = ann['target_id'] if is_binary else None

  def __repr__(self):
    name = ''.join(x.capitalize() for x in self.kind.split('_'))
    id = f'{self.id_src}' if self.id_trg is None else f'{self.id_src} -> {self.id_trg}'
    return f'{name}(id={id}, cname={self.cname})'
