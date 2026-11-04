import random


def make_split(ridb, n_enrol_people=15, seed=42):
    """Enrol: image 1 of 15 people. Probes: their images 2-5.
    Impostors: all 5 images of the remaining 5 people (never enrolled)."""
    people = sorted(ridb)
    random.Random(seed).shuffle(people)
    enrolled, unknown = people[:n_enrol_people], people[n_enrol_people:]

    enrol = {p: ridb[p][0][1] for p in enrolled}
    probes = [(p, im) for p in enrolled for _, im in ridb[p][1:]]
    impostors = [(p, im) for p in unknown for _, im in ridb[p]]
    return enrol, probes, impostors