"""Opt-in exact rule-ledger assistance for Query 40; no external model calls."""

from collections import defaultdict
import hashlib
import json
import random
import re

LITERAL = re.compile(r"!?[a-z][a-z0-9_]*$")

SEMANTICS = """Use only the ledger. Atom names are complete symbolic facts; infer no additional real-world relationships.
An exclamation mark denotes explicit negation. Missing support for an atom does not establish its negation.
Apply u-lines in increasing numeric order to the base facts, then recompute all derived conclusions from scratch. REMOVE deletes exactly the stated literal and adds no opposite literal. ADD establishes exactly its literal. Other facts remain unchanged.
The graph of rule and priority-condition dependencies is acyclic. Resolve every prerequisite before its dependent atom. A rule provides applicable support only when all premises are established with their written polarities. Multiple rules can independently support the same literal. A blocked intermediate cannot satisfy a later premise.
If only one polarity of an atom has applicable support, establish that polarity. If neither does, establish neither. If both have applicable support, consider that atom's p-lines in increasing numeric order. The first line whose conditions are all established selects its stated winner and blocks the opposite polarity. A priority acts only in a conflict and cannot itself create support. If no priority applies, block both polarities; neither is established and neither can propagate.
Supported means the QUERY literal is established. Negated means its explicit opposite is established. Unresolved means neither is established."""


def literal(text):
    text = text.strip()
    if not LITERAL.fullmatch(text):
        raise ValueError(f"Unsupported literal: {text!r}")
    return text.lstrip("!"), -1 if text.startswith("!") else 1


def parse(text):
    blocks = re.findall(r"BEGIN_LOGIC\s*\n(.*?)\n\s*END_LOGIC", text, re.S)
    if len(blocks) != 1:
        raise ValueError("Expected one BEGIN_LOGIC/END_LOGIC ledger")
    facts, rules, priorities, updates, query = set(), [], [], [], None
    ids = set()
    for line in blocks[0].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("FACTS:"):
            body = line.split(":", 1)[1].strip()
            if body and body != "none":
                facts.update(literal(x) for x in body.split(","))
        elif line.startswith("QUERY:"):
            if query is not None:
                raise ValueError("Duplicate query")
            query = literal(line.split(":", 1)[1])
        elif match := re.fullmatch(r"(r\d+):\s*(.+?)\s*->\s*(\S+)", line):
            key, body, head = match.groups()
            if key in ids:
                raise ValueError("Duplicate rule/priority/update ID")
            ids.add(key)
            rules.append({"id": key, "body": [literal(x) for x in body.split("&")], "head": literal(head)})
        elif match := re.fullmatch(r"(p\d+):\s*(\S+)\s*>\s*(\S+)\s+IF\s+(.+)", line):
            key, winner, loser, body = match.groups()
            if key in ids:
                raise ValueError("Duplicate rule/priority/update ID")
            ids.add(key)
            winner, loser = literal(winner), literal(loser)
            if winner[0] != loser[0] or winner[1] == loser[1]:
                raise ValueError("A priority must compare opposite polarities of one atom")
            priorities.append({"id": key, "head": winner, "body": [literal(x) for x in body.split("&")]})
        elif match := re.fullmatch(r"(u\d+):\s*(ADD|REMOVE)\s+(\S+)", line):
            key, action, value = match.groups()
            if key in ids:
                raise ValueError("Duplicate rule/priority/update ID")
            ids.add(key)
            updates.append({"id": key, "action": action, "literal": literal(value)})
        else:
            raise ValueError(f"Unrecognized ledger line: {line}")
    if query is None or not rules:
        raise ValueError("Missing query or rules")
    heads = {rule["head"][0] for rule in rules}
    if {a for a, _ in facts}.intersection(heads):
        raise ValueError("Base facts and derived heads must be distinct")
    if any(update["literal"][0] in heads for update in updates):
        raise ValueError("Updates may change base facts only")
    dependencies = defaultdict(set)
    for record in rules + priorities:
        dependencies[record["head"][0]].update(a for a, _ in record["body"])
    ordered, visiting, visited = [], set(), set()

    def visit(atom):
        if atom in visiting:
            raise ValueError(f"Cyclic rule/priority dependency at {atom}")
        if atom in visited:
            return
        visiting.add(atom)
        for parent in sorted(dependencies[atom]):
            visit(parent)
        visiting.remove(atom)
        visited.add(atom)
        ordered.append(atom)

    for atom in sorted(heads | {query[0]}):
        visit(atom)
    return {"facts": facts, "rules": rules, "priorities": priorities, "updates": updates,
            "query": query, "ordered": ordered, "dependencies": dependencies}


def solve(program, *, omit_rule=None, omit_priority=None, omit_update=None):
    facts = set(program["facts"])
    for update in sorted(program["updates"], key=lambda u: int(u["id"][1:])):
        if update["id"] == omit_update:
            continue
        if update["action"] == "ADD":
            facts.add(update["literal"])
        else:
            facts.discard(update["literal"])
    for atom, polarity in facts:
        if (atom, -polarity) in facts:
            raise ValueError(f"Contradictory base facts after updates: {atom}")
    values = {atom: polarity for atom, polarity in facts}
    trace = {}
    rules = defaultdict(list)
    policies = defaultdict(list)
    for rule in program["rules"]:
        if rule["id"] != omit_rule:
            rules[rule["head"][0]].append(rule)
    for policy in sorted(program["priorities"], key=lambda p: int(p["id"][1:])):
        if policy["id"] != omit_priority:
            policies[policy["head"][0]].append(policy)
    for atom in program["ordered"]:
        if atom not in rules:
            values.setdefault(atom, 0)
            continue
        supports = {1: [], -1: []}
        for rule in rules[atom]:
            if all(values.get(parent, 0) == polarity for parent, polarity in rule["body"]):
                supports[rule["head"][1]].append(rule["id"])
        chosen = None
        if supports[1] and supports[-1]:
            for policy in policies[atom]:
                if all(values.get(parent, 0) == polarity for parent, polarity in policy["body"]):
                    chosen = policy
                    break
            values[atom] = chosen["head"][1] if chosen else 0
        else:
            values[atom] = 1 if supports[1] else -1 if supports[-1] else 0
        trace[atom] = {"positive_rules": supports[1], "negative_rules": supports[-1],
                       "priority": chosen["id"] if chosen else None, "value": values[atom]}
    result = values.get(program["query"][0], 0) * program["query"][1]
    return result, values, trace



def make_ledger(seed, layers, updates, unresolved=False):
    rng = random.Random(seed)
    roots = [f"b{i}" for i in range(1, 9)]
    signs = {atom: rng.choice((1, -1)) for atom in roots}
    facts = ["base"] + [("!" if signs[a] < 0 else "") + a for a in roots]
    rules, groups, previous = [], [], roots
    counter = 0
    for layer in range(layers):
        current = []
        for column in range(4):
            counter += 1
            head = f"g{counter}"
            current.append(head)
            a, b, c = rng.sample(previous, 3)
            family = rng.choice(("xor", "mux", "majority", "equivalence", "ordered"))
            winner = rng.choice((head, "!" + head))
            loser = head if winner.startswith("!") else "!" + head
            if family == "xor":
                conditions = [(f"{a} & !{b}", winner), (f"!{a} & {b}", winner)]
            elif family == "mux":
                conditions = [(f"{a} & {b}", winner), (f"!{a} & {c}", winner)]
            elif family == "majority":
                conditions = [(f"{a} & {b}", winner), (f"{a} & {c}", winner), (f"{b} & {c}", winner)]
            elif family == "equivalence":
                conditions = [(f"{a} & {b}", winner), (f"!{a} & !{b}", winner)]
            else:
                conditions = [(a, winner), (f"{b} & {c}", loser), (b, winner)]
            if not (unresolved and layer == layers - 1 and column == 3):
                conditions.append(("base", loser))
            groups.append((head, conditions))
            rules.extend([(head, "base"), ("!" + head, "base")])
        previous = current
    rule_ids = list(range(1, len(rules) + 1))
    rng.shuffle(rule_ids)
    lines = [f"r{key}: {body} -> {head}" for key, (head, body) in zip(rule_ids, rules)]
    policy_count = sum(len(conditions) for _, conditions in groups)
    policy_ids = list(range(1, policy_count + 1))
    rng.shuffle(policy_ids)
    cursor = 0
    for head, conditions in groups:
        ids = sorted(policy_ids[cursor:cursor + len(conditions)])
        cursor += len(conditions)
        for key, (body, winner) in zip(ids, conditions):
            loser = head if winner.startswith("!") else "!" + head
            lines.append(f"p{key}: {winner} > {loser} IF {body}")
    rng.shuffle(lines)
    if updates:
        selected = rng.sample(roots, 4)
        pending = []
        for i, atom in enumerate(selected):
            old = ("!" if signs[atom] < 0 else "") + atom
            pending.append(("REMOVE", old))
            if i != 3:
                new = ("" if signs[atom] < 0 else "!") + atom
                pending.append(("ADD", new))
        # Distinct scopes can be interleaved; order within each replacement is preserved.
        for i, (action, atom) in enumerate(pending, 1):
            lines.append(f"u{i}: {action} {atom}")
    query = previous[-1]
    if rng.choice((True, False)):
        query = "!" + query
    return "BEGIN_LOGIC\nFACTS: " + ", ".join(facts) + "\n" + "\n".join(lines) + "\nQUERY: " + query + "\nEND_LOGIC"


ASSISTANCE_TYPE = 'query40_priority_ledger_v1'
TASK_ID = 'text_rule_conflicts_exceptions_and_conclusion_revision_v1'
FIXTURE_MARKER = 'fixture:rule_ledger'


def validate_assistance(value, benchmark_task_id):
    """Allow one explicit local constructor, never arbitrary code from guidance."""
    if benchmark_task_id != TASK_ID:
        raise ValueError('Rule-ledger construction assistance requires the Query 40 task ID')
    if not isinstance(value, dict) or set(value) - {'type', 'layers', 'updates', 'max_prompt_chars'}:
        raise ValueError('Invalid rule-ledger construction settings')
    if value.get('type') != ASSISTANCE_TYPE:
        raise ValueError('Unknown construction assistance type')
    layers = value.get('layers', 12)
    if type(layers) is not int or not 2 <= layers <= 16:
        raise ValueError('Rule-ledger layers must be an integer between 2 and 16')
    if type(value.get('updates')) is not bool:
        raise ValueError('Rule-ledger updates must be explicitly true or false')
    limit = value.get('max_prompt_chars', 80000)
    if type(limit) is not int or not 30000 <= limit <= 120000:
        raise ValueError('Rule-ledger prompt limit must be between 30000 and 120000 characters')
    return {'type': ASSISTANCE_TYPE, 'layers': layers, 'updates': value['updates'], 'max_prompt_chars': limit}


def build_fixture(settings, item_key):
    """Generate and replay once, before authoring, independently of target answers."""
    encoded = json.dumps(item_key, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    start = 5000000 + int(digest[:12], 16) * 1000
    desired = (1, -1, 0)[int(digest[12:16], 16) % 3]
    query_sign = 1 if int(digest[16:20], 16) % 2 == 0 else -1
    for trial in range(1000):
        seed = start + trial
        ledger = make_ledger(seed, settings['layers'], settings['updates'], unresolved=desired == 0)
        program = parse(ledger)
        if program['query'][1] != query_sign:
            continue
        result, values, trace = solve(program)
        if result != desired:
            continue
        influential = []
        for update in program['updates']:
            try:
                if solve(program, omit_update=update['id'])[0] != result:
                    influential.append(update['id'])
            except ValueError:
                pass  # Omitting only REMOVE from a correction can contradict the base.
        if settings['updates'] and not influential:
            continue
        query_atom, query_polarity = program['query']
        final_policies = []
        for policy in sorted(program['priorities'], key=lambda item: int(item['id'][1:])):
            if policy['head'][0] == query_atom:
                conditions = [{'literal': ('!' if sign < 0 else '') + atom,
                               'established': values.get(atom, 0) == sign}
                              for atom, sign in policy['body']]
                final_policies.append({'id': policy['id'],
                    'winner': ('!' if policy['head'][1] < 0 else '') + query_atom,
                    'conditions': conditions, 'applicable': all(item['established'] for item in conditions)})
        return {'type': ASSISTANCE_TYPE, 'seed': seed, 'layers': settings['layers'], 'width': 4,
                'semantics': SEMANTICS, 'ledger': ledger,
                'query': ('!' if query_polarity < 0 else '') + query_atom,
                'reference_status': {1: 'Supported', -1: 'Negated', 0: 'Unresolved'}[result],
                'reference_values': values, 'influential_updates': influential,
                'checked_derived_atoms': len(trace),
                'query_audit': {'visibility': 'Private machine audit, NOT facts or hints supplied to the solver',
                    'query_is_base_fact': any(atom == query_atom for atom, _ in program['facts']),
                    'positive_support_rules': trace[query_atom]['positive_rules'],
                    'negative_support_rules': trace[query_atom]['negative_rules'],
                    'ordered_priorities': final_policies,
                    'first_applicable_priority': trace[query_atom]['priority'],
                    'established_atom_polarity': values[query_atom],
                    'influential_updates': influential,
                    'semantic_check': 'Direct support is not establishment when opposite support also applies. A later unconditional priority is a fallback only after all earlier numeric guards fail. The private audit is not part of solver-visible FACTS.'},
                'provenance': 'A fictional rule fixture created inside the main Executor before any target-model evaluation; original source facts remain separate.'}
    raise ValueError('Could not construct the requested rule-ledger fixture')


def fixture_errors(candidate, fixture):
    """Recheck the actual final candidate after every edit and choice permutation."""
    try:
        context = candidate['input']['context']
        question = candidate['input']['question']
        if not isinstance(context, str) or not isinstance(question, str):
            return ['Rule-ledger context and question must be text']
        blocks = re.findall(r'BEGIN_LOGIC\s*\n(.*?)\n\s*END_LOGIC', context, re.S)
        expected = re.findall(r'BEGIN_LOGIC\s*\n(.*?)\n\s*END_LOGIC', fixture['ledger'], re.S)[0]
        if len(blocks) != 1 or blocks[0].strip() != expected.strip():
            return ['Preserve the construction_fixture ledger verbatim; change only presentation outside its block']
        if fixture['semantics'] not in context:
            return ['Include construction_fixture.semantics verbatim in visible context']
        result = solve(parse(context))[0]
        status = {1: 'Supported', -1: 'Negated', 0: 'Unresolved'}[result]
        options = re.findall(r'(?mi)^\s*([A-E])[.)]\s*(Supported|Negated|Unresolved)\s*$', question)
        if len(options) != 3 or {name.lower() for _, name in options} != {'supported', 'negated', 'unresolved'}:
            return ['Use exactly three separate labeled option lines: Supported, Negated, Unresolved']
        labels = [label.upper() for label, name in options if name.lower() == status.lower()]
        if candidate['output']['answer'] != labels[0]:
            return [f'The replayed QUERY status is {status}; the gold must match its current option {labels[0]}']
        return []
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        return [f'Rule-ledger reference check failed: {exc}']


def bind_fixture(candidate, fixture):
    """Expand one explicit author-side marker before the ordinary content review."""
    context = candidate.get('input', {}).get('context') if isinstance(candidate, dict) else None
    if isinstance(context, str) and context.count(FIXTURE_MARKER) == 1:
        candidate['input']['context'] = context.replace(FIXTURE_MARKER,
            fixture['semantics'] + '\n\n' + fixture['ledger'])
    return candidate


