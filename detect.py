import streets
import itertools

test_input = """
what day is alternate side of the street on 19th ave and 74th and 73rd Brooklyn ny 11204
"""

def tokenize(s) :
    return s.lower().split()

def identify_some_borough(toks) :
    poss = set()
    for tok in toks :
        poss = poss.union(streets.Borough.from_keyword(tok))
    return poss

class TokenOptions(object) :
    """For fuzzy matching, we can expand things like abbreviations to aid matching."""
    def get(self, t) :
        raise NotImplemented

class IdentityTokenOptions(TokenOptions) :
    def get(self, t) :
        yield (1, t)

class StreetNumberTokenOptions(TokenOptions) :
    numbers = "0123456789"
    suffixes = ["st", "nd", "rd", "th"]
    def is_number(self, s) :
        return len(s) > 0 and all(c in self.numbers for c in s)
    def get(self, t) :
        for suffix in self.suffixes :
            if t.endswith(suffix) and self.is_number(t[:-len(suffix)]) :
                yield (0.95, t[:-len(suffix)])

class AbbreviationTokenOptions(TokenOptions) :
    abbrevs = {"ave" : ["avenue"],
               "st" : ["street"],
               "n" : ["north"],
               "e" : ["east"],
               "s" : ["south"],
               "w" : ["west"]}
    def get(self, t) :
        for longer in self.abbrevs.get(t, []) :
            yield (0.8, longer)

token_options = [IdentityTokenOptions(), StreetNumberTokenOptions(), AbbreviationTokenOptions()]

def get_token_options(t) :
    for to in token_options :
        for opt in to.get(t) :
            yield opt

def get_all_token_options(toks) :
    return [list(get_token_options(t)) for t in toks]

def streets_for_token(t) :
    return streets.Street.keywords_to_streets.get(t, set())

# (List((score, streets)), List((score, streets))) -> List((score, streets))
def join_poss(last_poss, opts) :
    next_poss = []
    for lp in last_poss :
        for o in opts :
            no = (lp[0] * o[0], lp[1] & o[1])
            if no[1] :
                next_poss.append(no)
    return next_poss

def detect_street_1(tok_opts, start) :
    joined = None
    for i in xrange(start, len(tok_opts)) :
        streets = [(opt[0], streets_for_token(opt[1])) for opt in tok_opts[i]]
        #streets = [(sc, ss) for sc, ss in streets if ss]
        if streets :
            if joined == None :
                joined = streets
            else :
                joined2 = join_poss(joined, streets)
                if not joined2 :
                    return joined
                joined = joined2
        if joined != None and not joined :
            return None
    # (end - start + 1) is a heuristic meaning "longer match is better"
    return [(sc * (end - start + 1), s) for sc, ss in joined if ss for s in ss]

def detect_street_all(tok_opts) :
    for i in xrange(0, len(tok_opts)) :
        for j in xrange(i, len(tok_opts)) :
            detected = detect_street_1(tok_opts, i, j)
            if detected :
                yield (i, j, detected)

# gets Dict(start, (end, streets))
def get_memozied_streets(tok_opts) :
    mem = {}
    for i, j, detected in detect_street_all(tok_opts) :
        mem.setdefault(i, []).append((j, detected))
    return mem

def get_poss_in_range(mem, start, end, remaining=3) :
    if start > end :
        yield []
    for i in xrange(start, end+1) :
        for end2, detected in mem.get(i, []) :
            yield [detected]
            for poss in get_poss_in_range(mem, end2+1, end, remaining-1) :
                if poss :
                    yield [detected] + poss

def filter_for_borough(score, borough, posses) :
    def filter_street(ss) :
        return [(score * sc, st) for sc, st in ss if st.borough == borough]
    def filter_poss(poss) :
        return [filter_street(ss) for ss in poss]
    return [filter_poss(poss) for poss in posses]

# assumes the designation of a borough means all of the streets are on one side of it
def with_boroughs(tok_opts) :
    mem = get_memozied_streets(tok_opts)
    all_posses = []
    for i in xrange(len(tok_opts)) :
        for score, keyword in tok_opts[i] :
            for borough in streets.Borough.from_keyword(keyword) :
                posses = list(itertools.chain(get_poss_in_range(mem, 0, i-1),
                                              get_poss_in_range(mem, i+1, len(tok_opts)-1)))
                filtered = filter_for_borough(5*score, borough, posses)
                all_posses.extend(filtered)
    all_posses.extend(get_poss_in_range(mem, 0, len(tok_opts)-1))
    return all_posses

# checks for proximity
def are_streets_sensible(ss) :
    a = None
    for s in ss :
        b = set(streets.get_related_streets(s) + [s])
        if a == None :
            a = b
        else :
            a = a & b
    if a :
        return True
    else :
        return False

def eval_poss(poss) :
    def simplify_streets(ss) :
        best = {}
        for score, st in ss :
            best[st] = max(score, best.get(st, 0))
        for st, score in best.iteritems() :
            yield (score, st)
    def compute_score(ss) :
        score = 0
        for s in ss :
            score += s[0]
        return score * len(ss)
    sposs = []
    for ss in poss :
        simp = list(simplify_streets(ss))
        if simp :
            sposs.append(simp)
    sposs2 = [(compute_score(ss), [s[1] for s in ss])
              for ss in itertools.product(*sposs) if are_streets_sensible([s[1] for s in ss])]
    return sposs2

def test(s) :
    toks = tokenize(s)
    tok_opts = get_all_token_options(toks)
    posses = with_boroughs(tok_opts)
    evaled = list(itertools.chain(*[eval_poss(poss) for poss in posses]))
    evaled.sort(key=lambda e:-e[0])
    return evaled[0]
    posses.sort(key=lambda poss:sum(p[0] for p in poss))
    return posses[0]

def detect_streets(toks) :
    tok_opts = get_all_token_options(toks)
    street_opts = [[(score, streets_for_token(s)) for score, s in opts] for opts in tok_opts]
    street_opts = [[(score, streets) for score, streets in opts if streets]
                   for opts in street_opts if opts]
    return street_opts
    poss = []
    curr_street_opts = None
    for opts in street_opts :
        if curr_street_opts == None :
            if len(opts) > 0 :
                curr_street_opts = [(score, streets) for score, streets in opts if streets]
        else :
            next_poss = join_poss(curr_street_opts, opts)
            if not next_poss :
                poss.append(curr_street_opts)
                curr_street_opts = None
    if curr_street_opts :
        poss.append(curr_street_opts)
    return poss

toks = tokenize(test_input)

print identify_some_borough(toks)
print get_all_token_options(toks)
