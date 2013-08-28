import streets
import itertools
import re
import collections

def tokenize(s) :
    def whsep(sep, s) :
        return (" " + sep + " ").join(s.split(sep))
    return re.findall(r"[a-z0-9'/]+|[^a-z0-9'/\s]", s.lower())
    return whsep(".", whsep(",", s.lower())).split()

def get_streets(s) :
    return streets.Street.keywords_to_streets.get(s, set())

def get_all_streets(ss) :
    return reduce(lambda x, y: x.union(y), (get_streets(s) for s in ss), set())

def nice_street_list(streets) :
    """Just to print things out nicely for debugging."""
    return [s.borough.name + ": " + s.normalized_name() for s in streets]

class SpellChecker(object) :
    other_words = ["n", "north",
                   "e", "east",
                   "s", "south",
                   "w", "west",
                   "ave", "avenue",
                   "av", "avenue",
                   "st", "street",
                   "rd", "road",
                   "ct", "court",
                   "si", "staten", "island",
                   "n", "north",
                   "no", "north",
                   "e", "east",
                   "s", "south",
                   "w", "west",
                   "hts", "heights",
                   "manhattan", "bronx", "brooklyn", "queens",
                   "first", "second", "third",
                   "fourth",
                   "fifth",
                   "sixth",
                   "seventh",
                   "eighth",
                   "ninth",
                   "tenth",
                   "eleventh",
                   "twelfth",
                   "thirteenth",
                   "fourteenth",
                   "fifteenth",
                   "sixteenth",
                   "seventeenth",
                   "eighteenth",
                   "nineteenth",
                   "twentieth"]

    def __init__(self) :
        self.words = collections.defaultdict(lambda : 1)
        self.alphabet = 'abcdefghijklmnopqrstuvwxyz\''
        self.init_from_dict_words()
        self.init_from_Street()
        self.add_words(self.other_words)
    def init_from_Street(self) :
        for street in streets.Street.streets.itervalues() :
            words = [w for w in street.normalized_name().lower().split()
                     if all(c in self.alphabet for c in w)]
            self.add_words(words)
    def init_from_dict_words(self) :
        with open("/usr/share/dict/words", "r") as f :
            for line in f :
                self.add_words(line.split())
    def add_words(self, words) :
        for word in words :
            self.words[word] += 1
    def edits1(self, word) :
        splits = [(word[:i], word[i:]) for i in xrange(len(word) + 1)]
        deletes = [a + b[1:] for a, b in splits if b]
        transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
        replaces = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
        inserts = [a + c + b for a, b in splits for c in self.alphabet]
        return set(deletes + transposes + replaces + inserts)
    def known_edits2(self, word) :
        return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.words)
    def known(self, words) :
        return set(w for w in words if w in self.words)
    def corrections(self, word) :
        word = word.lower()
        if not all(c in self.alphabet for c in word) :
            return [word]
        candidates = (self.known([word])
                      or self.known(self.edits1(word))
                      or self.known_edits2(word)
                      or [word])
        best = []
        score = -1
        for c in candidates :
            if score == self.words[c] :
                best.append(c)
            elif score < self.words[c] :
                scoer = self.words[c]
                best = [c]
        return best

class TokenOptions(object) :
    """For fuzzy matching, we can expand things like abbreviations to aid matching."""
    def get(self, t) :
        raise NotImplemented

class IdentityTokenOptions(TokenOptions) :
    def get(self, t) :
        yield t

class StreetNumberTokenOptions(TokenOptions) :
    numbers = "0123456789"
    prefixes = {"w" : ["w", "west"],
                "n" : ["n", "north"],
                "e" : ["e", "east"],
                "s" : ["s", "south"],
                "" : [""]}
    suffixes = ["st", "nd", "rd", "th", ""]
    def is_number(self, s) :
        return len(s) > 0 and all(c in self.numbers for c in s)
    def get(self, t) :
        for prefix, exps in self.prefixes.iteritems() :
            for suffix in self.suffixes :
                end = -len(suffix) or len(t)
                if (prefix or suffix) and t.startswith(prefix) and t.endswith(suffix) and self.is_number(t[len(prefix):end]) :
                    for exp in exps :
                        yield (exp + " " + t[len(prefix):end]).strip()
                        if suffix == "rd" :
                            yield (exp + " " + t[len(prefix):end] + " road").strip()
                        elif suffix == "st" :
                            yield (exp + " " + t[len(prefix):end] + " street").strip()

class AbbreviationTokenOptions(TokenOptions) :
    abbrevs = [("ave", "avenue"),
               ("av", "avenue"),
               ("st", "street"),
               ("rd", "road"),
               ("ct", "court"),
               ("pl", "place"),
               ("si", "staten island"),
               ("n", "north"),
               ("no", "north"),
               ("e", "east"),
               ("s", "south"),
               ("w", "west"),
               ("hts", "heights"),
               ("first", "1"),
               ("second", "2"),
               ("third", "3"),
               ("fourth", "4"),
               ("fifth", "5"),
               ("sixth", "6"),
               ("seventh", "7"),
               ("eighth", "8"),
               ("ninth", "9"),
               ("tenth", "10"),
               ("eleventh", "11"),
               ("twelfth", "12"),
               ("thirteenth", "13"),
               ("fourteenth", "14"),
               ("fifteenth", "15"),
               ("sixteenth", "16"),
               ("seventeenth", "17"),
               ("eighteenth", "18"),
               ("nineteenth", "19"),
               ("twentieth", "20")]
    def get(self, t) :
        for s1, s2 in self.abbrevs :
            if t == s1 :
                yield s2
            elif t == s2 :
                yield s1

class SpellCheckTokenOptions(TokenOptions) :
    sc = SpellChecker()
    def get(self, t) :
        for c in self.sc.corrections(t) :
            if c != t :
                yield c

token_options = [IdentityTokenOptions(), StreetNumberTokenOptions(), AbbreviationTokenOptions(), SpellCheckTokenOptions()]

def get_token_options(t) :
    for to in token_options :
        for opt in to.get(t) :
            yield opt

def get_all_token_options(toks) :
    return set([" ".join(os)
                for os in itertools.product(*(list(get_token_options(t)) for t in toks))])

class StreetOpt(object) :
    def __init__(self, i, j, s, streets) :
        self.i = i
        self.j = j
        self.s = s
        self.streets = streets
    def __repr__(self) :
        return "StreetOpt(%r, %r, %r, len=%r)" % (self.i, self.j, self.s, len(self.streets))
    def dist(self, other) :
        return min(abs(other.i - self.j - 1), abs(self.i - other.j - 1))
    def text_intersects(self, other) :
        return self.i <= other.i <= self.j or self.i <= other.j <= self.j
    def empty(self) :
        return len(self.streets) == 0
    def with_borough(self, borough) :
        return StreetOpt(self.i, self.j, self.s,
                         set(s for s in self.streets if s.borough == borough))
    def boroughs(self) :
        return set(s.borough for s in self.streets)
    def make_singleton(self, street) :
        return StreetOpt(self.i, self.j, self.s, set([street]))
class BoroughOpt(object) :
    def __init__(self, i, j, s, borough) :
        self.i = i
        self.j = j
        self.s = s
        self.borough = borough
    def __repr__(self) :
        return "BoroughOpt(%r, %r, %r)" % (self.i, self.j, self.borough)
class AddressOpt(object) :
    def __init__(self, i, j, s, addresses) :
        self.i = i
        self.j = j
        self.s = s
        self.addresses = addresses
    def __repr__(self) :
        return "AddressOpt(%r, %r, len=%r)" % (self.i, self.j, len(self.addresses))

def get_all_possible_streets(s) :
    toks = tokenize(s)
    tokopts = [list(get_token_options(t)) for t in toks]
    def get_all_token_options(i, j) :
        return set([" ".join(os)
                    for os in itertools.product(*(tokopts[k] for k in xrange(i, j)))])
    poss = {}
    bposs = []
    aposs = []
    for i in xrange(0, len(toks)) :
        for j in xrange(i, len(toks)) :
            if j - i >= 4 : break # (based on: 5-or-more streets are silly looking)
            #opts = get_all_token_options(toks[i:j+1])
            opts = get_all_token_options(i, j+1)
            ss = get_all_streets(opts)
            if 0 < len(ss) <= 50 : # ARBITRARY
                poss.setdefault(i, {})[j] = StreetOpt(i, j, " ".join(toks[i:j+1]), ss)
            for opt in opts :
                for b in streets.Borough.from_keyword(opt) :
                    bposs.append(BoroughOpt(i, j, " ".join(toks[i:j+1]), b))
                addresses = streets.Address.from_keyword(opt)
                if addresses :
                    aposs.append(AddressOpt(i, j, " ".join(toks[i:j+1]), addresses))
    return poss, bposs, aposs


def print_poss(poss, bposs, aposs) :
    for i, poss2 in poss.iteritems() :
        for j, opts in poss2.iteritems() :
            print i, j, opts
    for poss in bposs :
        print poss
    for poss in aposs :
        print poss

def get_reasonable_3_groups(poss, bposs) :
    bound = max(poss.iterkeys()) if poss else -1
    def _get(start, num) :
        if num == 0 :
            yield []
        elif start > bound :
            pass
        else :
            for j, opt in poss.get(start, {}).iteritems() :
                for nextopts in _get(j+1, num-1) :
                    yield [opt] + nextopts
            for nextopts in _get(start + 1, num) :
                yield nextopts
    def having_small_dist(group) :
        return group[0].dist(group[1]) < 3 and group[1].dist(group[2]) < 3
    def with_borough(group, borough_opt) :
        if any(g.text_intersects(borough_opt) for g in group) :
            return None
        new = [g.with_borough(borough_opt.borough) for g in group]
        if any(g.empty() for g in new) :
            return None
        return new
    groups = list(group for group in _get(0, 3) if having_small_dist(group))
    boroughed = []
    maybe_has_addr = len(groups) > 0
    for group in groups :
        added_borough = False
        for bopt in bposs :
            bed = with_borough(group, bopt)
            if bed :
                boroughed.append(bed)
                added_borough = True
        if not added_borough :
            boroughed.append(group)
    def neighboring(s1, s2, s3) :
        return streets.are_colocated(s1, s2, s3)
    def regroup(group) :
        new = []
        for s1, s2, s3 in itertools.product(group[0].streets, group[1].streets, group[2].streets) :
            if s1 == s2 or s1 == s3 or s2 == s3 :
                continue
            if streets.are_colocated(s1, s2, s3) :
                new.append([group[0].make_singleton(s1),
                            group[1].make_singleton(s2),
                            group[2].make_singleton(s3)])
        return new
    regrouped = []
    for group in boroughed :
        new = regroup(group)
        if new :
            regrouped.extend(new)
    return regrouped, maybe_has_addr
    
def evaluate_group(group) :
    s1 = list(group[0].streets)[0]
    s2 = list(group[1].streets)[0]
    s3 = list(group[2].streets)[0]
    score = 1.0
    score = score * len(group[0].s) / len(s1.name)
    score = score * len(group[1].s) / len(s2.name)
    score = score * len(group[2].s) / len(s3.name)
    score = score / (group[0].dist(group[1]) + 1)
    score = score / (group[1].dist(group[2]) + 1)
    locs = []
    for loc in streets.get_locations(s1, s2, s3) :
        locs.append((score, loc, min(s.i for s in group), max(s.j for s in group)))
    return locs

def get_reasonable_addresses(poss, bposs, aposs) :
    bposs = [None] + bposs
    poss2 = [p for poss2 in poss.itervalues() for p in poss2.itervalues()]
    found = []
    possibly_missing_address = False
    for b in bposs :
        for p in poss2 :
            if b :
                if p.text_intersects(b) : continue
                p = p.with_borough(b)
                mini = min(p.i, b.i)
                maxj = max(p.j, b.j)
            else :
                mini = p.i
                maxj = p.j
            for s in p.streets :
                locs = streets.Location.mstreet_to_locations.get(s, [])
                if locs and len(p.s) >= 5 : # ARBITRARY
                    possibly_missing_address = True
                for a in aposs :
                    mini2 = min(mini, a.i)
                    maxj2 = max(maxj, a.j)
                    if p.text_intersects(a) : continue
                    for a2 in a.addresses :
                        alocs = set(l for l in locs if a2 in l.addresses)
                        if alocs :
                            found.append((a, a2, b, p, s, alocs, mini2, maxj2))
    best = []
    lbound = 1000
    hbound = -1
    score = -1
    for f in found :
        sc, mini, maxj = evaluate_address(f)
        if sc > score :
            score = sc
            best = list(f[5])
            lbound = mini
            hbound = maxj
        elif sc == score :
            best.extend(f[5])
            lbound = min(lbound, mini)
            hbound = max(hbound, maxj)
    return best, lbound, hbound, possibly_missing_address

def evaluate_address(res) :
    a, a2, b, p, s, alocs, mini, maxj = res
    score = 1.0
    score = score * len(a.s) / len(a2.num)
    score = score * len(p.s) / len(s.name)
    score = score / (p.dist(a) + 1)
    if b :
        score = score / (p.dist(b) + 1)
    return score, mini, maxj

def determine_address(s) :
    """Different from get_reasonable_addresses because it doesn't
    attempt to get a location; just try to get a substring which is an
    address."""
    tokens = tokenize(s)
    def has_house_number(i) :
        def is_house_number(j) :
            if j < 0 : return False
            tok = tokens[j]
            return re.match(r"\d{1,6}[\w]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty", tok)
        return (is_house_number(i-1) and -1) \
            or (is_house_number(i-2) and -2) \
            or (is_house_number(i-3) and -3)
    poss, bposs, aposs = get_all_possible_streets(s)
    maxlen = -1
    bestopts = []
    restrict_boroughs = False
    missing_house_number = False
    for i, poss2 in poss.iteritems() :
        for j, opt in poss2.iteritems() :
            opt2 = opt
            bopt = None
            for b in bposs :
                if opt.text_intersects(b) : continue
                opt3 = opt.with_borough(b.borough)
                if opt3.streets :
                    opt2 = opt3
                    bopt = b
            opt = opt2
            di = has_house_number(i)
            if not di :
                if len(opt.s) >= 5 : # ARBITRARY
                    missing_house_number = True
                continue
            if len(opt.boroughs()) > 1 :
                restrict_boroughs = True
                continue
            if len(opt.s) > maxlen :
                bestopts = [(opt, di, bopt)]
                maxlen = len(opt.s)
            elif len(opt.s) == maxlen :
                bestopts.append((opt, di, bopt))
    def tok_str(bestopt) :
        opt, di, bopt = bestopt
        start = min(opt.i+di, bopt.i)
        end = max(opt.j, bopt.j)
        return {"whole_address" : " ".join(tokens[start:end+1]),
                "borough" : list(opt.boroughs())[0].name,
                "poss_housenum" : " ".join(tokens[opt.i+di:opt.i]),
                "poss_streets" : list(set(s.normalized_name().split("*", 2)[0].strip()
                                          for s in opt.streets))}
    if bestopts :
        return {'result' : 'ok',
                'addresses' : [tok_str(b) for b in bestopts]}
    else :
        return {'result' : 'none',
                'needs_borough' : restrict_boroughs,
                'needs_house_number' : missing_house_number}

def determine_locations(s) :
    poss, bposs, aposs = get_all_possible_streets(s)
    groups, has_addr = get_reasonable_3_groups(poss, bposs)
    locs = set()
    for group in groups :
        locs.update(evaluate_group(group))
    best = []
    score = -1
    lbound = 1000
    hbound = -1
    for sc, loc, i, j in locs :
        if sc == score :
            best.append(loc)
            lbound = min(lbound, i)
            hbound = max(hbound, j)
        elif sc > score :
            score = sc
            best = [loc]
            lbound = i
            hbound = j
    if not best :
        best, lbound, hbound, has_addr = get_reasonable_addresses(poss, bposs, aposs)
    if not best :
        return {"result" : "none", "maybe" : has_addr}
    if len(set(b.borough for b in best)) > 1 :
        return {"result" : "boroughs",
                "question" : "Please specify a borough.",
                "state" : [b.id for b in best]}
    return {"result" : "ok",
            "locs" : [b.id for b in best],
            "substring" : " ".join(tokenize(s)[lbound:hbound+1])}

#def determine_address(s) :

def signs_for_locations(locs) :
    signs = []
    for locid in locs :
        loc = streets.Location.locations[locid]
        for sign in loc.signs :
            signs.append((loc.side, sign))
    return signs

def serial_comma(parts, conj="and", comma=",", none="nothing") :
    parts = list(parts)
    if len(parts) == 0 :
        return none
    elif len(parts) == 1 :
        return parts[0]
    elif len(parts) == 2 :
        return parts[0] + " " + conj + " " + parts[1]
    else :
        return (comma + " ").join(parts[:-1]) + comma + " " + conj + " " + parts[-1]

def sign_desc_for_locations(locs) :
    signs = {}
    for side, sign in signs_for_locations(locs) :
        signs.setdefault(side.name, set()).add(sign.parsed)
    if all(len(ss) == 1 for ss in signs.itervalues()) :
        parts = []
        for side, signs in signs.iteritems() :
            parts.append("%s on the %s side" % (list(signs)[0], side.lower()))
        return serial_comma(parts, none="no signs")
    else :
        parts = []
        for side, signs in signs.iteritems() :
            parts.append("%s on the %s side" % (serial_comma(signs, conj="or"), side.lower()))
        return serial_comma(parts, comma=";", none="no signs")

def get_sign_descs(s) :
    locs = determine_locations(s)
    if locs[0] == "ok" :
        return sign_desc_for_locations(locs[1])
    else :
        return None

test_inputs = ["""
what day is alternate side of the street on 19th ave and 74th and 73rd Brooklyn ny 11204
"""
,
"""
Ridge Boulevard and colonial road on 77th street what days are alternate parking
"""
,
"""
when is asp on W85th Street (between Columbus and Amsterdam)?
"""
,
"""
Are all alternate side parking rules in effect this week for Wash Hts + Inwood north of 181st street? If not how long will they be suspended? Thx.
"""
,
"""
made up: e13th between 3rd ave and 4th
"""
]

def test1(inp) :
    print
    print "Input:", inp.strip()
    print "Output:", get_sign_descs(inp)

def test() :
    for test_input in test_inputs :
        test1(test_input)
