# import sqlite3

# DB = None

# def db_connect(dbfile) :
#     global DB
#     DB = sqlite3.connect(dbfile)
#     DB.executescript("""pragma foreign_keys = ON;""")
#     DB.row_factory = sqlite3.Row

# db_connect("streets.db")

import csv
import re
import itertools

known_order_nums = {}
def order_to_id(order) :
    order = order.strip()
    try :
        return known_order_nums[order]
    except KeyError :
        id = len(known_order_nums) + 1
        known_order_nums[order] = id
        return id

class Identified(object) :
    next_gid = 1
    def __init__(self) :
        self.id = Identified.next_gid
        Identified.next_gid += 1
    def __eq__(self, other) :
        return self.id == other.id # less safe but faster!
        return type(self) == type(other) and self.id == other.id
    def __hash__(self) :
        return hash(self.id)

class Borough(Identified) :
    boroughs = {}
    from_keyword_index = {}
    def __init__(self, name, keywords) :
        Identified.__init__(self)
        self.name = name
        self.keywords = keywords
        for kw in self.keywords :
            Borough.from_keyword_index.setdefault(kw, []).append(self)
        self.boroughs[self.id] = self
    @staticmethod
    def from_keyword(keyword) :
        return Borough.from_keyword_index.get(keyword, [])
    def __repr__(self) :
        return "Borough(id=%r,name=%r)" % (self.id, self.name)

borough_mapping = {'B' : Borough('Bronx', ['bronx']),
                   'K' : Borough('Brooklyn', ['brooklyn']),
                   'M' : Borough('Manhattan', ['manhattan']),
                   'Q' : Borough('Queens', ['queens']),
                   'S' : Borough('Staten Island', ["staten island", 'staten', 'island'])}

class Side(Identified) :
    sides = {}
    def __init__(self, name) :
        Identified.__init__(self)
        self.name = name
        Side.sides[self.id] = self
    def __repr__(self) :
        return "Side(%r)" % self.name

sides = {'N' : Side('North'),
         'E' : Side('East'),
         'S' : Side('South'),
         'W' : Side('West'),
         'M' : Side(None)}

class Street(Identified) :
    """These are just plain street names.  They may exist in multiple locations or worse."""
    streets = {}
    official_name_to_street = {}
    destarred_official_name_to_street = {}
    keywords_to_streets = {}
    def __init__(self, borough, name) :
        Identified.__init__(self)
        self.borough = borough
        self.name = name
        self.streets[self.id] = self
        Street.official_name_to_street[borough.id, name] = self
        Street.destarred_official_name_to_street.setdefault((borough.id, name.split("*", 2)[0]), set()).add(self)
        for keyword in self.keywords() :
            self.keywords_to_streets.setdefault(keyword, set()).add(self)
    @staticmethod
    def get(borough, name) :
        name = name.strip()
        try :
            return Street.official_name_to_street[borough.id, name]
        except KeyError :
            return Street(borough, name)
    def normalized_name(self) :
        return " ".join(self.name.split()).title()
    def keywords(self) :
        parts = self.name.split("*", 2)[0].lower().split()
        kwds = []
        for i in xrange(0, len(parts)) :
            for j in xrange(i, len(parts)) :
                kwds.append(" ".join(parts[i:j+1]))
        return kwds
    def __repr__(self) :
        return "Street(id=%r,borough=%r,name=%r)" % (self.id, self.borough.name, self.normalized_name())

STREETS_PARALLEL = "parallel"
STREETS_ADJACENT = "adjacent"
street_relations = {}
def add_street_relation(street1, street2, relation) :
    street_relations.setdefault(street1.id, set()).add((street2.id, relation))
    street_relations.setdefault(street2.id, set()).add((street1.id, relation))
def get_related_streets(street) :
    return [(Street.streets[s], r) for (s, r) in street_relations.get(street.id, [])]
colocation_relations = {}
def add_colocated_relation(s1, s2, s3, loc) :
    key = tuple(sorted([s1, s2, s3], key=lambda s : s.id))
    colocation_relations.setdefault(key, []).append(loc)
def are_colocated(s1, s2, s3) :
    key = tuple(sorted([s1, s2, s3], key=lambda s : s.id))
    return key in colocation_relations
def get_locations(s1, s2, s3) :
    key = tuple(sorted([s1, s2, s3], key=lambda s : s.id))
    return colocation_relations.get(key, set())

class Location(Identified) :
    locations = {}
    mstreet_to_locations = {}
    def __init__(self, borough, order, mstreet, fstreet, tstreet, side) :
        Identified.__init__(self)
        self.borough = borough
        self.order = order
        self.mstreet = mstreet
        self.fstreet = fstreet
        self.tstreet = tstreet
        add_street_relation(mstreet, fstreet, STREETS_ADJACENT)
        add_street_relation(mstreet, tstreet, STREETS_ADJACENT)
        add_street_relation(fstreet, tstreet, STREETS_PARALLEL)
        add_colocated_relation(mstreet, fstreet, tstreet, self)
        self.side = side
        self.signs = []
        self.addresses = []
        self.locations[self.id] = self
        self.mstreet_to_locations.setdefault(self.mstreet, []).append(self)
    def __repr__(self) :
        return "Location(%r, %r, %r, %r)" % (self.side, self.mstreet, self.fstreet, self.tstreet)
    @staticmethod
    def get_possible(borough, ms_name, fs_name, ts_name) :
        mss = Street.destarred_official_name_to_street.get((borough.id, ms_name), [])
        fss = Street.destarred_official_name_to_street.get((borough.id, fs_name), [])
        tss = Street.destarred_official_name_to_street.get((borough.id, ts_name), [])
        locs = set()
        for trip in itertools.product(mss, fss, tss) :
            locs.update(loc for loc in get_locations(*trip) if loc.mstreet == trip[0])
        return locs

class Sign(Identified) :
    def __init__(self, seq, curbdist, pointsdir, desc) :
        Identified.__init__(self)
        self.seq = seq
        self.curbdist = curbdist
        self.pointsdir = pointsdir
        self.desc = desc
        self.parsed = self.parse_range(self.desc)
    @staticmethod
    def maybe_make(seq, curbdist, pointsdir, desc) :
        seq = int(seq.strip())
        curbdist = int(curbdist.strip())
        pointsdir = sides.get(pointsdir.strip(), None)
        desc = desc.strip()
        if "(SANITATION BROOM SYMBOL)" in desc :
            return Sign(seq, curbdist, pointsdir, desc)
        else :
            return None
    def parse_range(self, desc) :
        m = re.search("BROOM SYMBOL\\)(.*?)(<--+>|W/|SUPERSEDED|\\(|SEE)", desc)
        part = m.group(1)
        return part.strip().lower()
    def __repr__(self) :
        return "Sign(%r)" % self.parsed

class Address(Identified) :
    keywords_to_address = {}
    official_name_to_address = {}
    """Represents an address."""
    def __init__(self, num) :
        Identified.__init__(self)
        self.num = num
        self.official_name_to_address[num] = self
        for keyword in self.keywords() :
            self.keywords_to_address.setdefault(keyword, set()).add(self)
    @staticmethod
    def maybe_make(num) :
        try :
            return Address.official_name_to_address[num]
        except KeyError :
            a = Address(num)
            return a
    def keywords(self) :
        parts = self.num.lower().split()
        kwds = []
        for i in xrange(0, len(parts)) :
            for j in xrange(i, len(parts)) :
                kwds.append(" ".join(parts[i:j+1]))
        return kwds
    @staticmethod
    def from_keyword(kw) :
        return Address.keywords_to_address.get(kw, set())
    def __repr__(self) :
        return "Address(id=%r,num=%r)" % (self.id, self.num)

def read_locations() :
    lines_read = 0
    csv_matcher = {}
    try :
        with open('locations.CSV', 'rb') as locations :
            if locations.read(3) != "\xef\xbb\xbf" :
                raise Exception("Incorrect byte order mark")
            location_reader = csv.reader(locations)
            for row in location_reader:
                lines_read += 1
                borough = borough_mapping[row[0]]
                l = Location(borough,
                             order_to_id(row[1]),
                             Street.get(borough, row[2]),
                             Street.get(borough, row[3]),
                             Street.get(borough, row[4]),
                             sides[row[5]])
                csv_matcher[(row[0], l.order)] = l
    except :
        print "Offending line:", row
        print "Line number:", lines_read
        raise
    return csv_matcher

def read_signs(csv_matcher) :
    lines_read = 0
    try :
        with open('signs.CSV', 'rb') as signs :
            if signs.read(3) != "\xef\xbb\xbf" :
                raise Exception("Incorrect byte ordre mark")
            signs_reader = csv.reader(signs)
            for row in signs_reader:
                lines_read += 1
                s = Sign.maybe_make(row[2], row[3], row[4], row[5])
                if s != None :
                    l = csv_matcher[(row[0], order_to_id(row[1]))]
                    l.signs.append(s)
    except :
        print "Offending line:", row
        print "Line number:", lines_read
        raise

def read_addresses() :
    borough_codes = {
        1 : borough_mapping['M'],
        2 : borough_mapping['B'],
        3 : borough_mapping['K'],
        4 : borough_mapping['Q'],
        5 : borough_mapping['S']
    }
    lines_read = 0
    num_worked = 0
    num_failed = 0
    try :
        with open('addresses.csv', 'rb') as addresses :
            addresses_reader = csv.reader(addresses)
            for row in addresses_reader :
                lines_read += 1
                if not row : continue
                borough = borough_codes[int(row[0])]
                num = row[1].strip()
                ms = row[2].strip()
                fs = row[3].strip()
                ts = row[4].strip()
                locs = Location.get_possible(borough, ms, fs, ts)
                if locs :
                    num_worked += 1
                    for loc in locs :
                        a = Address.maybe_make(num)
                        loc.addresses.append(a)
                else :
                    num_failed += 1
        print "num_worked:",num_worked,"num_failed",num_failed
    except :
        print "Offending line:", row
        print "Line number:", lines_read
        raise

def read_all() :
    csv_matcher = read_locations()
    read_signs(csv_matcher)
    read_addresses()

def initialize() :
    read_all()

initialize()
