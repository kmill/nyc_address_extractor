import sys
from math import log
import itertools
import re

class HMM(object) :
    def __init__(self) :
        pass
    def state_pairs(self) :
        return itertools.product(self.states, self.states)
    def initialize(self, states=None, alphabet=None, p_init=None, p_trans=None, p_emit=None) :
        self.states = states
        self.alphabet = alphabet
        self.pi = p_init
        self.A = p_trans
        self.B = p_emit
    def train(self, obs, N=100, delta=0.0001) :
        p_O_lambda = -1
        p_del = 1
        for iteration in xrange(N) :
            print iteration
            p_O_lambda_old = p_O_lambda
            self.expectation(obs)
            self.maximization(obs)
            p_0_lambda = reduce(lambda x, y : x * y, self.P)
            p_del =  p_0_lambda - p_O_lambda_old
            if p_del < 0 :
                raise Exception(r'P(O|\lambda) not monotone.', p_del)
            if p_del < delta :
                break
        self.smooth()
        return N, p_del
    def smooth(self) :
        for key in self.pi :
            self.pi[key] = (self.pi[key] + 0.01) / (1 + 0.01 * len(self.pi))
        for i in self.A :
            for j in self.A[i] :
                self.A[i][j] = (self.A[i][j] + 0.01) / (1 + 0.01 * len(self.A[i]))
            for k in self.B[i] :
                self.B[i][k] = (self.B[i][k] + 0.01) / (1 + 0.01 * len(self.B[i]))
    def viterbi(self, obs=[]) : 
        # Initialize base cases (t == 0)
        V = [{y:(log(self.pi[y]) + log(self.B[y][obs[0]])) for y in self.states}]
        path = {y:[y] for y in self.states}
 
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
 
            for y in self.states:
                # y0 is previous state, y is current state
                # 'state' will get the previous state that maximizes P(state y at time t)
                (prob, state) = max((V[t-1][y0] + log(self.A[y0][y]) + log(self.B[y][obs[t]]), y0) for y0 in self.states)
                V[t][y] = prob #probability of being in state y at time t (conditional only on previous state)
                newpath[y] = path[state] + [y]
 
            # Don't need to remember the old paths
            # at any point, path[y] for y \in states will be the most likely path leading to state y at time t
            path = newpath
 
        # calculate the most likely state to be in at time t (end), then grab the most likely path
        (prob, state) = max((V[t][y], y) for y in self.states)
        return (prob, path[state])

    def expectation(self, obs) :
        self.calculate_alpha(obs)
        self.calculate_beta(obs)
        self.calculate_xi(obs)
        for i in xrange(len(self.xi)) :
            for j in xrange(len(self.xi[i])) :
                if sum(self.xi[i][j].values()) - 1 > 0.0001 :
                    raise Exception('Xi not stochastic.', sum(self.xi[i][j].values()))
        self.calculate_gamma(obs)
        for i in xrange(len(self.xi)) :
            for j in xrange(len(self.gamma[i])) :
                if sum(self.gamma[i][j].values()) - 1 > 0.0001 :
                    raise Exception('Gamma not stochastic.', sum(self.gamma[i][j].values()))
        self.calculate_P(obs)
    def maximization(self, obs) :
        self.pi = {i : sum(1.0 * self.P[k] * self.gamma[k][0][i] for k in xrange(len(obs)))
                       / sum(self.P[k] for k in xrange(len(obs)))
                   for i in self.states}

        self.A = {i : {j : sum(1.0 * self.P[k] * sum(self.xi[k][t][(i,j)] for t in xrange(len(obs[k]) - 1))
                               for k in xrange(len(obs)))
                           / sum(1.0 * self.P[k] * sum(self.gamma[k][t][i] for t in xrange(len(obs[k]) - 1))
                                 for k in xrange(len(obs)))
                       for j in self.states}
                  for i in self.states}

        self.B = {i : { v : sum(1.0 * self.P[k] * sum(self.gamma[k][t][i] if obs[k][t] == v else 0
                                                      for t in xrange(len(obs[k])-1))
                                for k in xrange(len(obs)))
                            / sum(1.0 * self.P[k] * sum(self.gamma[k][t][i] 
                                                        for t in xrange(len(obs[k])-1))
                                  for k in xrange(len(obs)))
                        for v in self.alphabet }
                  for i in self.states}
    def calculate_xi_i_j_t(self, k, i, j, t, obs) :
        obs_t1 = obs[t+1]
        
        numerator = self.alpha[k][t][i] \
                    * self.A[i][j] \
                    * self.B[j][obs_t1] \
                    * self.beta[k][t+1][j]

        denominator = sum(self.alpha[k][t][i_d] 
                          * self.A[i_d][j_d] 
                          * self.B[j_d][obs_t1] 
                          * self.beta[k][t+1][j_d] 
                          for (i_d, j_d) in self.state_pairs())
        try :
            a = numerator / denominator
        except :
            print 'A'
            print self.A
            print 'B'
            print self.B
            print 'alpha'
            print self.alpha[k]
            print 'beta'
            print self.beta[k]
            raise

        return numerator / denominator
    def calculate_xi(self, obs) :
        self.xi = [[{(i,j) : self.calculate_xi_i_j_t(k, i, j, t, obs[k])
                        for (i,j) in self.state_pairs()}
                    for t in xrange(len(obs[k]) - 1)]
                   for k in xrange(len(obs))]

    def calculate_gamma(self, obs) :
        self.gamma = [[{i : sum(self.xi[k][t][(i,j)] 
                               for j in self.states)
                        for i in self.states}
                       for t in xrange(len(obs[k]) - 1)]
                      for k in xrange(len(obs))]
    def calculate_P(self, obs) :
        self.P = [sum(self.alpha[k][len(obs[k]) - 1][i] for i in self.states) for k in xrange(len(obs))]
    #forward procedure
    def calculate_alpha(self, obs) :
        # 1) Initialization
        self.alpha = [[ { i : 1.0 * self.pi[i] * self.B[i][obs[k][0]] for i in self.states} ]
                      for k in xrange(len(obs)) ]
        # 2) Induction
        for k in xrange(len(obs)) :
            for t in xrange(len(obs[k]) - 1) :
                self.alpha[k].append({ j : 1.0 * self.B[j][obs[k][t+1]] 
                                               * sum(self.alpha[k][t][i] * self.A[i][j]
                                                     for i in self.states) 
                                       for j in self.states })
    #backward procedure
    def calculate_beta(self, obs) :
        # 1) Initialization
        self.beta = [ [0] * len(obs[k]) for k in xrange(len(obs)) ]
        # 2) Induction
        for k in xrange(len(obs)) : 
            self.beta[k][-1] = { i : 1 for i in self.states }
            for t in xrange(2, len(obs[k])+1) :
                self.beta[k][-t] = { i : sum(self.A[i][j] * self.B[j][obs[k][1-t]] * self.beta[k][1-t][j]
                                             for j in self.states)
                                     for i in self.states }
            


class Test(object) :
    @staticmethod
    def accuracy(a, b):
        total = float(max(len(a),len(b)))
        c = 0
        for i in range(min(len(a),len(b))):
            if a[i] == b[i]:
                c = c + 1          
        return c/total

    @classmethod
    def run(cls, training_data=[[]], test_data={}) :
        hmm = HMM()
        hmm.initialize(states=cls.states,
                       alphabet=cls.alphabet,
                       p_init=cls.p_init,
                       p_trans=cls.p_trans,
                       p_emit=cls.p_emit)
        iters, p_del = hmm.train(training_data)
        prob, path  = hmm.viterbi(obs=test_data['obs'])
        accuracy = Test.accuracy(test_data['cls'], path)
        print test_data['cls']
        print path
        print prob
        print accuracy
                       

class WeatherTest(Test) :
    states = ('sunny', 'rainy', 'foggy')
    alphabet = ('yes', 'no') #umbrella
    p_init = {'sunny' : 0.5, 'rainy' : 0.25, 'foggy' : 0.25}
    p_trans = { 'sunny' : { 'sunny' : 0.8,
                            'rainy' : 0.05,
                            'foggy' : 0.15 },
                'rainy' : { 'sunny' : 0.2,
                            'rainy' : 0.6,
                            'foggy' : 0.2 },
                'foggy' : { 'sunny' : 0.2,
                            'rainy' : 0.3,
                            'foggy' : 0.5}}
    p_emit = { 'sunny' : { 'yes' : 0.1,
                           'no'  : 0.9 },
               'rainy' : { 'yes' : 0.8,
                           'no'  : 0.2 },
               'foggy' : { 'yes' : 0.3,
                           'no'  : 0.7 } }
    @classmethod
    def run(cls, filename) :
        obs = []
        classes = []
        with open(filename, 'rb') as data :
            data.readline() #skip header
            for i in data.readlines() :
                j = i.split()[0].split(',')
                classes.append(j[0])
                obs.append(j[1])
        training_data = [obs]
        test_data = { 'obs' : obs,
                      'cls' : classes }
        super(WeatherTest, cls).run(training_data=training_data,
                                    test_data=test_data)

class AddressTest(Test) : 
    states = ('num', 'street', 'ext', 'boro', 'default')
    alphabet = ('num', 'street', 'ext', 'boro', 'default')
    default_prob = 0.9
    p_init = {'default' : default_prob}
    for state in states :
        if state == 'default' : 
            pass
        else : 
            p_init[state] = (1-default_prob) / (len(states) - 1)
    p_trans = {
        'num'     : { 'num' : 0.1, 'street' : 0.5, 'ext' : 0.05, 'boro' : 0.05, 'default' : 0.3 },
        'street'  : { 'num' : 0.05, 'street' : 0.2, 'ext' : 0.4, 'boro' : 0.1, 'default' : 0.25 },
        'ext'     : { 'num' : 0.05, 'street' : 0.1, 'ext' : 0.05, 'boro' : 0.3, 'default' : 0.5 },
        'boro'    : { 'num' : 0.05, 'street' : 0.05, 'ext' : 0.05, 'boro' : 0.05, 'default' : 0.8 },
        'default' : { 'num' : 0.05, 'street' : 0.05, 'ext' : 0.05, 'boro' : 0.05, 'default' : 0.8 }
    }
    p_emit = {
        'num'     : { 'num' : 0.9, 'street' : 0.01, 'ext' : 0.01, 'boro' : 0.01, 'default' : 0.07 },
        'street'  : { 'num' : 0.1, 'street' : 0.4, 'ext' : 0.05, 'boro' : 0.05, 'default' : 0.5 },
        'ext'     : { 'num' : 0.01, 'street' : 0.01, 'ext' : 0.96, 'boro' : 0.01, 'default' : 0.01 },
        'boro'    : { 'num' : 0.01, 'street' : 0.01, 'ext' : 0.01, 'boro' : 0.66, 'default' : 0.3 },
        'default' : { 'num' : 0.1, 'street' : 0.1, 'ext' : 0.1, 'boro' : 0.05, 'default' : 0.65 }
    }
    _num_regex = re.compile(r'^([0-9]{1,6}[^\s]+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty)$')
    _borough_regex = re.compile(r'^(manhattan|brooklyn|queens|bronx|staten island|qns|bx|bklyn|bkln|si)$')
    _ext_regex = re.compile(r'^(allee|alley|ally|aly|anex|annex|anx|arc|arcade|av|ave|aven|avenu|avenue|avn|avnue|bayoo|bayou|bch|beach|bend|bnd|blf|bluf|bluff|bluffs|bot|bottm|bottom|btm|blvd|boul|boulevard|boulv|br|branch|brnch|brdge|brg|bridge|brk|brook|brooks|burg|burgs|byp|bypa|bypas|bypass|byps|camp|cmp|cp|canyn|canyon|cnyn|cyn|cape|cpe|causeway|causway|cswy|cen|cent|center|centr|centre|cnter|cntr|ctr|centers|cir|circ|circl|circle|crcl|crcle|circles|clf|cliff|clfs|cliffs|clb|club|common|cor|corner|corners|cors|course|crse|court|crt|ct|courts|cove|cv|coves|ck|cr|creek|crk|crecent|cres|crescent|cresent|crscnt|crsent|crsnt|crest|crossing|crssing|crssng|xing|crossroad|curve|dale|dl|dam|dm|div|divide|dv|dvd|dr|driv|drive|drv|drives|est|estate|estates|ests|exp|expr|express|expressway|expw|expy|ext|extension|extn|extnsn|extensions|exts|fall|falls|fls|ferry|frry|fry|field|fld|fields|flds|flat|flt|flats|flts|ford|frd|fords|forest|forests|frst|forg|forge|frg|forges|fork|frk|forks|frks|fort|frt|ft|freeway|freewy|frway|frwy|fwy|garden|gardn|gdn|grden|grdn|gardens|gdns|grdns|gateway|gatewy|gatway|gtway|gtwy|glen|gln|glens|green|grn|greens|grov|grove|grv|groves|harb|harbor|harbr|hbr|hrbor|harbors|haven|havn|hvn|height|heights|hgts|ht|hts|highway|highwy|hiway|hiwy|hway|hwy|hill|hl|hills|hls|hllw|hollow|hollows|holw|holws|inlet|inlt|is|island|islnd|islands|islnds|iss|isle|isles|jct|jction|jctn|junction|junctn|juncton|jctns|jcts|junctions|key|ky|keys|kys|knl|knol|knoll|knls|knolls|lake|lk|lakes|lks|land|landing|lndg|lndng|la|lane|lanes|ln|lgt|light|lights|lf|loaf|lck|lock|lcks|locks|ldg|ldge|lodg|lodge|loop|loops|mall|manor|mnr|manors|mnrs|mdw|meadow|mdws|meadows|medows|mews|mill|ml|mills|mls|mission|missn|msn|mssn|motorway|mnt|mount|mt|mntain|mntn|mountain|mountin|mtin|mtn|mntns|mountains|nck|neck|orch|orchard|orchrd|oval|ovl|overpass|park|pk|prk|parks|parkway|parkwy|pkway|pkwy|pky|parkways|pkwys|pass|passage|path|paths|pike|pikes|pine|pines|pnes|pl|place|plain|pln|plaines|plains|plns|plaza|plz|plza|point|pt|points|pts|port|prt|ports|prts|pr|prairie|prarie|prr|rad|radial|radiel|radl|ramp|ranch|ranches|rnch|rnchs|rapid|rpd|rapids|rpds|rest|rst|rdg|rdge|ridge|rdgs|ridges|riv|river|rivr|rvr|rd|road|rds|roads|route|row|rue|run|shl|shoal|shls|shoals|shoar|shore|shr|shoars|shores|shrs|skyway|spg|spng|spring|sprng|spgs|spngs|springs|sprngs|spur|spurs|sq|sqr|sqre|squ|square|sqrs|squares|sta|station|statn|stn|stra|strav|strave|straven|stravenue|stravn|strvn|strvnue|stream|streme|strm|st|str|street|strt|streets|smt|sumit|sumitt|summit|ter|terr|terrace|throughway|trace|traces|trce|track|tracks|trak|trk|trks|trafficway|trfy|tr|trail|trails|trl|trls|tunel|tunl|tunls|tunnel|tunnels|tunnl|tpk|tpke|trnpk|trpk|turnpike|turnpk|underpass|un|union|unions|valley|vally|vlly|vly|valleys|vlys|vdct|via|viadct|viaduct|view|vw|views|vws|vill|villag|village|villg|villiage|vlg|villages|vlgs|ville|vl|vis|vist|vista|vst|vsta|walk|walks|wall|way|wy|ways|well|wells|wls)$')
    _street_regex = re.compile(r'^[A-Z]\w+|\d+(st|nd|rd|th)$')
    @classmethod
    def tokenize(cls, query) :
        import string
        exclude = set(string.punctuation)
        s = ''.join(ch for ch in query if ch not in exclude)
        return s.lower().split()
    @classmethod
    def tag(cls, query) :
        observations = []
        tokenized_query = cls.tokenize(query)
        for token in tokenized_query :
            if cls._borough_regex.search(token) : 
                observations.append('boro')
            elif cls._ext_regex.search(token) : 
                observations.append('ext')
            elif cls._street_regex.search(token) : 
                observations.append('street')
            elif cls._num_regex.search(token) : 
                observations.append('num')
            else : 
                observations.append('default')
        return observations

    @classmethod
    def run(cls, training_file, test_file) :
        training_data = []
        with open(training_file, 'rb') as data :
            for i in data.readlines() :
                training_data.append(cls.tag(i))
        test_data = []
                    
        with open(test_file, 'rb') as data :
            while True :
                line1 = data.readline()
                if not line1 : break
                line2 = data.readline()
                if not line2 : break
                test_data.append({ 'obs' : cls.tag(line1),
                                   'cls' : line1.split() })
        hmm = HMM()
        hmm.initialize(states=cls.states,
                       alphabet=cls.alphabet,
                       p_init=cls.p_init,
                       p_trans=cls.p_trans,
                       p_emit=cls.p_emit)
        iters, p_del = hmm.train(training_data)
        for i in xrange(len(test_data)) :
            print 'Test %d' % i
            prob, path = hmm.viterbi(obs=test_data[i]['obs'])
            accuracy = Test.accuracy(test_data[i]['cls'], path)
            print test_data[i]['cls']
            print path
            print prob
            print accuracy




if __name__ == '__main__' :
    if len(sys.argv) > 1 and sys.argv[1] == 'test_weather' :
        WeatherTest.run('weather_test_2.txt')
    if len(sys.argv) > 1 and sys.argv[1] == 'test_address' : 
        AddressTest.run('address_test.txt', 'address_tagged.txt')
