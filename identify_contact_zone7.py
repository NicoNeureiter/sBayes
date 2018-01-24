# In this document we test an algorithm to identify linguistic contact zones in simulated data.
# We import the data from a PostgreSQL database.

from contextlib import contextmanager
import psycopg2
from igraph import *
import random
import pandas
import numpy as np
import scipy.stats
import time
import csv
from math import sqrt
from math import acos
from scipy.spatial import distance


@contextmanager
def psql_connection(commit=False):
    """
        This function opens a connection to PostgreSQL,
        performs a DB operation and finally closes the connection
        :In
        - commit: a Boolean variable that specifies if the operation is actually executed
        :Out
        -
        """
    # Connection settings for PostgreSQL
    conn = psycopg2.connect(dbname='limits-db', port=5432, user='contact_zones',
                            password='letsfindthemcontactzones', host='limits.geo.uzh.ch')
    cur = conn.cursor()
    try:
        yield conn
    except psycopg2.DatabaseError as err:
        error, = err.args
        sys.stderr.write(error.message)
        cur.execute("ROLLBACK")
        raise err
    else:
        if commit:
            cur.execute("COMMIT")
        else:
            cur.execute("ROLLBACK")
    finally:
        conn.close()


def get_network():
    """
        This function first retrieves the edge list and the coordinates of the simulated languages
        from the DB, then converts these into a spatial network.
        :In
        -
        :Out
        - net: a dictionary containing the network (igraph), its vertices and edges, and its
        adjacency matrix
        """

    # Get vertices from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_v = "SELECT id, mx AS x, my AS y " \
                  "FROM cz_sim.contact_zone_7_test;"
        cursor = connection.cursor()
        cursor.execute(query_v)
        rows_v = cursor.fetchall()
    vertices = []

    for v in rows_v:

        vertex = {'gid': int(v[0]),
                  'x': v[1],
                  'y': v[2]}
        vertices.append(vertex)

    # Get edges from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_e = "SELECT v1, v2 " \
                  "FROM cz_sim.delaunay_edge_list;"

        cursor = connection.cursor()
        cursor.execute(query_e)
        rows_e = cursor.fetchall()

    edges = []
    for e in rows_e:
        edges.append((int(e[0]), int(e[1])))

    # Get the bounding box of the network from PostgreSQL
    with psql_connection(commit=True) as connection:
        query_bbox = "SELECT min(mx) AS min_x, max(mx) AS max_x," \
                     "min(my) AS min_y, max(my) AS max_y FROM " \
                     "cz_sim.contact_zone_7_test;"
        cursor = connection.cursor()
        cursor.execute(query_bbox)
        rows_bbox = cursor.fetchone()
        bbox = {'min_x': rows_bbox[0],
                'max_x': rows_bbox[1],
                'min_y': rows_bbox[2],
                'max_y': rows_bbox[3]}

    # Define the graph
    lang_network = Graph()

    # Populate the graph
    # Add vertices
    lang_network.add_vertices(len(vertices))
    # Add edges
    lang_network.add_edges(edges)

    # Convert graph to adjacency list
    adj_list = lang_network.get_adjlist()

    net = {'vertices': vertices,
           'edges': edges,
           'adj_list': adj_list,
           'network': lang_network,
           'bbox': bbox}
    return net


def create_table_for_results():
    """
        This function creates a table in PostgreSQL to store the results
        : In
        -
        : Out
        -
        """
    with psql_connection(commit=True) as connection:
        query_c = "CREATE TABLE cz_sim.mcmc_results"\
                  "(iteration INTEGER, languages INTEGER[]," \
                  "log_likelihood DOUBLE PRECISION," \
                  "size INTEGER);"
        cursor = connection.cursor()
        cursor.execute(query_c)


def get_features():
    """
        This function retrieves the features of all simulated languages from the DB
        :In
        -
        :Out
        - f: a binary matrix of all features
        """
    with psql_connection(commit=True) as connection:

        # Retrieve the features for all languages
        f = pandas.read_sql_query("SELECT f1, f2, f3, f4, f5, f6,"
                                  "f7, f8, f9, f10, f11, f12, f13,"
                                  "f14, f15, f16, f17, f18, f19, f20,"
                                  "f21, f22, f23, f24, f25, f26, f27,"
                                  "f28, f29, f30 "
                                  "FROM cz_sim.contact_zone_7_test ORDER BY id ASC;", connection)
    f = f.as_matrix()

    # Change A to 0 and P to 1
    f[f == 'A'] = 0
    f[f == 'P'] = 1

    return f


def compute_feature_prob(feat):
    """
        This function computes the base-line probabilities for a feature to be present
        :In
        - feat: a matrix of features
        :Out
        - p_present: a vector containing the probabilities of features to be present"""
    n = len(feat)
    present = np.count_nonzero(feat == 1, axis=0)
    p_present = present/n

    return p_present


def compute_direction(a, b):
    """
        This function computes the direction between two points a and b in clockwise direction
        north = 0 , east = pi/2, south = pi, west = 3pi/2
        :In
          - A: x and y coordinates of the point a, in a metric CRS
          - B: x and y coordinates of the point b, in a metric CRS.
        :Out
          - dir_rad: the direction from A to B in radians
        """

    a = np.asarray(a)
    b = np.asarray(b)
    ab = b - a
    north = [0, 1]
    dot_prod = north[0]*ab[0] + north[1]*ab[1]
    length_north = sqrt(north[0]**2 + north[1]**2)
    length_ab= sqrt(ab[0]**2 + ab[1]**2)
    cos_dir = dot_prod/(length_north * length_ab)

    # Direction in radians
    dir_rad = acos(cos_dir)

    return dir_rad


def compute_distance(a, b):
    """
        This function computes the Euclidean distance between two points a and b
        :In
          - A: x and y coordinates of the point a, in a metric CRS
          - B: x and y coordinates of the point b, in a metric CRS.
        :Out
          - dist: the Euclidean distance from a to b
        """
    a = np.asarray(a)
    b = np.asarray(b)
    ab = b-a
    dist = sqrt(ab[0]**2 + ab[1]**2)

    return dist


def propose_diffusion(net, start_lang, max_size, mu, kappa, dir_lookup):
    """This function proposes a random diffusion in a network. A diffusion is a possible contact zone.
        :In
        - net: a dictionary comprising the vertices, edges, adjacency list and the graph of the network
        - start_lang: the starting point of the diffusion
        - max_size: the number of languages in the diffusion
        - mu: the direction of the diffusion, i.e. the center of the von Mises distribution
        - kappa: the spread of the diffusion, i.e. the dispersion of the von Mises distribution
        - dir_lookup: the lookup table of all directions between all languages
        :Out
        - diffusion: a list of vertices that are part of the diffusion"""

    # Append the starting point to the diffusion, thereby marking it as visited
    diffusion = [start_lang]

    # Find all neighbours of the starting point
    # Compute the direction to the starting point
    # Make them potential candidates of the diffusion process

    candidates = {}
    for n in net['adj_list'][start_lang]:
        candidates[n] = {'direction': dir_lookup[start_lang, n]}

    # Expand the diffusion until the contact zone reaches the desired size
    while len(diffusion) < max_size:

        # Pick a random direction mu, kappa is the tendency of mu to spread
        random_dir = random.vonmisesvariate(mu, kappa)

        # Choose the candidate that lies in the direction closest to the direction of random_dir
        # and append it to the diffusion
        cur_lang = min(candidates, key=lambda x: abs(candidates[x]['direction'] - random_dir))
        diffusion.append(cur_lang)

        # Find all neighbours of the current line
        # Make them potential candidates of the diffusion
        for n in net['adj_list'][cur_lang]:
            if n not in candidates.keys() and n not in diffusion:
                candidates[n] = {'direction': dir_lookup[start_lang, n]}

        # remove the current line from the candidates
        candidates.pop(cur_lang)
    return diffusion


def lookup_direction(vertices):
    """This function generates a lookup table of directions between all languages pairs
    :In
    - vertices: the vertices in the network
    : Out
    - lookup_dir: a matrix with the direction between all language pairs
    """
    lookup_dir = np.empty([len(vertices), len(vertices)])
    for i in range(0, len(vertices)):
        for j in range(0, len(vertices)):
            a = [vertices[i]['x'], vertices[i]['y']]
            b = [vertices[j]['x'], vertices[j]['y']]
            if i == j:
                lookup_dir[i, j] = 0
            else:
                lookup_dir[i, j] = compute_direction(a, b)
    return lookup_dir


def lookup_distance(vertices):
    """This function generates a lookup table of distances between all languages pairs
    :In
    - vertices: the vertices in the network
    : Out
    - lookup_dir: a matrix with the distance between all language pairs
    """
    lookup_dist = np.empty([len(vertices), len(vertices)])
    for i in range(0, len(vertices)):
        for j in range (0, len(vertices)):
            a = [vertices[i]['x'], vertices[i]['y']]
            b = [vertices[j]['x'], vertices[j]['y']]
            if i == j:
                lookup_dist[i, j] = 0
            else:
                lookup_dist[i, j] = compute_distance(a, b)
    return lookup_dist


def lookup_lh(min_size, max_size, feat_prob):
    """This function generates a lookup table of likelihoods
    :In
    - min_size: the minimum number of languages in a diffusion
    - max_size: the maximum number of languages in a diffusion
    - feat_prob: the probability of a feature to be present
    :Out
    - lookup_dict: the lookup table of likelihoods for a specific feature, sample size and presence
    """

    lookup_dict = {}
    for f in range(0, len(feat_prob)):
        lookup_dict[f] = {}
        for s in range(min_size, max_size+1):
            lookup_dict[f][s] = {}
            for p in range(0, s+1):
                lookup_dict[f][s][p] = -np.log(scipy.stats.binom_test(p, s, feat_prob[f], alternative='two-sided'))

    return lookup_dict


def compute_likelihood(diffusion, feat, lookup):

    """This function computes the likelihood of a diffusion.
    The function performs a two-sided binomial test. The test computes the probability of the observed presence/absence
    in the diffusion given the presence/absence in the data. Then the function takes the logarithm of the binomial
    test and multiplies it by -1. If the data in the diffusion follow the general trend, the diffusion is not indicative
    of language contact. When the binomial test yields a high value, the negative logarithm is small."""

    # Retrieve all languages in the diffusion
    idx = diffusion
    n = len(diffusion)

    # Slice up the matrix
    f = feat[idx]

    # Count the presence and absence
    present = np.count_nonzero(f == 1, axis=0)
    # absent = np.count_nonzero(f == 0, axis=0)
    log_lh = []

    for x in range(0, len(present)):
        log_lh.append(lookup[x][n][present[x]])
    log_lh = sum(log_lh)
    return log_lh


def propose_cand_start_location(mu, sigma_x, sigma_y, net):
    """This function proposes a new candidate for the start location.
    First, it generates a random number from a Bivariate normal distribution centered at the current start location.
    Then it picks the language closest to the proposed location as the new candidate start location.

        :In
        - mu: the current start_location
        - sigma_x: the dispersion of the Bivariate normal in the x-direction
        - sigma_y: the dispersion of the Bivariate normal in the y-direction
        - net: the network of languages

        :Out
        - cand: a new candidate for the start_location"""

    cov = [[sigma_x, 0], [0, sigma_y]]
    x_cand, y_cand = np.random.multivariate_normal([mu['x'], mu['y']], cov, 1).T

    # Find the point in the network closest to the tuple x_cand, y_cand
    cand = min(net['vertices'], key=lambda v: sqrt((v['x'] - x_cand) ** 2 + (v['y'] - y_cand) ** 2))

    return cand


def propose_cand_direction(mu, sigma):
    """This function proposes a new candidate for direction using a von Mise distribution,
    centered at mu, the current direction.
        :In
        - mu: the current direction
        - sigma: the reciprocal of sigma^2 defines kappa, the measure of concentration of the von Mise distribution

        :Out
        - cand: a new candidate for direction
        """
    kappa = 1/sigma**2
    cand = random.vonmisesvariate(mu, kappa)
    return cand


def propose_cand_size(mu, sigma, min_size, max_size):
    """This function proposes a new candidate for size using a truncated normal distribution,
    centered at mu, the current size. The sampled values follow a normal distribution with variance sigma^2
    except that values smaller than min_size or larger than max_size are re-picked. The sampled values are rounded to
    the nearest integer.
            :In
            - mu: the current size
            - sigma: the standard deviation of the normal distribution
            - min_size: the minimum value the candidate can take
            - max_size: the maximum value the candidate can take
            :Out
            - cand: a new candidate for size
            """
    while True:
        cand = np.around(random.gauss(mu, sigma))
        if cand < min_size or cand > max_size:
            continue
        else:
            break
    return cand


def propose_cand_spread(mu, sigma, min_spread, max_spread):
    """This function proposes a new candidate for spread using a truncated normal distribution,
    centered at mu, the current spread. The sampled values follow a normal distribution with variance sigma^2
    except that values smaller than min_spread or larger than max_spread are re-picked.
        :In
        - mu: the current spread
        - sigma: the standard deviation of the normal distribution
        - min_spread: the minimum value the candidate can take
        - max_spread: the maximum value the candidate can take
        :Out
        - cand: a new candidate for spread
        """

    while True:
        cand = np.around(random.gauss(mu, sigma))
        if cand < min_spread or cand > max_spread:
            continue
        else:
            break
    return cand


def run_metropolis_hastings(it, net, feat, prior, lookup):

    # Choose a random start location, size, direction and spread
    start_location = random.choice(net['vertices'])
    size = random.randint(prior['size']['min'], prior['size']['max'])
    direction = random.uniform(0, 2 * pi)
    spread = random.uniform(prior['spread']['min'], prior['spread']['max'])

    # Propose a random diffusion
    diff = propose_diffusion(net, start_location['gid'], size, direction, spread, lookup['direction'])

    # Compute the likelihood of the diffusion
    lh = compute_likelihood(diff, feat, lookup['lh'])

    # This dictionary stores statistics and results of the MCMC
    mcmc_stats = {'posterior': [],
                  'acceptance_ratio': []}

    # The number of accepted moves in the MCMC
    acc = 0

    # Propose a candidate diffusion
    # The candidate diffusion differs from the current diffusion
    # The proposal distribution defines how much the candidate diffusion differs

    # Adaptive MCMC
    # Sigma in the proposal distribution is set to an initial value and then updated with the
    # variance in the target distribution! Only pick when variable is updated!

    # Define the sigma of the proposal distributions
    sigma = {'start_location_x': 100,
             'start_location_y': 100,
             'size': 2,
             'direction': 0.5,
             'spread': 10}

    for i in range(1, it):
        # Propose a new candidate for the start location
        start_location_cand = propose_cand_start_location(start_location, sigma['start_location_x'],
                                                sigma['start_location_y'], net)

        # Propose a new candidate for size
        size_cand = propose_cand_size(size, sigma['size'], prior['size']['min'], prior['size']['max'])

        # Propose a new candidate for direction
        direction_cand = propose_cand_direction(direction, sigma['direction'])

        # Propose a new candidate for spread
        spread_cand = propose_cand_spread(spread, sigma['spread'], prior['spread']['min'], prior['spread']['max'])

        # Propose a new candidate diffusion
        diff_cand = propose_diffusion(net, start_location_cand['gid'],
                                      size_cand, direction_cand, spread_cand,
                                      lookup['direction'])

        # Compute the likelihood of the candidate diffusion
        lh_cand = compute_likelihood(diff_cand, feat, lookup['lh'])

        # This is the core of the MCMC: We compare the candidate to the current diffusion
        # Usually, we go for the better of the two diffusions,
        # but sometimes we decide for the candidate, even if it's worse
        a = lh_cand - lh

        if np.log(random.uniform(0, 1)) < a:
            start_location = start_location_cand
            diff = diff_cand
            lh = lh_cand
            size = size_cand
            direction = direction_cand
            spread = spread_cand
            acc += 1

        mcmc_stats['posterior'].append(dict([('iteration', i),
                                             ('languages', diff),
                                             ('start_location', start_location),
                                             ('size', size),
                                             ('spread', spread),
                                             ('direction', direction),
                                             ("log_likelihood", lh)]))
    mcmc_stats['acceptance_ratio'] = acc / it
    return mcmc_stats


if __name__ == "__main__":

    start_time = time.time()

    # Get all necessary data
    # Retrieve the network from the DB
    network = get_network()

    # Retrieve the features for all languages in the sample
    features = get_features()

    # Compute the probability of a feature to be present/absent
    feature_prob = compute_feature_prob(features)

    # Define the hyper priors for the hyper parameters
    hyper_priors = {'size': {'prior': 'uniform',
                             'min': 5,
                             'max': 50},
                    'direction': {'prior': 'uniform'},
                    'spread': {'prior': 'uniform',
                               'min': 0,
                               'max': 100},
                    'start_location': {'prior': 'uniform'}}

    # Compute lookup tables for likelihood and direction,
    # this speeds up the processing time of the algorithm
    lookup_tables = {'lh': lookup_lh(hyper_priors['size']['min'],
                                     hyper_priors['size']['max'], feature_prob),
                     'direction': lookup_direction(network['vertices'])}

    # Compute a lookup table for distances between all vertices
    # distance_lookup = lookup_distance(network['vertices'])

    # Tune the MCMC
    # Number of iterations of the Markov Chain while tuning
    iterations = 1000

    mcmc = run_metropolis_hastings(iterations, network, features, hyper_priors, lookup_tables)

    print(mcmc['acceptance_ratio'])


    #if i % 1000 == 0:
    #    posterior.append(dict([("iterations", i), ("languages", diff), ("log_likelihood", lh), ("size", size)]))
    #    print(i, " iterations performed in", (time.time() - start_time), "seconds")

    #  Write the results to a csv file
    #with open('mcmc_results.csv', 'w') as csvfile:
    #    result_writer = csv.writer(csvfile, delimiter=';', quotechar='"',
    #                               quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    #    result_writer.writerow(["Iterations", "Languages", "Log-lh", "size"])
    #    for item in posterior:
    #        result_writer.writerow([item['iterations'], item['languages'], item['log_likelihood'], item['size']])






