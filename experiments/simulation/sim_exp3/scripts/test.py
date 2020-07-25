import pickle
from sbayes.util import decode_area, format_area_columns, parse_area_columns
import numpy
#
# with open('../results/number_zones_n1_0.pkl', 'rb') as f:
#     data = pickle.load(f)
#     print(data['sample_zones'])
#     print(len(data['sample_zones']))
#
# print('Reading input data...')

sample_path = '/home/olga/PycharmProjects/contact_zones/experiments/simulation/sim_exp3/results/2020-07-02_18-43/n4/ground_truth/areas.txt'

results = []

with open(sample_path, 'r') as f_sample:
    # Read a byte string, decode as a numpy array
    byte_results = (f_sample.read()).split('\n')

    for result in byte_results:

        parsed_result = parse_area_columns(result)
        # decoded_result = decode_area(parsed_result)
        results.append(parsed_result)
    # Transform numpy array into a string, parse columns, return numpy array
    # formatted_areas = format_area_columns(decoded_results)

print(len(results[0]))