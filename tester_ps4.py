import unittest
import numpy as np
import random
import sys

from ps4 import predicted_sea_level_rise, simulate_year, wait_a_bit, water_level_est, repair_only, prepare_immediately

SEA_LEVEL_DATA = np.array([ [2020.0, 4.15, 3.9, 4.4, 0.1519892079779423], 
                        [2021.0, 4.180000000000001, 3.92, 4.44, 0.15806877629706], 
                        [2022.0, 4.21, 3.94, 4.48, 0.16414834461617775], 
                        [2023.0, 4.24, 3.96, 4.5200000000000005, 0.17022791293529546], 
                        [2024.0, 4.27, 3.98, 4.5600000000000005, 0.1763074812544132], 
                        [2025.0, 4.3, 4.0, 4.6, 0.1823870495735309], 
                        [2026.0, 4.33, 4.02, 4.64, 0.18846661789264862], 
                        [2027.0, 4.359999999999999, 4.04, 4.68, 0.19454618621176636], 
                        [2028.0, 4.39, 4.06, 4.72, 0.20062575453088408], 
                        [2029.0, 4.419999999999999, 4.08, 4.76, 0.20670532285000182], 
                        [2030.0, 4.449999999999999, 4.1, 4.8, 0.21278489116911953], 
                        [2031.0, 4.484999999999999, 4.119999999999999, 4.85, 0.22190424364779604], 
                        [2032.0, 4.52, 4.14, 4.8999999999999995, 0.23102359612647255], 
                        [2033.0, 4.555, 4.16, 4.95, 0.24014294860514906], 
                        [2034.0, 4.59, 4.18, 5.0, 0.24926230108382555], 
                        [2035.0, 4.625, 4.199999999999999, 5.05, 0.2583816535625021], 
                        [2036.0, 4.659999999999999, 4.22, 5.1, 0.26750100604117855], 
                        [2037.0, 4.694999999999999, 4.24, 5.1499999999999995, 0.27662035851985506], 
                        [2038.0, 4.7299999999999995, 4.26, 5.2, 0.28573971099853157], 
                        [2039.0, 4.765, 4.279999999999999, 5.25, 0.2948590634772081], 
                        [2040.0, 4.8, 4.3, 5.3, 0.3039784159558846], 
                        [2041.0, 4.84, 4.32, 5.359999999999999, 0.31613755259411996], 
                        [2042.0, 4.88, 4.34, 5.42, 0.3282966892323554], 
                        [2043.0, 4.92, 4.359999999999999, 5.48, 0.34045582587059076], 
                        [2044.0, 4.96, 4.38, 5.54, 0.3526149625088262], 
                        [2045.0, 5.0, 4.4, 5.6, 0.36477409914706155], 
                        [2046.0, 5.04, 4.42, 5.66, 0.3769332357852969], 
                        [2047.0, 5.08, 4.4399999999999995, 5.720000000000001, 0.38909237242353234], 
                        [2048.0, 5.12, 4.46, 5.78, 0.4012515090617677], 
                        [2049.0, 5.16, 4.48, 5.84, 0.41341064570000313], 
                        [2050.0, 5.2, 4.5, 5.9, 0.4255697823382385], 
                        [2051.0, 5.245, 4.52, 5.970000000000001, 0.44076870313603267], 
                        [2052.0, 5.29, 4.54, 6.04, 0.45596762393382684], 
                        [2053.0, 5.335, 4.5600000000000005, 6.11, 0.471166544731621], 
                        [2054.0, 5.38, 4.58, 6.18, 0.48636546552941523], 
                        [2055.0, 5.425000000000001, 4.6, 6.25, 0.5015643863272095], 
                        [2056.0, 5.470000000000001, 4.62, 6.32, 0.5167633071250035], 
                        [2057.0, 5.515000000000001, 4.640000000000001, 6.39, 0.5319622279227978], 
                        [2058.0, 5.5600000000000005, 4.66, 6.46, 0.547161148720592], 
                        [2059.0, 5.605, 4.68, 6.529999999999999, 0.5623600695183861], 
                        [2060.0, 5.65, 4.7, 6.6, 0.5775589903161803], 
                        [2061.0, 5.695, 4.720000000000001, 6.67, 0.5927579111139746], 
                        [2062.0, 5.74, 4.74, 6.739999999999999, 0.6079568319117689], 
                        [2063.0, 5.785, 4.76, 6.81, 0.6231557527095631], 
                        [2064.0, 5.83, 4.78, 6.88, 0.6383546735073574], 
                        [2065.0, 5.875, 4.800000000000001, 6.949999999999999, 0.6535535943051517], 
                        [2066.0, 5.92, 4.82, 7.02, 0.668752515102946], 
                        [2067.0, 5.965, 4.84, 7.09, 0.6839514359007403], 
                        [2068.0, 6.01, 4.86, 7.16, 0.6991503566985345], 
                        [2069.0, 6.055, 4.880000000000001, 7.2299999999999995, 0.7143492774963288], 
                        [2070.0, 6.1, 4.9, 7.3, 0.7295481982941231], 
                        [2071.0, 6.145, 4.91, 7.38, 0.750826687411035], 
                        [2072.0, 6.1899999999999995, 4.92, 7.46, 0.7721051765279469], 
                        [2073.0, 6.234999999999999, 4.930000000000001, 7.54, 0.7933836656448587], 
                        [2074.0, 6.279999999999999, 4.94, 7.62, 0.8146621547617706], 
                        [2075.0, 6.324999999999999, 4.95, 7.699999999999999, 0.8359406438786825], 
                        [2076.0, 6.37, 4.96, 7.779999999999999, 0.8572191329955945], 
                        [2077.0, 6.415, 4.97, 7.859999999999999, 0.8784976221125064], 
                        [2078.0, 6.46, 4.98, 7.9399999999999995, 0.8997761112294183], 
                        [2079.0, 6.505, 4.99, 8.02, 0.9210546003463301], 
                        [2080.0, 6.55, 5.0, 8.1, 0.942333089463242], 
                        [2081.0, 6.6049999999999995, 5.02, 8.19, 0.963611578580154], 
                        [2082.0, 6.66, 5.04, 8.28, 0.9848900676970659], 
                        [2083.0, 6.715, 5.0600000000000005, 8.37, 1.0061685568139778], 
                        [2084.0, 6.77, 5.08, 8.459999999999999, 1.0274470459308898], 
                        [2085.0, 6.824999999999999, 5.1, 8.55, 1.0487255350478017], 
                        [2086.0, 6.88, 5.12, 8.64, 1.0700040241647137], 
                        [2087.0, 6.935, 5.140000000000001, 8.73, 1.0912825132816257], 
                        [2088.0, 6.989999999999999, 5.16, 8.82, 1.1125610023985375], 
                        [2089.0, 7.045, 5.18, 8.91, 1.1338394915154495], 
                        [2090.0, 7.1, 5.2, 9.0, 1.1551179806323615], 
                        [2091.0, 7.16, 5.220000000000001, 9.1, 1.179436253908832], 
                        [2092.0, 7.22, 5.24, 9.2, 1.203754527185303], 
                        [2093.0, 7.279999999999999, 5.26, 9.3, 1.2280728004617738], 
                        [2094.0, 7.34, 5.28, 9.4, 1.2523910737382444], 
                        [2095.0, 7.4, 5.300000000000001, 9.5, 1.276709347014715], 
                        [2096.0, 7.46, 5.32, 9.6, 1.301027620291186], 
                        [2097.0, 7.52, 5.34, 9.7, 1.3253458935676568], 
                        [2098.0, 7.58, 5.36, 9.8, 1.3496641668441274], 
                        [2099.0, 7.640000000000001, 5.380000000000001, 9.9, 1.373982440120598], 
                        [2100.0, 7.7, 5.4, 10.0, 1.3983007133970689]])

class TestPS4(unittest.TestCase):
    def test_part_1_1_predicted_sea_level_rise(self):
        student_out = predicted_sea_level_rise()
        expected_out = SEA_LEVEL_DATA.copy()

        # determining equality between floats
        epsilon = 1e-7

        self.assertIsNotNone(student_out, "predicted_sea_level_rise() returned None instead of numpy array!")
        self.assertTrue(type(student_out)==np.ndarray, f"predicted_sea_level_rise() returned incorrect type. Expected numpy array. Got {type(student_out)}")
        self.assertEqual(expected_out.shape[0], student_out.shape[0], f"predicted_sea_level_rise() returned incorrect number of rows. Expected: {expected_out[0]} Got: {student_out.shape[0]}")
        self.assertEqual(expected_out.shape[1], student_out.shape[1], f"predicted_sea_level_rise() returned incorrect number of columns. Expected: {expected_out[1]} Got: {student_out.shape[1]}")
        for row in expected_out:
            year = row[0]
            student_row = None
            for r in student_out:
                if r[0] == year:
                    student_row = r
                    break
            self.assertIsNotNone(student_row, f"predicted_sea_level_rise() is missing data for year {year}")
            self.assertAlmostEqual(row[1], student_row[1], delta=epsilon, msg=f"For year {year}, expected mean of {row[1]} but got mean of {student_row[1]}")
            self.assertAlmostEqual(row[2], student_row[2], delta=epsilon, msg=f"For year {year}, expected lower 25% of {row[2]} but got lower 25% of {student_row[2]}")
            self.assertAlmostEqual(row[3], student_row[3], delta=epsilon, msg=f"For year {year}, expected upper 25% of {row[3]} but got upper 25% of {student_row[3]}")
            self.assertAlmostEqual(row[4], student_row[4], delta=epsilon, msg=f"For year {year}, expected std_dev of {row[4]} but got std_dev of {student_row[4]}")
            
    def test_part_1_2_simulate_year_1(self):
        #tests simulate_year for N=1

        NUM_TRIALS = 10000
        YEAR = 2073
        N = 1

        epsilon = 1e-1
        
        student_out = simulate_year(SEA_LEVEL_DATA, YEAR, N)

        self.assertIsNotNone(student_out, "simulate_year() returned None instead of numpy array!")
        self.assertTrue(type(student_out)==np.ndarray, f"simulate_year() returned incorrect type. Expected numpy array. Got {type(student_out)}")
        self.assertTrue(len(student_out.shape)==1, f"Expected simulate_year() to return a 1 dimensional numpy array. Got numpy array of shape {student_out.shape}")
        self.assertEqual(N, student_out.shape[0], f"Expected simulate_year() to return 1-D numpy array with {N} elements. Got {student_out.shape[0]}")

        student_outputs = []
        for i in range(NUM_TRIALS):
            student_outputs.append(simulate_year(SEA_LEVEL_DATA, YEAR, N))
        
        student_mean = np.mean(np.array(student_outputs))
        student_std = np.std(np.array(student_outputs))

        expected_mean = SEA_LEVEL_DATA[YEAR-2020][1]
        expected_std = SEA_LEVEL_DATA[YEAR-2020][4]

        self.assertAlmostEqual(expected_mean, student_mean, delta=epsilon, msg=f"For year {YEAR}, expected simulate_year() to return outputs with mean {expected_mean}. Got {student_mean}")
        self.assertAlmostEqual(expected_std, student_std, delta=epsilon, msg=f"For year {YEAR}, expected simulate_year() to return outputs with std_dev {expected_std}. Got {student_std}")
        
    def test_part_1_2_simulate_year_2(self):
        #tests simulate_year for N>1
        YEAR = 2048
        N = 3

        student_out = simulate_year(SEA_LEVEL_DATA, YEAR, N)

        self.assertIsNotNone(student_out, "simulate_year() returned None instead of numpy array!")
        self.assertTrue(type(student_out)==np.ndarray, f"simulate_year() returned incorrect type. Expected numpy array. Got {type(student_out)}")
        self.assertTrue(len(student_out.shape)==1, f"Expected simulate_year() to return a 1 dimensional numpy array. Got numpy array of shape {student_out.shape}")
        self.assertEqual(N, student_out.shape[0], f"Expected simulate_year() to return 1-D numpy array with {N} elements. Got {student_out.shape[0]}")

        N = 100

        student_out = simulate_year(SEA_LEVEL_DATA, YEAR, N)

        self.assertIsNotNone(student_out, "simulate_year() returned None instead of numpy array!")
        self.assertTrue(type(student_out)==np.ndarray, f"simulate_year() returned incorrect type. Expected numpy array. Got {type(student_out)}")
        self.assertTrue(len(student_out.shape)==1, f"Expected simulate_year() to return a 1 dimensional numpy array. Got numpy array of shape {student_out.shape}")
        self.assertEqual(N, student_out.shape[0], f"Expected simulate_year() to return 1-D numpy array with {N} elements. Got {student_out.shape[0]}")

    def test_part_2_1_water_level_est(self):
        np.random.seed(0)
        random.seed(0)
        expected_output = [4.4181169189, 4.2432523603, 4.3706582199, 4.6214625724, 4.5992644453, 4.1217571709, 4.5090599508, 4.3305540324, 4.36929164, 4.5048728959, 4.4806502956, 4.8077094626, 4.6958176721, 4.5842193972, 4.7006383708, 4.7112153245, 5.0596676552, 4.6382490475, 4.8194558746, 4.5131621302, 4.0239461998, 5.0466333831, 5.1637915421, 4.667325595, 5.7603494416, 4.4694850712, 5.057247906, 5.0071681917, 5.7350299728, 5.7674485578, 5.2659409422, 5.4116822033, 4.8851984421, 4.4017149723, 5.2107875455, 5.5034190747, 6.1057690809, 6.1546206632, 5.3480698137, 5.4349970042, 5.0443988082, 4.8532731338, 4.7026613805, 7.0006369098, 5.5046611479, 5.5886949656, 5.0821899521, 6.4967656453, 4.8816427442, 5.9030291345, 5.4467139836, 6.4354967208, 5.7956047091, 5.29830571, 6.2570410051, 6.6830600196, 6.4270198357, 6.6807208429, 5.8892521333, 6.1708955803, 5.9163182687, 6.2585304104, 5.8591403032, 4.9780687254, 6.9522959657, 6.4036420727, 5.1356812086, 7.4400261829, 5.9805752222, 7.1038977412, 7.9421856179, 7.3121271211, 8.5915587323, 5.7635439967, 7.84388908, 6.525696556, 6.3270688572, 6.7528239738, 7.1595087113, 7.717170194, 6.070770146]
        student_output = water_level_est(SEA_LEVEL_DATA)
        
        self.assertIsNotNone(student_output, "water_level_est() returned None instead of a list!")
        self.assertTrue(type(student_output)==type(expected_output), f"Expected water_level_est() to return {type(expected_output)}. Got {type(student_output)}")
        self.assertEqual(len(expected_output), len(student_output), f"Expected water_level_est() to return list of length {len(expected_output)}. Got {len(student_output)}")
        
        epsilon = 1e-2

        for i in range(len(expected_output)):
            self.assertAlmostEqual(expected_output[i], student_output[i], delta=epsilon, msg=f"For water_level_est(), for year {2020+i}, expected water level of {expected_output[i]}. Got {student_output[i]}")

    def test_part_2_2a_repair_only(self):
        np.random.seed(0)
        random.seed(0)

        water_level_loss_no_prevention = np.array([[5,6,7,8,9,10],[0,10,25,45,75,100]]).T

        water_level = [4.4181169189, 4.2432523603, 4.3706582199, 4.6214625724, 4.5992644453, 4.1217571709, 4.5090599508, 4.3305540324, 4.36929164, 4.5048728959, 4.4806502956, 4.8077094626, 4.6958176721, 4.5842193972, 4.7006383708, 4.7112153245, 5.0596676552, 4.6382490475, 4.8194558746, 4.5131621302, 4.0239461998, 5.0466333831, 5.1637915421, 4.667325595, 5.7603494416, 4.4694850712, 5.057247906, 5.0071681917, 5.7350299728, 5.7674485578, 5.2659409422, 5.4116822033, 4.8851984421, 4.4017149723, 5.2107875455, 5.5034190747, 6.1057690809, 6.1546206632, 5.3480698137, 5.4349970042, 5.0443988082, 4.8532731338, 4.7026613805, 7.0006369098, 5.5046611479, 5.5886949656, 5.0821899521, 6.4967656453, 4.8816427442, 5.9030291345, 5.4467139836, 6.4354967208, 5.7956047091, 5.29830571, 6.2570410051, 6.6830600196, 6.4270198357, 6.6807208429, 5.8892521333, 6.1708955803, 5.9163182687, 6.2585304104, 5.8591403032, 4.9780687254, 6.9522959657, 6.4036420727, 5.1356812086, 7.4400261829, 5.9805752222, 7.1038977412, 7.9421856179, 7.3121271211, 8.5915587323, 5.7635439967, 7.84388908, 6.525696556, 6.3270688572, 6.7528239738, 7.1595087113, 7.717170194, 6.070770146]

        expected_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3867062080000068, 0.0, 0.0, 0.0, 0.0, 1.8653353239999857, 6.55166168400001, 0.0, 30.41397766399999, 0.0, 2.2899162399999895, 0.2867276679999975, 29.40119891199998, 30.69794231199999, 10.637637688000012, 16.467288131999993, 0.0, 0.0, 8.431501819999987, 20.136762988, 46.346144854, 49.27723979200002, 13.92279254799998, 17.399880168000017, 1.775952328000017, 0.0, 0.0, 100.05095278399997, 20.18644591600001, 23.54779862400001, 3.2875980840000096, 69.805938718, 0.0, 36.12116537999999, 17.868559343999983, 66.12980324799999, 31.824188364, 11.932228400000007, 55.42246030599998, 80.98360117600001, 65.62119014200002, 80.84325057399998, 35.57008533200001, 50.253734818, 36.65273074800002, 55.511824623999985, 34.36561212800001, 0.0, 97.13775794200002, 64.21852436200001, 5.427248344000013, 135.20209463199993, 39.22300888799999, 108.311819296, 175.374849432, 124.97016968799997, 250.9870478759999, 30.541759868000007, 167.51112640000002, 71.54179335999999, 59.62413143199997, 85.169438428, 112.760696904, 157.37361552000004, 44.24620876]
        student_output = repair_only(water_level, water_level_loss_no_prevention)

        self.assertIsNotNone(student_output, "repair_only() returned None instead of a list!")
        self.assertTrue(type(student_output)==type(expected_output), f"Expected repair_only() to return {type(expected_output)}. Got {type(student_output)}")
        self.assertEqual(len(expected_output), len(student_output), f"Expected repair_only() to return list of length {len(expected_output)}. Got {len(student_output)}")
        
        epsilon = 1e-2

        for i in range(len(expected_output)):
            self.assertAlmostEqual(expected_output[i], student_output[i], delta=epsilon, msg=f"For repair_only(), for year {2020+i}, expected damage costs of {expected_output[i]}. Got {student_output[i]}")

    def test_part_2_2b_wait_a_bit(self):
        np.random.seed(0)
        random.seed(0)

        water_level_loss_no_prevention = np.array([[5,6,7,8,9,10],[0,10,25,45,75,100]]).T
        water_level_loss_with_prevention = np.array([[5,6,7,8,9,10],[0,5,15,30,70,100]]).T

        water_level = [4.4181169189, 4.2432523603, 4.3706582199, 4.6214625724, 4.5992644453, 4.1217571709, 4.5090599508, 4.3305540324, 4.36929164, 4.5048728959, 4.4806502956, 4.8077094626, 4.6958176721, 4.5842193972, 4.7006383708, 4.7112153245, 5.0596676552, 4.6382490475, 4.8194558746, 4.5131621302, 4.0239461998, 5.0466333831, 5.1637915421, 4.667325595, 5.7603494416, 4.4694850712, 5.057247906, 5.0071681917, 5.7350299728, 5.7674485578, 5.2659409422, 5.4116822033, 4.8851984421, 4.4017149723, 5.2107875455, 5.5034190747, 6.1057690809, 6.1546206632, 5.3480698137, 5.4349970042, 5.0443988082, 4.8532731338, 4.7026613805, 7.0006369098, 5.5046611479, 5.5886949656, 5.0821899521, 6.4967656453, 4.8816427442, 5.9030291345, 5.4467139836, 6.4354967208, 5.7956047091, 5.29830571, 6.2570410051, 6.6830600196, 6.4270198357, 6.6807208429, 5.8892521333, 6.1708955803, 5.9163182687, 6.2585304104, 5.8591403032, 4.9780687254, 6.9522959657, 6.4036420727, 5.1356812086, 7.4400261829, 5.9805752222, 7.1038977412, 7.9421856179, 7.3121271211, 8.5915587323, 5.7635439967, 7.84388908, 6.525696556, 6.3270688572, 6.7528239738, 7.1595087113, 7.717170194, 6.070770146]

        expected_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.3867062080000068, 0.0, 0.0, 0.0, 0.0, 1.8653353239999857, 6.55166168400001, 0.0, 30.41397766399999, 0.0, 2.2899162399999895, 0.2867276679999975, 29.40119891199998, 30.69794231199999, 10.637637688000012, 16.467288131999993, 0.0, 0.0, 8.431501819999987, 20.136762988, 46.346144854, 49.27723979200002, 13.92279254799998, 17.399880168000017, 1.775952328000017, 0.0, 0.0, 100.05095278399997, 10.093222958000005, 11.773899312000005, 1.6437990420000048, 39.870625812, 0.0, 18.060582689999993, 8.934279671999992, 37.41986883199999, 15.912094182, 5.966114200000003, 30.28164020399998, 47.322400784, 37.080793428000014, 47.22883371599999, 17.785042666000006, 26.83582321199999, 18.32636537400001, 30.341216415999988, 17.182806064000005, 0.0, 58.09183862800001, 36.14568290800001, 2.7136241720000065, 86.40157097399998, 19.611504443999994, 66.233864472, 116.53113707399999, 78.727627266, 214.64939716799992, 15.270879934000003, 110.63334480000002, 41.02786223999999, 33.08275428799998, 50.112958952, 69.570522678, 103.03021164, 22.830805840000004]        
        student_output = wait_a_bit(water_level, water_level_loss_no_prevention, water_level_loss_with_prevention)

        self.assertIsNotNone(student_output, "wait_a_bit() returned None instead of a list!")
        self.assertTrue(type(student_output)==type(expected_output), f"Expected wait_a_bit() to return {type(expected_output)}. Got {type(student_output)}")
        self.assertEqual(len(expected_output), len(student_output), f"Expected wait_a_bit() to return list of length {len(expected_output)}. Got {len(student_output)}")
        
        epsilon = 1e-2

        for i in range(len(expected_output)):
            self.assertAlmostEqual(expected_output[i], student_output[i], delta=epsilon, msg=f"For wait_a_bit(), for year {2020+i}, expected damage costs of {expected_output[i]}. Got {student_output[i]}")

    def test_part_2_2c_prepare_immediately(self):
        np.random.seed(0)
        random.seed(0)

        water_level_loss_with_prevention = np.array([[5,6,7,8,9,10],[0,5,15,30,70,100]]).T

        water_level = [4.4181169189, 4.2432523603, 4.3706582199, 4.6214625724, 4.5992644453, 4.1217571709, 4.5090599508, 4.3305540324, 4.36929164, 4.5048728959, 4.4806502956, 4.8077094626, 4.6958176721, 4.5842193972, 4.7006383708, 4.7112153245, 5.0596676552, 4.6382490475, 4.8194558746, 4.5131621302, 4.0239461998, 5.0466333831, 5.1637915421, 4.667325595, 5.7603494416, 4.4694850712, 5.057247906, 5.0071681917, 5.7350299728, 5.7674485578, 5.2659409422, 5.4116822033, 4.8851984421, 4.4017149723, 5.2107875455, 5.5034190747, 6.1057690809, 6.1546206632, 5.3480698137, 5.4349970042, 5.0443988082, 4.8532731338, 4.7026613805, 7.0006369098, 5.5046611479, 5.5886949656, 5.0821899521, 6.4967656453, 4.8816427442, 5.9030291345, 5.4467139836, 6.4354967208, 5.7956047091, 5.29830571, 6.2570410051, 6.6830600196, 6.4270198357, 6.6807208429, 5.8892521333, 6.1708955803, 5.9163182687, 6.2585304104, 5.8591403032, 4.9780687254, 6.9522959657, 6.4036420727, 5.1356812086, 7.4400261829, 5.9805752222, 7.1038977412, 7.9421856179, 7.3121271211, 8.5915587323, 5.7635439967, 7.84388908, 6.525696556, 6.3270688572, 6.7528239738, 7.1595087113, 7.717170194, 6.070770146]

        expected_output = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.1933531040000034, 0.0, 0.0, 0.0, 0.0, 0.9326676619999928, 3.275830842000005, 0.0, 15.206988831999995, 0.0, 1.1449581199999948, 0.14336383399999875, 14.70059945599999, 15.348971155999996, 5.318818844000006, 8.233644065999997, 0.0, 0.0, 4.2157509099999935, 10.068381494, 24.230763236, 26.18482652800001, 6.96139627399999, 8.699940084000009, 0.8879761640000084, 0.0, 0.0, 60.03821458799998, 10.093222958000005, 11.773899312000005, 1.6437990420000048, 39.870625812, 0.0, 18.060582689999993, 8.934279671999992, 37.41986883199999, 15.912094182, 5.966114200000003, 30.28164020399998, 47.322400784, 37.080793428000014, 47.22883371599999, 17.785042666000006, 26.83582321199999, 18.32636537400001, 30.341216415999988, 17.182806064000005, 0.0, 58.09183862800001, 36.14568290800001, 2.7136241720000065, 86.40157097399998, 19.611504443999994, 66.233864472, 116.53113707399999, 78.727627266, 214.64939716799992, 15.270879934000003, 110.63334480000002, 41.02786223999999, 33.08275428799998, 50.112958952, 69.570522678, 103.03021164, 22.830805840000004]        
        student_output = prepare_immediately(water_level, water_level_loss_with_prevention)

        self.assertIsNotNone(student_output, "prepare_immediately() returned None instead of a list!")
        self.assertTrue(type(student_output)==type(expected_output), f"Expected prepare_immediately() to return {type(expected_output)}. Got {type(student_output)}")
        self.assertEqual(len(expected_output), len(student_output), f"Expected prepare_immediately() to return list of length {len(expected_output)}. Got {len(student_output)}")
        
        epsilon = 1e-2

        for i in range(len(expected_output)):
            self.assertAlmostEqual(expected_output[i], student_output[i], delta=epsilon, msg=f"For prepare_immediately(), for year {2020+i}, expected damage costs of {expected_output[i]}. Got {student_output[i]}")


# Dictionary mapping function names from the above TestCase class to
# the point value each test is worth.
point_values = {
    'test_part_1_1_predicted_sea_level_rise': 1,
    'test_part_1_2_simulate_year_1': 0.5,
    'test_part_1_2_simulate_year_2': 0.5,
    'test_part_2_1_water_level_est': 1,
    'test_part_2_2a_repair_only': 0.33,
    'test_part_2_2b_wait_a_bit': 0.33,
    'test_part_2_2c_prepare_immediately': 0.34
}

class Results_600(unittest.TextTestResult):
    # We override the init method so that the Result object can store the score and appropriate test output.
    def __init__(self, *args, **kwargs):
        super(Results_600, self).__init__(*args, **kwargs)
        self.output = []
        self.points = sum(point_values.values())
        self.total_points = sum(point_values.values())

    def addFailure(self, test, err):
        test_name = test._testMethodName
        msg = str(err[1])
        self.handleDeduction(test_name, msg)
        super(Results_600, self).addFailure(test, err)

    def addError(self, test, err):
        test_name = test._testMethodName
        self.handleDeduction(test_name, None)
        super(Results_600, self).addError(test, err)

    def handleDeduction(self, test_name, message):
        point_value = point_values[test_name]
        if message is None:
            message = 'Your code produced an error on test %s.' % test_name
        self.output.append('[-%s]: %s' % (point_value, message))
        self.points -= point_value

    def getOutput(self):
        if len(self.output) == 0:
            return "All correct!"
        return '\n'.join(self.output)

    def getPoints(self):
        return self.points

    def getTotalPoints(self):
        return self.total_points


if __name__=="__main__":
    try:
        STUDENT_KERBEROS = sys.argv[1]
    except IndexError as e:
        STUDENT_KERBEROS = ''
    print("Running unit tests for student %s" % STUDENT_KERBEROS)

    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestPS4))
    result = unittest.TextTestRunner(verbosity=2, resultclass=Results_600).run(suite)

    output = result.getOutput()
    points = result.getPoints()

    # weird bug with rounding
    if points < .1:
        points = 0

    print("\nProblem Set 4 Unit Test Results:")
    print(output)
    print(f"Points: {points}/{result.getTotalPoints()}\n")
