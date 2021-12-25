# Problem Set 4
# Miguel Chacon
# Collaborators: N/A
# Time: 8 hours

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.interpolate import interp1d

#####################
# Begin helper code #
#####################

def calculate_std(upper, mean):
    """
	Calculate standard deviation based on the upper 95th percentile

	Args:
		upper: a 1-d numpy array with length N, representing the 95th percentile
            values from N data points
		mean: a 1-d numpy array with length N, representing the mean values from
            the corresponding N data points

	Returns:
		a 1-d numpy array of length N, with the standard deviation corresponding
        to each value in upper and mean
	"""
    return (upper-mean)/st.norm.ppf(.95)

def interp(target_year, input_years, years_data):
    """
	Interpolates data for a given year, based on the data for the years around it

	Args:
		target_year: an integer representing the year which you want the predicted
            sea level rise for
		input_years: a 1-d numpy array that contains the years for which there is data
		    (can be thought of as the "x-coordinates" of data points)
        years_data: a 1-d numpy array representing the current data values
            for the points which you want to interpolate, eg. the SLR mean per year data points
            (can be thought of as the "y-coordinates" of data points)

	Returns:
		the interpolated predicted value for the target year
	"""
    return np.interp(target_year, input_years, years_data, right=-99)

def load_slc_data():
    """
	Loads data from sea_level_change.csv and puts it into a numpy array

	Returns:
		a length 3 tuple of 1-d numpy arrays:
		    1. an array of years as ints
		    2. an array of 2.5th percentile sea level rises (as floats) for the years from the first array
		    3. an array of 97.5th percentile of sea level rises (as floats) for the years from the first array
        eg.
            (
                [2020, 2030, ..., 2100],
                [3.9, 4.1, ..., 5.4],
                [4.4, 4.8, ..., 10]
            )
            can be interpreted as:
                for the year 2020, the 2.5th percentile SLR is 3.9ft, and the 97.5th percentile would be 4.4ft.
	"""
    df = pd.read_csv('sea_level_change.csv')
    df.columns = ['Year','Lower','Upper']
    return (df.Year.to_numpy(),df.Lower.to_numpy(),df.Upper.to_numpy())

###################
# End helper code #
###################


##########
# Part 1 #
##########

def predicted_sea_level_rise(show_plot=False):
    """
	Creates a numpy array from the data in sea_level_change.csv where each row
    contains a year, the mean sea level rise for that year, the 2.5th percentile
    sea level rise for that year, the 97.5th percentile sea level rise for that
    year, and the standard deviation of the sea level rise for that year. If
    the year is between 2020 and 2100 and not included in the data, the values
    for that year should be interpolated. If show_plot, displays a plot with
    mean and the 95%, assuming sea level rise follows a linear trend.

	Args:
		show_plot: displays desired plot if true

	Returns:
		a 2-d numpy array with each row containing a year in order from 2020-2100
        inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
        deviation of the sea level rise for the given year
	"""
    # get data from file
    init_data = load_slc_data()
    sea_level_data = []
    # get means of the available year
    means = []
    for year in range(len(init_data[0])):
        means.append((init_data[1][year] + init_data[2][year]) / 2)
    # for every year
    for year in range(2020, 2101):
        # interpret 2.5th and 97.5th percentiles and mean for each year
        lower_percentile = interp(year, init_data[0], init_data[1])
        upper_percentile = interp(year, init_data[0], init_data[2])
        mean = interp(year, init_data[0], means)

        sea_level_data.append([year, mean, lower_percentile, upper_percentile, calculate_std(upper_percentile, mean)])
    sea_level_data = np.array(sea_level_data)  
    if show_plot:
        plt.plot(sea_level_data[:, 0], sea_level_data[:, 3], label = "Upper", linestyle = "--")
        plt.plot(sea_level_data[:, 0], sea_level_data[:, 2], label = "Lower", linestyle = "--")
        plt.plot(sea_level_data[:, 0], sea_level_data[:, 1], label = "Mean")
        plt.legend()
        plt.ylabel("Projected Annual Mean Water Level (ft)")
        plt.xlabel("Year")
        plt.title("Expected Results")
        plt.show()
    return sea_level_data


def simulate_year(data, year, num):
    """
	Simulates the sea level rise for a particular year based on that year's
    mean and standard deviation, assuming a normal distribution.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
		year: the year to simulate sea level rise for
        num: the number of samples you want from this year

	Returns:
		a 1-d numpy array of length num, that contains num simulated values for
        sea level rise during the year specified
	"""
    return np.random.normal(data[year-2020, 1], data[year-2020, 4], num)


def plot_mc_simulation(data):
    """
	Runs and plots a Monte Carlo simulation, based on the values in data and
    assuming a normal distribution. Five hundred samples should be generated
    for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year
	"""
    simulation_data = {year: simulate_year(data, year, 500) for year in range(2020, 2101)}
    plt.plot(data[:, 0], data[:, 3], label = "Upper", linestyle = "--")
    plt.plot(data[:, 0], data[:, 2], label = "Lower", linestyle = "--")
    plt.plot(data[:, 0], data[:, 1], label = "Mean")
    for (x, y_s) in simulation_data.items():
        plt.scatter([x] * 500, y_s, c = '#808080', alpha = 0.25, s = 1)
    plt.legend()
    plt.ylabel("Projected Annual Mean Water Level (ft)")
    plt.xlabel("Year")
    plt.title("Expected Results")
    plt.show()


##########
# Part 2 #
##########

def water_level_est(data):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, mean, the 2.5th percentile, 97.5th percentile, and standard
            deviation of the sea level rise for the given year

	Returns:
		a list of simulated water levels for each year, in the order in which
        they would occur temporally
	"""
    return [simulate_year(data, year, 1)[0] for year in range(2020, 2101)]

def repair_only(water_level_list, water_level_loss_no_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a repair only strategy, where you would only pay
    to repair damage that already happened.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the first column is
            the SLR levels and the second column is the corresponding property damage expected
            from that water level with no flood prevention (as an integer percentage)
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    f = interp1d(water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1], fill_value = "extrapolate")
    damages = []
    for water_level in water_level_list:
        percentage = f(water_level)
        if percentage < 100 and percentage > 0:
            damages.append(percentage*house_value/100_000)
        elif percentage >= 100:
            damages.append(house_value/100_000)
        else:
            damages.append(0)
    return damages


def wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a wait a bit to repair strategy, where you start
    flood prevention measures after having a year with an excessive amount of
    damage cost.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_no_prevention and water_level_loss_with_prevention, where
    each water level corresponds to the percent of property that is damaged.
    You should be using water_level_loss_no_prevention when no flood prevention
    measures are in place, and water_level_loss_with_prevention when there are
    flood prevention measures in place.

    Flood prevention measures are put into place if you have any year with a
    damage cost above the cost_threshold.

    The wait a bit to repair only strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    # get the functions of both lists provided
    f_no_prevention = interp1d(water_level_loss_no_prevention[:,0], water_level_loss_no_prevention[:,1], fill_value = "extrapolate")
    f_prevention = interp1d(water_level_loss_with_prevention[:,0], water_level_loss_with_prevention[:,1], fill_value = "extrapolate")
    def get_annual_damage(percentage, house_value):
        """
        helper function that ensures numbers make physial sense 
        (i.e no negatives and not greater than house value)
        """
        if percentage < 100 and percentage > 0:
            return percentage*house_value/100_000
        elif percentage >= 100:
            return house_value/100_000
        else:
            return 0
    damages = []
    # use the no cost prevention until a cost_threshold has been reached
    index = 0
    while (f_no_prevention(water_level_list[index]) * house_value)/100 <= cost_threshold:
        damages.append(get_annual_damage(f_no_prevention(water_level_list[index]), house_value))
        index += 1
    # still use the no prevention cost for the year that threshold was reached
    damages.append(get_annual_damage(f_no_prevention(water_level_list[index]), house_value))
    # use prevention cost for the rest of the years
    for i in range(index+1, 81):
        damages.append(get_annual_damage(f_prevention(water_level_list[i]), house_value))
    return damages


def prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value=400000):
    """
	Simulates the water level for all years in the range 2020 to 2100, inclusive,
    and calculates damage costs in 1000s resulting from a particular water level
    for each year dependent on a prepare immediately strategy, where you start
    flood prevention measures immediately.

    The specific damage cost can be calculated using the numpy array
    water_level_loss_with_prevention, where each water level corresponds to the
    percent of property that is damaged.

    The prepare immediately strategy is as follows:
        1) If the water level is less than or equal to 5ft, the cost is 0.
        2) If the water level is between 5ft and 10ft, the cost is the
           house_value times the percentage of property damage for that water
           level, which is affected by the implementation of flood prevention
           measures. If the water level is not an integer value, the percentage
           should be interpolated.
        3) If the water level is at least 10ft, the cost is the entire value of
           the house.

	Args:
		water_level_list: list of simulated water levels for 2020-2100
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for

	Returns:
		an list of damage costs in 1000s, in the order in which the costs would
        be incurred temporally
	"""
    f_prevention = interp1d(water_level_loss_with_prevention[:,0], water_level_loss_with_prevention[:,1], fill_value = "extrapolate")
    damages = []
    for water_level in water_level_list:
        percentage = f_prevention(water_level)
        if percentage < 100 and percentage > 0:
            damages.append(percentage*house_value/100_000)
        elif percentage >= 100:
            damages.append(house_value/100_000)
        else:
            damages.append(0)
    return damages


def plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value=400000, cost_threshold=100000):
    """
	Runs and plots a Monte Carlo simulation of all of the different preparation
    strategies, based on the values in data and assuming a normal distribution.
    Five hundred samples should be generated for each year.

	Args:
		data: a 2-d numpy array with each row containing a year in order from 2020-2100
            inclusive, the 5th percentile, 95th percentile, mean, and standard
            deviation of the sea level rise for the given year
        water_level_loss_no_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with no flood prevention
        water_level_loss_with_prevention: a 2-d numpy array where the columns are
            water levels and the corresponding percent of property damage expected
            from that water level with flood prevention
        house_value: the value of the property we are estimating cost for
        cost_threshold: the amount of cost incurred before flood prevention
            measures are put into place
	"""
    years = list(range(2020, 2101))
    # 81 by 500 array. rows are years and columns are simulations
    # every entry is the j-th simulation of the i-th year
    simulation_data = np.array([simulate_year(data, year, 500) for year in years])
    # for every simulation...
    for simulation in range(500):
        # get simulated water levels
        water_level_list = simulation_data[:,simulation]

        # get damages for each strategy
        damage_repair_only = repair_only(water_level_list, water_level_loss_no_prevention, house_value)
        damage_wait_a_bit = wait_a_bit(water_level_list, water_level_loss_no_prevention, water_level_loss_with_prevention, house_value, cost_threshold)
        damage_prepare_immediately = prepare_immediately(water_level_list, water_level_loss_with_prevention, house_value)

        # plot the points for each simulation
        plt.scatter(years, damage_repair_only, c = '#00b300', alpha = 0.5, s = 1)
        plt.scatter(years, damage_wait_a_bit, c = '#0000FF', alpha = 0.5, s = 1) 
        plt.scatter(years, damage_prepare_immediately, c = '#FF0000', alpha = 0.5, s = 1)

    # plot the mean values for each simulation
    mean_water_levels = [sum(simulation)/500 for simulation in simulation_data]
    damage_repair_only = repair_only(mean_water_levels, water_level_loss_no_prevention)
    damage_wait_a_bit = wait_a_bit(mean_water_levels, water_level_loss_no_prevention, water_level_loss_with_prevention)
    damage_prepare_immediately = prepare_immediately(mean_water_levels, water_level_loss_with_prevention)

    plt.plot(years, damage_repair_only, label = "Repair-only Scenario", c="#00b300")
    plt.plot(years, damage_wait_a_bit, label = "Wait-a-bit Scenario", c="#0000FF") 
    plt.plot(years, damage_prepare_immediately, label = "Prepare-immediately scenario", c="#FF0000")
    plt.legend()
    plt.ylabel("Estimated Damage Cost ($K)")
    plt.xlabel("Year")
    plt.title("Annual Average Damage Cost")
    plt.show()


if __name__ == '__main__':
    data = predicted_sea_level_rise()
    water_level_loss_no_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 10, 25, 45, 75, 100]]).T
    water_level_loss_with_prevention = np.array([[5, 6, 7, 8, 9, 10], [0, 5, 15, 30, 70, 100]]).T
    plot_mc_simulation(data)
    plot_strategies(data, water_level_loss_no_prevention, water_level_loss_with_prevention)
