import pandas as pd

dataset = 'Affordance_November19_alldata'
file_name = 'Datasets/cleansed_data/'+dataset+"cleansed_validated_new_data.csv"
response_table = pd.read_csv(file_name)

res_data = response_table[['ResponseId','device_use','Q20']]

#short_table = short_table.drop([0, 1]).dropna(how='any')


#create a mapping from the choice to the numeric data
#order of both 'Device Use' and 'Q20' codes are the same so we can use one map 
device_map = {'Laptop':'1', 'Smart_Phone':'2', 'Desktop_Computer':'3', 'Tablet':'4', 'Smart_Speaker':'5', 'Smart_Watch':'6'}

#print(short_table[short_table['device_use'].str.contains(device_map['Laptop'])])


# laptop_table = short_table[short_table['device_use'].str.contains(device_map['Laptop'])]

# laptop_table['Actual_Use'] = (laptop_table['Q20'].str.contains(device_map['Laptop'])).astype(int)

# SmartPhone_table = short_table[short_table['device_use'].str.contains(device_map['Smart Phone'])]

# SmartPhone_table['Actual_Use'] = (SmartPhone_table['Q20'].str.contains(device_map['Smart Phone'])).astype(int)

# DesktopComputer_table = short_table[short_table['device_use'].str.contains(device_map['Desktop Computer'])]

# DesktopComputer_table['Actual_Use'] = (DesktopComputer_table['Q20'].str.contains(device_map['Desktop Computer'])).astype(int)

# Tablet_table = short_table[short_table['device_use'].str.contains(device_map['Tablet'])]
# Tablet_table['Actual_Use'] = (Tablet_table['Q20'].str.contains(device_map['Tablet'])).astype(int)

# SmartSpeaker_table = short_table[short_table['device_use'].str.contains(device_map['Smart Speaker'])]
# SmartSpeaker_table['Actual_Use'] = (SmartSpeaker_table['Q20'].str.contains(device_map['Smart Speaker'])).astype(int)

# SmartWatch_table = short_table[short_table['device_use'].str.contains(device_map['Smart Watch'])]
# SmartWatch_table['Actual_Use'] = (SmartWatch_table['Q20'].str.contains(device_map['Smart Watch'])).astype(int)

# #some tests for validation
# laptop_table.to_csv("Laptop_data.csv")
# SmartPhone_table.to_csv("smartphone_data.csv")
# DesktopComputer_table.to_csv("Desktop_data.csv")
# Tablet_table.to_csv("tablet_data.csv")
# SmartSpeaker_table.to_csv("smartspeaker_data.csv")
# SmartWatch_table.to_csv("smartWatch_data.csv")


laptop_data = res_data[res_data['device_use'].str.contains('1')]
laptop_data['actual_use'] = (res_data['Q20'].str.contains('1')).astype(int)

smartphone_data = res_data[res_data['device_use'].str.contains('2')]
smartphone_data['actual_use'] = (res_data['Q20'].str.contains('2')).astype(int)

desktop_data = res_data[res_data['device_use'].str.contains('3')]
desktop_data['actual_use'] = (res_data['Q20'].str.contains('3')).astype(int)

tablet_data = res_data[res_data['device_use'].str.contains('4')]
tablet_data['actual_use'] = (res_data['Q20'].str.contains('4')).astype(int)

smartspeaker_data = res_data[res_data['device_use'].str.contains('5')]
smartspeaker_data['actual_use'] = (res_data['Q20'].str.contains('5')).astype(int)

smartwatch_data = res_data[res_data['device_use'].str.contains('6')]
smartwatch_data['actual_use'] = (res_data['Q20'].str.contains('6')).astype(int)

print('total number of data points:' , res_data.shape[0])
print('number of laptop_points:', (laptop_data.shape[0]),",no_of_(1's):", laptop_data["actual_use"].value_counts()[1])
print('number of smartphone_points:', (smartphone_data.shape[0]),",no_of_(1's):", smartphone_data["actual_use"].value_counts()[1])
print('number of desktop_points:', (desktop_data.shape[0]),",no_of_(1's):", desktop_data["actual_use"].value_counts()[1])
print('number of tablet_points:', (tablet_data.shape[0]),",no_of_(1's):", tablet_data["actual_use"].value_counts()[1])
print('number of smartspeaker_points:', (smartspeaker_data.shape[0]),",no_of_(1's):", smartspeaker_data["actual_use"].value_counts()[1])
print('number of smartwatch_points:', (smartwatch_data.shape[0]),",no_of_(1's):", smartwatch_data["actual_use"].value_counts()[1])






