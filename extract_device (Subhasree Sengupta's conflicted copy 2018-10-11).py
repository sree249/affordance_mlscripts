__author__ = 'billcarmack'


import pandas as pd

response_table = pd.read_csv('test_data.csv')

short_table = response_table[['ResponseId','device_use','Q20']]

short_table = short_table.drop([0, 1]).dropna(how='any')

#print(short_table)

laptop_table = short_table[short_table['device_use'].str.contains('Laptop')]

laptop_table['Actual_Use'] = (laptop_table['Q20'].str.contains('Laptop')).astype(int)

SmartPhone_table = short_table[short_table['device_use'].str.contains('Smart Phone')]

SmartPhone_table['Actual_Use'] = (SmartPhone_table['Q20'].str.contains('Smart Phone')).astype(int)

DesktopComputer_table = short_table[short_table['device_use'].str.contains('Desktop Computer')]

DesktopComputer_table['Actual_Use'] = (DesktopComputer_table['Q20'].str.contains('Desktop Computer')).astype(int)

Tablet_table = short_table[short_table['device_use'].str.contains('Tablet')]
Tablet_table['Actual_Use'] = (Tablet_table['Q20'].str.contains('Tablet')).astype(int)

SmartSpeaker_table = short_table[short_table['device_use'].str.contains('Smart Speaker')]
SmartSpeaker_table['Actual_Use'] = (SmartSpeaker_table['Q20'].str.contains('Smart Speaker')).astype(int)

SmartWatch_table = short_table[short_table['device_use'].str.contains('Smart Watch')]
SmartWatch_table['Actual_Use'] = (SmartWatch_table['Q20'].str.contains('Smart Watch')).astype(int)

print(SmartWatch_table)


