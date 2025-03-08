from datetime import date

def calculate_days_between(date1, date2):
    delta = date2 - date1
    return delta.days

date1 = date(2023, 11, 21)
date2 = date(2024, 2, 15)

days_between = calculate_days_between(date1, date2)

print("Number of days:", type(days_between))