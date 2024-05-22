""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""
import pandas
import joblib
import numpy
import matplotlib
import seaborn
import itertools
import re

enron_data = joblib.load(open(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\final_project\final_project_dataset.pkl", "rb"))
enron_dataframe = pandas.DataFrame(enron_data)
print("Datapoints:",len(enron_data))
print("Features:", len(enron_data[enron_dataframe.keys()[0]]))

poidatasetcount = 0
for name, features in enron_dataframe.items():
    if features['poi']:
        poidatasetcount+=1
print("Number of POIs in DataSet:",poidatasetcount)

poiallcount = 0
with open(r"C:\Users\dell\OneDrive\Documents\VS Codes\Udacity-UD120-Migrated-to-Python-3\final_project\poi_names.txt") as f:
    content = f.readlines()
for lines in content:
    if re.match(r"\((y|n)\)", lines):
        poiallcount+=1
print("Number of POIs in total:",poiallcount)

print("Total stock value of Prentice James:", enron_dataframe["PRENTICE JAMES"]['total_stock_value'])

print("Total emails by Wesley Colwell to POIs:", enron_dataframe["COLWELL WESLEY"]['from_this_person_to_poi'])

print("Total value of exercised stock options of Jeff Skilling:", enron_dataframe["SKILLING JEFFREY K"]['exercised_stock_options'])

maxpay = max(enron_dataframe["SKILLING JEFFREY K"]['total_payments'],enron_dataframe["LAY KENNETH L"]['total_payments'],enron_dataframe["FASTOW ANDREW S"]['total_payments'])
print("Most Money Earned:",maxpay)

salariescount = 0
emailscount = 0
totalpaymentscount = 0
totalpaymentspoicount = 0
for name in enron_data:
    if not numpy.isnan(float(enron_data[name]['salary'])):
        salariescount += 1
    if enron_data[name]['email_address'] != "NaN":
        emailscount += 1
    if numpy.isnan(float(enron_data[name]['total_payments'])):
        totalpaymentscount += 1
        if enron_data[name]['poi']:
            totalpaymentspoicount += 1
print("Total Salaries Available:",salariescount)
print("Total Emails Available:",emailscount)
print("NaN for total payments:",totalpaymentscount)
print("NaN Percentage for total payments:",float(totalpaymentscount)/len(enron_data)*100)
print("NaN for total payments of POIs:",totalpaymentspoicount)
print("NaN Percentage for total payments of POIs:",float(totalpaymentspoicount)/poidatasetcount*100)
print("Total Number of People in the DataSet:",len(enron_data))