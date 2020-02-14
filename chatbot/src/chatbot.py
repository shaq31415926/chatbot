import os
import uuid
import logging
from nltk.metrics import edit_distance
from clint.textui import prompt, puts, colored
from word2number import w2n
import pandas as pd
import datetime
import numpy as np

class Chatbot(object):
    """
    Chatbot chats with you about trucks. It stores information about trucks and keeps a transcript of each chat.
    """

    def __init__(self, name):
        self.name = name
        self.startTime = datetime.datetime.now()
        self.transcript = "{}_transcript".format(self.startTime.strftime("%Y%m%d%H%M"))
        self.dirname = os.path.dirname(__file__)
        logging.basicConfig(filename=os.path.join(self.dirname, '../data/transcript logs/{}.csv'.format(self.transcript)),
                            filemode="w",
                            format='%(asctime)s %(message)s',
                            level=logging.INFO)
        self.logger = logging.getLogger('transcript_logger')
        self.store_transcript = {'questions': [], 'answers': []}
        self.store_dat = []

    def get_numeric_input(self, message):
        """converts word names to numbers"""
        input = self.ask(message)
        if not input.isnumeric():
            try:
                input = w2n.word_to_num(input)
            except:
                puts(colored.red("That wasn't a number, please answer the question again"))
                input = self.get_numeric_input(message)
        return input

    def ask(self, question):
        """used when chatbot does not want to fix the user input"""
        response = prompt.query(question)
        self.record(question, response)
        return response

    def ask_multiple(self, question, options):
        """use when chatbot wants user to select a predefined options"""
        response = prompt.options(question, options)
        self.record(question, response)
        return response

    def answer(self, message):
        """usee when chatbot wants to respond with only a statement"""
        puts(colored.blue(message))

    def record(self, question, response):
        """records the conversation"""
        self.store_transcript['questions'].append([question])
        self.store_transcript['answers'].append([response])
        self.logger.info("{} {}".format(question, response))

    def finish_chat(self, occupation, dat_dict):
        """final message before chatbot finishes"""
        self.store_transcript['finished'] = datetime.datetime.now()

        if occupation == "other":
            puts(colored.red("This is for owners and managers only. Farewell!"))
            dat_dict = {'id': str(uuid.uuid4()),
                        'started': self.startTime,
                        'finished': datetime.datetime.now(),
                        'occupation': occupation,
                        'truck_name': np.nan,
                        'fleet_number': np.nan,
                        'year': np.nan,
                        'cylinders': np.nan,
                        'horsepower': np.nan,
                        'weight': np.nan
                        }
            self.store_data(dat_dict)
        else:
            puts(colored.cyan("Thanks for talking about trucks with me today! Bye for now :)"))
            self.store_data(dat_dict)

    def collect_truck(self):
        """"collects truck name and matches against a predefined list of trucks"""
        known_trucks = pd.read_csv(os.path.join(self.dirname,"../data/input data/trucks.csv"))

        truck_name = self.ask("What is the name of your truck?")

        matching_truck = []

        if truck_name.lower() in np.unique(known_trucks['truck_name_lower']):
            self.answer("Thanks! I would like to get to know more about your {}.".format(truck_name))
        else:
            for truck in np.unique(known_trucks['truck_name']):
                dist = edit_distance(truck_name.lower(), truck.lower())
                matching_truck.append({"name": truck,
                                       "dist": dist
                                      })

            matching_truck_df = pd.DataFrame(matching_truck)
            matched_truck = matching_truck_df['name'].loc[matching_truck_df['dist'].idxmin()]

            truck_options = [{'selector': '1', 'prompt': 'Yes', 'return': 'yes'},
                             {'selector': '2', 'prompt': 'No', 'return': 'no'}
                           ]

            truck_matched = self.ask_multiple("Is {} the name of your truck?".format(matched_truck), truck_options)

            if truck_matched == "yes":
                truck_name = matched_truck
                self.answer("Thanks! I would like to get to know more about your {}.".format(truck_name))
            else:
                self.answer("That's a new one! I'll add {} to my database.".format(truck_name))
                self.answer("Next, I would like to get to know more about your truck.")
                new_truck = pd.DataFrame.from_dict({"truck_name": [truck_name],
                                                    "truck_name_lower": [truck_name.lower()]
                                                    })
                new_truck.to_csv(os.path.join(self.dirname,"../data/input data/trucks.csv"), mode='a', header=False, index=False)

        return truck_name

    def store_data(self, str_dat_dict):
        """store the data we are interested in"""
        self.store_dat.append(str_dat_dict)
        pd.DataFrame(self.store_dat).to_csv(os.path.join(self.dirname,"../data/captured data/final_data.csv"), mode='a', header=False, index=False)

    def start_chat(self):
        """run to start chat and ask a series of questions"""

        puts(colored.cyan("Hey! I'm {} :) Let's talk trucks.".format(self.name)))
        username = self.ask("Let's start.. What is your name?")
        self.answer("Hi {}".format(username))

        # occupation
        occupations = [{'selector': '1', 'prompt': 'Fleet Owner', 'return': 'owner'},
                       {'selector': '2', 'prompt': 'Manager', 'return': 'manager'},
                       {'selector': '3', 'prompt': 'Other', 'return': 'other'}]

        occupation = self.ask_multiple("What is your occupation?", occupations)

        if occupation == "other":
            self.finish_chat(occupation, {})
            exit()

        # truck name
        truck_name = self.collect_truck()

        # fleet number
        fleet_number = self.get_numeric_input("What is your fleet number?")

        # get specification
        year = self.get_numeric_input("What year was your truck made?")  # if time, format year
        cylinders = self.get_numeric_input("How many cylinders does your truck have?")
        horsepower = self.get_numeric_input("What is the horsepower of the truck?")
        weight = self.get_numeric_input("How much does the truck weigh?")

        dat_dict = {'id': str(uuid.uuid4()),
                    'started': self.startTime,
                    'finished': datetime.datetime.now(),
                    'occupation': occupation,
                    'truck_name': truck_name.lower(),
                    'fleet_number': fleet_number,
                    'year': year,
                    'cylinders': cylinders,
                    'horsepower': horsepower,
                    'weight': weight
                    }

        self.finish_chat(occupation, dat_dict)


truck_chatbot = Chatbot(name="Sarah")
truck_chatbot.start_chat()
