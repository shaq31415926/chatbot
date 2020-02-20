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
    Purpose of this Chatbot to stores information about the users trucks. Every conversation is recorded.
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

    def display_model_names(self, data, manufacturer_name):
        """displays list of models based on manufacturer name.
        For now assuming if we know the manufacturer name, we can match the model name"""
        model_name_options = data['model_name'][data['manufacturer_name_lower'] == manufacturer_name.lower()]

        model_selector_options = []

        for n, name in enumerate(model_name_options):
            model_selector_options.append({'selector': n + 1,
                                           'prompt': name,
                                           'return': name
                                           })

        model_selector_options.append({'selector': len(model_selector_options) + 1,
                                        'prompt': 'Other',
                                        'return': 'other'
                                        })

        return model_selector_options

    def store_new_truck(self, manufacturer_name, model_name):
        """where manufacturer name and/or model name is unknown, stores to truck dataset"""
        new_truck = pd.DataFrame.from_dict({"manufacturer_name": [manufacturer_name],
                                            "model_name": [model_name],
                                            "class": "Unknown",
                                            "cabin": "Unknown",
                                            "country_of_origin": "Unknown",
                                            "manufacturer_name_lower": [manufacturer_name.lower()],
                                            "model_name_lower": [model_name.lower()],
                                            "user_entry": 1
                                            # future development would include maintaining and tidying the manual entries
                                            })

        new_truck.to_csv(os.path.join(self.dirname, "../data/input data/trucks.csv"), mode='a', header=False,
                         index=False)


    def collect_truck(self):
        """"collects manufacturer and model name and matches against the trucks dataset"""
        known_trucks = pd.read_csv(os.path.join(self.dirname,"../data/input data/trucks.csv"))

        # ask for manufacturer name
        manufacturer_name = self.ask("What is the name of the manufacturer of your truck?")

        matching_manufacturer = []

        if manufacturer_name.lower() in np.unique(known_trucks['manufacturer_name_lower']):
            model_selector_options = self.display_model_names(known_trucks, manufacturer_name)
            model_name = self.ask_multiple("Please select the name of your model?", model_selector_options)
            if model_name == "other":
                model_name = self.ask("Could you please tell me, what is model name of the your truck?")
                self.store_new_truck(manufacturer_name, model_name)  # save data to dataset
            self.answer("Thanks! I would like to get to know more about your {}.".format(model_name))

        else:
            for manufacturer in np.unique(known_trucks['manufacturer_name']):
                dist = edit_distance(manufacturer_name.lower(), manufacturer.lower())
                matching_manufacturer.append({"name": manufacturer,
                                       "dist": dist
                                       })

            matching_manufacturer_df = pd.DataFrame(matching_manufacturer)
            matched_manufacturer = matching_manufacturer_df['name'].loc[matching_manufacturer_df['dist'].idxmin()]

            manufacturer_options = [{'selector': '1', 'prompt': 'Yes', 'return': 'yes'},
                                    {'selector': '2', 'prompt': 'No', 'return': 'no'}
                                    ]

            manufacturer_matched = self.ask_multiple("Is {} the name of the manufacturer?".format(matched_manufacturer),\
                                                     manufacturer_options)

            if manufacturer_matched == "yes":
                manufacturer_name = matched_manufacturer

                model_selector_options = self.display_model_names(known_trucks, manufacturer_name)
                model_name = self.ask_multiple("Please select the name of your model?", model_selector_options)
                if model_name == "other":
                    model_name = self.ask("Could you please tell me, what is model name of the your truck?")
                    self.store_new_truck(manufacturer_name, model_name)  # save data to dataset
                self.answer("Thanks! I would like to get to know more about your {}".format(model_name))

            else:
                self.answer("I was not aware of this one. Could I get some more information so I can add {} to my database.".\
                            format(manufacturer_name))
                model_name = self.ask("What is model name of the your truck?")

                self.store_new_truck(manufacturer_name, model_name) # save data to dataset

        return manufacturer_name, model_name

    def store_data(self, str_dat_dict):
        """store the data we are interested in"""
        self.store_dat.append(str_dat_dict)
        pd.DataFrame(self.store_dat).to_csv(os.path.join(self.dirname,"../data/captured data/final_data.csv"), \
                                            mode='a', header=False, index=False)

    def start_chat(self):
        """run to start chat and ask a series of questions"""
        puts(colored.cyan("Hey! I'm {} :) Let's talk trucks.".format(self.name)))

        # data compliance check
        puts(colored.red("Please note we keep a copy of all conversations, and use this information to learn more about your trucks"))
        data_compliance_options = [{'selector': '1', 'prompt': 'Yes', 'return': 'yes'},
                                   {'selector': '2', 'prompt': 'No', 'return': 'no'}
                                  ]
        data_compliance_check = self.ask_multiple("Are you happy to proceed?", data_compliance_options)

        if data_compliance_check == "no":
            puts(colored.cyan("Thanks! Bye for now :)"))
            exit()

        # start chat once data compliance check is completed
        username = self.ask("Let's start.. What is your name?")
        self.answer("Hi {}!".format(username))

        # ask occupation - if they are not owners or managers, exit chat
        occupations = [{'selector': '1', 'prompt': 'Fleet Owner', 'return': 'owner'},
                       {'selector': '2', 'prompt': 'Manager', 'return': 'manager'},
                       {'selector': '3', 'prompt': 'Other', 'return': 'other'}]

        occupation = self.ask_multiple("What is your occupation?", occupations)

        if occupation == "other":
            self.finish_chat(occupation, {})
            exit()

        # collect information on manufacturer and model name
        manufacturer_name, model_name = self.collect_truck()

        # collect fleet number
        fleet_number = self.get_numeric_input("What is the fleet number of your truck?")

        # collect some details on the truck specifications
        cylinders = self.get_numeric_input("How many cylinders does your truck have?")
        horsepower = self.get_numeric_input("What is the horsepower of your truck?")
        weight = self.get_numeric_input("How much does your truck roughly weigh?")

        dat_dict = {'id': str(uuid.uuid4()),
                    'started': self.startTime,
                    'finished': datetime.datetime.now(),
                    'occupation': occupation,
                    'manufacturer_name': manufacturer_name.lower(),
                    'model_name': model_name.lower(),
                    'fleet_number': fleet_number,
                    'cylinders': cylinders,
                    'horsepower': horsepower,
                    'weight': weight
                    }

        self.finish_chat(occupation, dat_dict)


truck_chatbot = Chatbot(name="Sarah")
truck_chatbot.start_chat()
