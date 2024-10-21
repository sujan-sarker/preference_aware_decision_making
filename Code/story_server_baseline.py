# imports for server-client communication

import socket

import os
import math
import random
import numpy as np
import spacy
from tabulate import tabulate
import nltk
from nltk.corpus import gutenberg
from collections import Counter
from scipy.spatial.distance import cosine
import json
import openai
import itertools
import re
import time
from itertools import combinations
from unidecode import unidecode


# Set server ip and port here
server_id = "localhost"
server_port = 8086



# Initialize global dictionaries
predicted_words = {}
contributions = {}
predicted_themes = {}
theme_list = ["mystery", "fairy tale", "teamwork", "adventure"]

### have to change before evey session
theme_2 = "adventure" 
theme_1 = "fairy tale"
robot_theme = ""



pid_1 = ""
pid_2 = ""
task_id = ""




nltk.download('gutenberg')

gutenberg_words = [word.lower() for word in gutenberg.words() if word.isalpha() and len(word) > 2 and len(word) < 10]
word_freq = Counter(gutenberg_words)
total_words = sum(word_freq.values())

#set openai api key here
client = openai.OpenAI(api_key="")





class Word:
    def __init__(self, text, nlp=None):
        self.text = text
        self.text_lower = text.lower()
        self.difficulty = None

        if self.text!="no_word" and "END":
            self.count = word_freq[self.text.lower()]
            self.frequency = self.count / total_words
            self.length = len(self.text)
        
    def __eq__(self, other):
        if isinstance(other, Word):
            return self.text_lower == other.text_lower
        return False

    def __hash__(self):
        return hash(self.text_lower)

class Story:
    def __init__(self, words, nlp, robot_theme, participant_1_theme, participant_2_theme):
        self.story = []
        self.story_start_time = None
        self.story_end_time = None
        self.words = [w if isinstance(w, Word) else Word(w, nlp) for w in words]
        self.remaining_words = self.words.copy()
        self.is_story_ends = False
        self.total_reward = 0
        
        self.robot_theme = robot_theme
        self.participant_1_theme = participant_1_theme
        self.participant_2_theme = participant_2_theme
        self.calculate_relative_difficulties()
        self.agent_contributions = {}
        self.agent_selected_words = {}
        self.agents = []
        self.turns = []

    
    def calculate_relative_difficulties(self):
        counts = [word.count for word in self.words]
        lengths = [word.length for word in self.words]

        min_count, max_count = min(counts), max(counts)
        min_length, max_length = min(lengths), max(lengths)

        for word in self.words:
            count_factor = (max_count - word.count) / (max_count - min_count) if max_count != min_count else 0.5
            length_factor = (word.length - min_length) / (max_length - min_length) if max_length != min_length else 0.5
            word.difficulty = round((0.8 * count_factor + 0.2 * length_factor) * 9 + 1)

    def start_story(self):
        # Capture the start time in UNIX format
        self.story_start_time = time.time()

    def end_story(self):
        # Capture the end time in UNIX format
        self.story_end_time = time.time()

    def update_total_reward(self, reward):
        self.total_reward += reward

    def append_to_story(self, narrator, narrative, word):
        self.story.append((narrator, narrative))
        if narrator not in self.agent_contributions:
            self.agent_contributions[narrator] = []
        self.agent_contributions[narrator].append(narrative)

        # Calculate and update the reward dynamically
        contribution_level = contributions.get(narrator, "moderate")
        contribution_score = WordDistributionMDP.contribution_level_to_value(contribution_level)
        reward = word.difficulty * contribution_score
        self.update_total_reward(reward)  # Add reward to the total

    def add_selected_word(self, agent_name, word):
        if agent_name not in self.agent_selected_words:
            self.agent_selected_words[agent_name] = []
        if word.text != "no_word":
            self.agent_selected_words[agent_name].append(word)

    def write_story_with_narrators(self, filename, session_string):
            total_time = self.story_end_time - self.story_start_time if self.story_end_time else 0
            with open(filename, 'w') as file:
                data = {
                    "session_string": session_string,
                    "story_start_time": self.story_start_time,
                    "story_end_time": self.story_end_time,
                    "total_time": total_time,
                    "total_reward": self.total_reward,
                    "robot_theme": self.robot_theme,
                    "participant_1_theme": self.participant_1_theme,
                    "participant_2_theme": self.participant_2_theme,
                    "words": [w.text for w in self.words],
                    "story": self.story,
                    "agent_contributions": self.agent_contributions,
                    "agent_selected_words": {agent: [w.text for w in words] for agent, words in self.agent_selected_words.items()},
                    "turns": self.turns,
                    "robot_predicted_words": self.agents[0].next_agent_predicted_words  # Assuming the robot is the first agent
                }
                json.dump(data, file, indent=4)

    def write_story_without_narrators(self, filename, session_string):
        total_time = self.story_end_time - self.story_start_time if self.story_end_time else 0
        with open(filename, 'w') as file:
            data = {
                "session_string": session_string,
                "story_start_time": self.story_start_time,
                "story_end_time": self.story_end_time,
                "total_time": total_time,
                "total_reward": self.total_reward,
                "robot_theme": self.robot_theme,
                "participant_1_theme": self.participant_1_theme,
                "participant_2_theme": self.participant_2_theme,
                "words": [w.text for w in self.words],
                "story": [narrative for _, narrative in self.story],
                "agent_contributions": self.agent_contributions,
                "agent_selected_words": {agent: [w.text for w in words] for agent, words in self.agent_selected_words.items()},
                "turns": self.turns,
                "robot_predicted_words": self.agents[0].next_agent_predicted_words  # Assuming the robot is the first agent
            }
            json.dump(data, file, indent=4)
    
    def update_turns(self, agent_name):
        self.turns.append(agent_name)


class WordDistributionMDP:
    def __init__(self, words, agents, first_agent, word_probs, agent_contributions):

           
        self.words = words
        self.agents = agents
        self.first_agent = first_agent
        self.word_probs = word_probs
        self.agent_contributions = agent_contributions
        self.all_words = words + [Word("no_word")]  # Add a placeholder for "no_word"
        self.num_words = len(self.all_words)
        self.num_agents = len(agents)
        
        self.states = self._generate_states()
        self.actions = self.all_words
        self.transition_probs = self._generate_transition_probs()
        self.rewards = self._generate_rewards()

    def _generate_states(self):
        states = []
        for agent in self.agents:
            for i in range(self.num_words + 1):
                for word_subset in self._get_word_subsets(i):
                    states.append((agent, frozenset(word_subset)))
        return states

    def _get_word_subsets(self, size):
        return combinations(self.all_words, size)

    def _normalize_probs(self, probs):
        print(f'In normalized probs: {probs}')
        # Convert all keys to lowercase
        probs = {k.lower(): v for k, v in probs.items()}
        total = sum(probs.values())
        if total == 0:
            return {word: 1 / len(probs) for word in probs}
        return {word: prob / total for word, prob in probs.items()}

    def _generate_transition_probs(self):
        global predicted_words, contributions
        transition_probs = {}
        for state in self.states:
            agent, words = state
            remaining_words = set(self.all_words) - set(words)
            if not remaining_words or remaining_words == {Word("no_word")}:
                transition_probs[state] = {action: {state: 1.0} for action in self.actions}
            else:
                next_agent = self.agents[(self.agents.index(agent) + 1) % self.num_agents]
                transition_probs[state] = {}
                agent_prob_dict = {item.split(':')[0].lower(): float(item.split(':')[1]) for item in self.word_probs[agent.name]}
                
                print(f'Agent {agent.name}: {agent_prob_dict}')

                normalized_probs = self._normalize_probs({w.text.lower(): agent_prob_dict.get(w.text.lower(), 0) for w in remaining_words})
                for action in remaining_words:
                    next_words = frozenset(words | {action}) if action.text != "no_word" else words
                    transition_probs[state][action] = {
                        (next_agent, next_words): normalized_probs[action.text.lower()],
                        (agent, words): 1 - normalized_probs[action.text.lower()]
                    }
                for action in set(self.actions) - remaining_words:
                    transition_probs[state][action] = {state: 1.0}
        return transition_probs

    def _generate_rewards(self):
        rewards = {}
        for state in self.states:
            agent, words = state
            rewards[state] = {
                action: self.calculate_reward(agent, action) for action in self.actions
            }
        return rewards

    def calculate_reward(self, agent, word):
        if word.text == "no_word":
            return 0
        contribution_level = self.agent_contributions.get(agent.name, 0)
        contribution_score = self.contribution_level_to_value(contribution_level)
        return contribution_score * word.difficulty
    
    @staticmethod
    def contribution_level_to_value(level):

        # print(f'Printing level: {level}')
        return {
            "very low": 0.2,
            "low": 0.4,
            "moderate": 0.6,
            "high": 0.8,
            "very high": 1.0
        }.get(level.strip(), 0.5)

    def value_iteration(self, gamma=0.9, epsilon=1e-6):
        V = {state: 0 for state in self.states}
        while True:
            delta = 0
            for state in self.states:
                v = V[state]
                V[state] = max(
                    sum(
                        prob * (self.rewards[state][action] + gamma * V[next_state])
                        for next_state, prob in self.transition_probs[state][action].items()
                    )
                    for action in self.actions
                )
                delta = max(delta, abs(v - V[state]))
            if delta < epsilon:
                break
        return V

    def get_optimal_policy(self, V):
        policy = {}
        for state in self.states:
            policy[state] = max(
                self.actions,
                key=lambda action: sum(
                    prob * (self.rewards[state][action] + 0.9 * V[next_state])
                    for next_state, prob in self.transition_probs[state][action].items()
                )
            )
        return policy

    def simulate_allocation(self, policy):
        allocation = []
        current_state = (self.first_agent, frozenset())
        safety_counter = 0
        max_iterations = 1000

        while len(allocation) < len(self.words) and safety_counter < max_iterations:
            agent, words = current_state
            remaining_words = set(self.all_words) - set(words) - {Word("no_word")}
            
            print(f"Iteration {safety_counter}:")
            print(f"Current agent: {agent.name}")
            print(f"Remaining words: {[w.text for w in remaining_words]}")
            
            if not remaining_words:
                print("No remaining words. Ending simulation.")
                break

            agent_prob_dict = {item.split(':')[0].lower(): float(item.split(':')[1]) for item in self.word_probs[agent.name]}
            print(f"Agent prob dict: {agent_prob_dict}")
            
            normalized_probs = self._normalize_probs({w.text.lower(): agent_prob_dict.get(w.text.lower(), 0) for w in remaining_words})
            print(f"Normalized probabilities: {normalized_probs}")
            
            action = policy[current_state]
            print(f"Policy suggested action: {action.text}")
            
            if action.text.lower() not in [w.text.lower() for w in remaining_words]:
                action = random.choice(list(remaining_words))
                print(f"Action not in remaining words. Randomly chose: {action.text}")
            
            # Always allocate a word, using weighted random choice if probabilities exist
            if sum(normalized_probs.values()) > 0:
                action_text = random.choices(list(normalized_probs.keys()), weights=list(normalized_probs.values()))[0]
                action = next(word for word in remaining_words if word.text.lower() == action_text)
            else:
                action = random.choice(list(remaining_words))
            
            print(f"Allocated word: {action.text}")
            
            allocation.append((agent.name, action.text))
            next_agent = self.agents[(self.agents.index(agent) + 1) % self.num_agents]
            new_words = frozenset(words | {action})
            current_state = (next_agent, new_words)
            
            print(f"Next agent: {next_agent.name}")
            print(f"Allocations so far: {allocation}")
            print("------------------------")
            
            safety_counter += 1
        
        if safety_counter >= max_iterations:
            print("Warning: Maximum iterations reached in simulate_allocation")
        
        return allocation



class Agent:
    def __init__(self, name, story=None, is_robot=False, in_contribution=None, theme=None):
        self.name = name
        self.contributions = ""
        self.in_contribution = in_contribution
        self.is_robot = is_robot
        self.word_history = []
        self.story = story
        self.theme = theme
        self.current_turn = 1

        # Only for robot agent: Dictionary to store predicted words for the next agent
        if self.is_robot:
            self.next_agent_predicted_words = {}
       
    def predict_next_word_for_agents(self, agent):
        global predicted_words
        selected_words = ', '.join([word.text for word in self.story.agent_selected_words.get(agent.name, [])])
        remaining_word_options = self.story.remaining_words
        options_text = ", ".join([word.text for word in remaining_word_options])
        agent_contributed_story = agent.contributions
        predicted_theme_for_agent = predicted_themes[agent.name]

        if not agent_contributed_story.strip():
            print(f"{agent.name}'s contribution is empty, keeping uniform probabilities.")
            return

        prompt = f"""
        Given the following overall story:
        {self.story}

        The part of the story contributed by {agent.name}:
        {agent_contributed_story}

        The list of words already selected by {agent.name}: {selected_words}
        Predicted theme for {agent.name}: {predicted_theme_for_agent}

        What is probability word chosing probability of {agent.name} under the given theme,
        part of the story contributed by the agent and the list of words already selceted by the 
        aget?
        Please provide a ranking with probabilities in decreasing order of likelihood.
        The format should be strictly 'word:probability' (e.g., 'tree:0.8, river:0.2').

        The words to choose from are: {options_text}
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        predicted_list = response.choices[0].message.content.strip().split(", ")
        prediction_with_probabilities = []
        for word_probability in predicted_list:
            parts = word_probability.split(":")
            if len(parts) == 2:
                clean_word = parts[0].strip()
                # Extract only the numerical part of the probability string
                probability_str = ''.join(char for char in parts[1] if char.isdigit() or char == '.')
                try:
                    probability = float(probability_str)
                    prediction_with_probabilities.append(f"{clean_word}: {probability}")
                except ValueError:
                    print(f"Warning: Could not convert '{probability_str}' to float for word '{clean_word}'")

        # Update the global dictionary
        predicted_words[agent.name] = prediction_with_probabilities

        # Print the updated prediction for the agent
        print(f"Updated prediction for {agent.name}: {prediction_with_probabilities}")


    ### calculate contrbutions of the agent

    def calculate_contribution_for_agents(self, agent):
        global contributions
        agent_contributed_story = agent.contributions
        selected_word_names = {word.text for word in self.story.agent_selected_words.get(agent.name, [])}
        print(f'selected words: {selected_word_names}')
        selected_word_difficulties = {word.text: word.difficulty for word in self.story.agent_selected_words.get(agent.name, [])}

        if not agent_contributed_story.strip():
            # print(f"{agent.name}'s contribution is empty, assigning 'moderate' contribution level.")
            # contributions[agent.name] = "moderate"
            return

        prompt = f"""
        Given the following overall story:
        {self.story}

        The part of the story contributed by {agent.name}:
        {agent_contributed_story}

        The words used by {agent.name} and their difficulties: {selected_word_difficulties}

        What is the most probable contribution level of {agent.name}'s contribution based on coherence, flow, length, engagement, and word difficulties?
        Please provide the answer in the format "agent_name:contribution_level" (e.g., robot1:moderate).

        The contribution levels are: very low, low, moderate, high, very high
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        full_response = response.choices[0].message.content.strip()

        print(f'contribution response:{full_response}')
        
        # Split the response and extract only the contribution level
        try:
            _, contribution_level = full_response.split(':')
            contribution_level = contribution_level.strip()
        except ValueError:
            print(f"Unexpected response format for {agent.name}: {full_response}")
            

        # Update the global dictionary with only the contribution level
        contributions[agent.name] = contribution_level

        # Print the updated contribution for the agent
        print(f"Updated contribution for {agent.name}: {contribution_level}")
   
    ### update probable theme for agents

    def predict_theme_for_agents(self, agent):
        global predicted_themes
        if agent.name == "sam":
            # Do nothing as theme is already saved
            pass
        else:
            # Prepare the word list and contributions for prediction
            selected_words = ', '.join([word.text for word in self.story.agent_selected_words.get(agent.name, [])])
            agent_contributed_story = agent.contributions

            # If no words or contribution, default to unknown theme
            if not selected_words.strip() or not agent_contributed_story.strip():
                predicted_themes[agent.name] = "unknown"
                return

            # Create a prompt for predicting the theme based on agent's words and contribution
            prompt = f"""
            Based on the following selected words and contributions:

            Selected words by {agent.name}: {selected_words}

            Contributions so far by {agent.name}: {agent_contributed_story}

            From the following list of themes, which theme is most closely aligned with the selected words and contributions?

            Themes: {', '.join(theme_list)}. If you think probability of themes list in Themes are very low. Then predict most
            probable theme and respose with that theme. In this case predicted theme can be outside of Themes.

            Please provide the theme name only. Strictly respond with the theme name, for example, "teamwork" or "adventure".
            """

            # Call the GPT model to predict the theme
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract the theme name from the response
            predicted_theme = response.choices[0].message.content.strip()

            # Update the global predicted theme dictionary
            predicted_themes[agent.name] = predicted_theme

            # Print the predicted theme for the agent
            print(f"Predicted theme for {agent.name}: {predicted_theme}")


    def update_belief_and_probabilities(self, agents):
        for agent in agents:
            self.predict_theme_for_agents(agent)
            self.predict_next_word_for_agents(agent)
            self.calculate_contribution_for_agents(agent)

    def get_contributed_story(self):
        return self.contributions

    def add_to_story(self, new_part):
        self.contributions += " " + new_part


    def create_remaining_word_weight_string(self):
        return ",".join([f"{word.text}:{word.difficulty}" for word in self.story.remaining_words])
  

    def get_robot_action(self, agents):
        global predicted_words, contributions

        # is_last_word = False
        # if len(self.story.remaining_words) == 0:
        #     narrative = self.generate_ending_narrative(word=None)
        #     self.story.is_story_ends = True
        #     first_agent_word = None

        if len(self.story.remaining_words) == 1:
            self.update_belief_and_probabilities(agents)
            narrative = self.generate_ending_narrative(word=self.story.remaining_words[0])
            self.story.is_story_ends = True
            first_agent_word = self.story.remaining_words[0]
        else:

            remaining_word_rewards = self.create_remaining_word_weight_string()
            first_agent = next((agent for agent in agents if agent.name == "sam"), None)
            
            first_agent_word, narrative = self.generate_robot_narrative(first_agent, remaining_word_rewards)

            first_agent_word = next(word for word in self.story.remaining_words if word.text == first_agent_word)
            time.sleep(15)
        return first_agent_word, narrative


    @staticmethod
    def contribution_level_to_value(level):

        # print(f'Printing level: {level}')
        return {
            "very low": 0.2,
            "low": 0.4,
            "moderate": 0.6,
            "high": 0.8,
            "very high": 1.0
        }.get(level.strip(), 0.5, "moderate")
    

    def generate_ending_narrative(self, word=None):
        # If there is no word provided for the final part of the story
        if word is None:
            word_text = "END"
        else:
            word_text = word.text
        
        # Get the current state of the story so far
        story_so_far = ' '.join([narrative for _, narrative in self.story.story])

        # Prepare a prompt for GPT-4 to generate a satisfying conclusion
        prompt = f"""
        Given the following story so far:
        "{story_so_far}"

        The last word provided for the story is: {word_text}

        Using this word, please generate a concise and conclusive ending to the story (30-50 words). 
        The ending should wrap up the main themes and plot points, and provide closure. Append "THE END" at the end of the narrative.
        """

        # Call GPT-4 API to generate the story ending
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a creative assistant for generating story endings."},
                {"role": "user", "content": prompt}
            ]
        )

        # Extract the generated ending narrative
        ending = response.choices[0].message.content.strip()

        # Return only the narrative, without any formatting or prefix
        return ending

    def generate_robot_narrative(self, first_agent, remaining_word_rewards):
        global robot_theme
        story_so_far = ' '.join([narrative for _, narrative in self.story.story])
        first_agent_story = first_agent.contributions
        turn = first_agent.current_turn
        
        if turn == 1:
            current_turn = "first"
        elif turn == 2:
            current_turn = "second"
        elif turn == 3:
            current_turn = "third"

        # Choose a word based on rewards and coherence
        word_selection_prompt = f"""
        Context:
        - Story so far: {story_so_far}
        - Your theme: {robot_theme}
        - Your previous contributions: {first_agent_story}
        - Current turn: {current_turn}
        - Available words and their rewards: {remaining_word_rewards}

        Task:
        Choose the most appropriate word from the available words, considering:
        1. The word's reward value
        2. Coherence with the story so far
        3. Relevance to your theme ({robot_theme})
        4. Potential to advance the story based on the current turn ({current_turn})

        Provide your choice in the format: "Chosen word: [word]"
        """

        word_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an AI assistant helping to choose the best word for a storytelling task."},
                {"role": "user", "content": word_selection_prompt}
            ]
        )

        chosen_word = word_response.choices[0].message.content.strip().split(": ")[1]

        # Generate narrative using the chosen word
        narrative_prompt = f"""
        Context:
        - Story so far: {story_so_far}
        - Your theme: {robot_theme}
        - Your chosen word: {chosen_word}
        - Your previous contributions: {first_agent_story}
        - Current turn: {current_turn}

        Task:
        Create a 2-3 line narrative (aproximately 20-30 characters for each line) using {chosen_word}.

        Guidelines:
        1. If the story is empty, start with an introduction.
        2. Follow the 3-act structure:
        - Turn 1: Setup (introduce up to 2 characters)
        - Turn 2: Confrontation (introduce a challenge)
        - Turn 3: Resolution (work towards climax and resolution)
        3. Keep the story simple and easy to follow.
        4. Only include story content; no meta-commentary.

        Generate the narrative:
        """

        narrative_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligent Nao robot participating in a collaborative storytelling task."},
                {"role": "user", "content": narrative_prompt}
            ]
        )

        narrative = narrative_response.choices[0].message.content.strip()

        return chosen_word, narrative
    


class Simulation:
    def __init__(self, story, agents, nlp, client_sockets, robot_client_socket, discount_factor=0.95):
        self.story = story
        self.agents = agents
        self.nlp = nlp
        self.discount_factor = discount_factor
        self.remaining_words = story.words.copy()
        self.client_sockets = client_sockets
        self.robot_client_socket = robot_client_socket

        # Initialize predictions and contributions globally
        self.initialize_global_dictionaries()

    def get_human_action(self, human):
        print(f"\n{human.name}'s turn:")

        # Find the correct client socket for this human agent
        client_socket = next((cs for cs in self.client_sockets if cs.name == human.name), None)
        
        if client_socket is None:
            raise ValueError(f"No client socket found for {human.name}")


        client_socket.send("TELL")

        # Receive action from the client
        message = client_socket.receive()
        print(f'Messge from {human.name}: message')

        try:
            word1, narrative = message.split(':', 1)
            word1 = word1.strip()
            narrative = narrative.strip()
        except ValueError:
            print(f"Invalid message format received from {human.name}")
            return None, None
        

        word = next((word for word in self.story.remaining_words if word.text == word1), None)
        
        return word, narrative
        

    

    def calculate_initial_contribution(self, theme, word, agent):
        global contributions

        if not agent.in_contribution.strip():
            print(f"{agent.name}'s contribution is empty, assigning 'very low' contribution level.")
            contributions[agent.name] = "very low"
            return

        prompt = f"""
        Given the following theme: {theme}
        
        And the word to be used: {word}
        
        The narrative contributed by {agent.name}:
        {agent.in_contribution}

        What is the most probable initial contribution level of {agent.name}'s narrative based on:
        1. Coherence with the theme
        2. Effective use of the given word
        3. Flow of the narrative

        Please provide the answer in the format "agent_name:contribution_level" (e.g., robot1:moderate).

        The contribution levels are: very low, low, moderate, high, very high
        """

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        full_response = response.choices[0].message.content.strip()

        print(f'Initial contribution response for {agent.name}: {full_response}')
        
        # Split the response and extract only the contribution level
        try:
            _, contribution_level = full_response.split(':')
            contribution_level = contribution_level.strip()
        except ValueError:
            print(f"Unexpected response format for {agent.name}: {full_response}")
            contribution_level = "low"  # Default to low if there's an error

        # Update the global dictionary with only the contribution level
        contributions[agent.name] = contribution_level

        # Print the updated contribution for the agent
        print(f"Initial contribution for {agent.name}: {contribution_level}")



    def initialize_global_dictionaries(self):
        global predicted_words, contributions, predicted_themes
        levels = ["very low", "low", "moderate", "high", "very high"]

        for agent in self.agents:
            # Initialize predicted words with uniform probabilities
            predicted_words[agent.name] = [
                f"{word.text}: {round(1 / len(self.remaining_words), 2)}"
                for word in self.remaining_words
            ]


            
            if not agent.is_robot:
                self.calculate_initial_contribution("Journey","Brave",agent)
            else:
                contributions[agent.name] = "very high"
            

            # contributions[agent.name] = "moderate"


    def update_state(self, agent, action):
        word, narrative = action
        self.story.update_turns(agent.name)
        self.story.append_to_story(agent.name, narrative, word)
        self.story.add_selected_word(agent.name, word)
        agent.add_to_story(narrative)
        agent.current_turn = agent.current_turn + 1

        # Find and remove the matching Word object from remaining_words
        for w in self.story.remaining_words:
            if w.text == word.text:
                self.story.remaining_words.remove(w)
                break

        agent.word_history.append(word)

        if (len(self.story.remaining_words) == 0):
            self.story.is_story_ends = True
    
    
    ##### story_game_simulation starts

    def simulate(self, num_iterations=1000):
        global pid_1, pid_2, task_id
        session_string = f'{pid_1}:{pid_2}:{task_id}' 
        is_intro_done = False

        robot_intro = """Hello, I’m Sam, your robot teammate!
        In this collaborative storytelling game, we’ll take turns in a clockwise direction,
        with each player choosing a word from the storyboard to continue our story.
        Our team goal is to collect rewarded words while keeping the story coherent.
        Let’s create something amazing together!. I will go first. Let me think for a while"""

        robot_ending = """Thank you, everyone, for your amazing collaboration!
        Together, we’ve created a wonderful story. 
        I had a great time working with all of you. Until next time, keep telling great stories!"""



        # Start the story and set the start time
        self.story.start_story()


        print("\nStarting Simulation:")
        current_turn = 0

        for iteration in range(num_iterations):
            print(f"\nIteration {iteration + 1}:")

            ## select current agent based on turn
            current_agent = self.agents[current_turn]

            print(f"\n{current_agent.name}'s turn:")

            if current_agent.is_robot:
                 chosen_word, narrative = current_agent.get_robot_action(self.agents)
                 action = chosen_word, narrative
                
                ######## send robot narrative, robot_name, next_payer_name to robot client
                 if not is_intro_done:
                    robot_intro = f'{current_agent.name}:{robot_intro}'
                    robot_intro = unidecode(robot_intro)
                    self.robot_client_socket.send(robot_intro.encode('utf-8'))
                    ack_msg = self.robot_client_socket.recv(1024).decode()
                    print(ack_msg)
                    is_intro_done = True
                
                 time.sleep(5)

                 if len(self.story.remaining_words) == 1:
                     msg = f'{current_agent.name}:{None}:{narrative}'
                 else:
                      msg = f'{current_agent.name}:{self.agents[(current_turn+1)%3].name}:{narrative}'
                 msg = unidecode(msg)
                 self.robot_client_socket.send(msg.encode('utf-8'))
                 ack_msg = self.robot_client_socket.recv(1024).decode()
                 print(ack_msg)


                ###### send robot narrattive to other human clients

                 for agent_socket in self.client_sockets:
                    client_msg = f"STORY:{current_agent.name}:{chosen_word.text}:{narrative}"
                    agent_socket.send(client_msg)
                    ack = agent_socket.receive()
                    print(ack) 

                ###### update state and agent turn
                 self.update_state(current_agent, action)
                 if self.story.is_story_ends:


                    robot_ending = f'{current_agent.name}:{robot_ending}'
                    robot_ending = unidecode(robot_ending)
                    self.robot_client_socket.send(robot_ending.encode('utf-8'))
                    ack_msg = self.robot_client_socket.recv(1024).decode()
                    print(ack_msg)

                    self.story.end_story()
                    self.story.write_story_with_narrators(f'{session_string}_story_with_narrators.json')
                    self.story.write_story_without_narrators(f'{session_string}_story_without_narrators.json')
                    print("\nStory has been saved to f'{session_string}_story_with_narrators.json' and f'{session_string}_story_without_narrators.json'.")
                    break
                 current_turn = (current_turn+1)%3
            
            else:
               
                chosen_word, narrative = self.get_human_action(current_agent)
                action = chosen_word, narrative

                ###### send robot narrattive to other human clients
                for agent_socket in self.client_sockets:
                    client_msg = f"STORY:{current_agent.name}:{chosen_word.text}:{narrative}"
                    agent_socket.send(client_msg)
                    ack = agent_socket.receive()
                    print(ack) 

                self.update_state(current_agent, action)

                if self.story.is_story_ends:

                    robot_ending = f'{current_agent.name}:{robot_ending}'
                    robot_ending = unidecode(robot_ending)
                    self.robot_client_socket.send(robot_ending.encode('utf-8'))
                    ack_msg = self.robot_client_socket.recv(1024).decode()
                    print(ack_msg)

                    self.story.end_story()
                    self.story.write_story_with_narrators(f'{session_string}_story_with_narrators.json', session_string)
                    self.story.write_story_without_narrators(f'{session_string}_story_without_narrators.json', session_string)
                    print("\nStory has been saved to f'{session_string}_story_with_narrators.json' and f'{session_string}_story_without_narrators.json'.")
                    break
                current_turn = (current_turn+1)%3

    
class ClientSocket:
    def __init__(self, socket, name):
        self.socket = socket
        self.name = name

    def send(self, message):
        self.socket.send(message.encode())

    def receive(self):
        return self.socket.recv(4096).decode()

class StorytellingGame:
    def __init__(self):
        global robot_theme, theme_1, theme_2
        print("Initializing Storytelling Game:")
        self.nlp = spacy.load("en_core_web_sm")
        robot_theme = random.choice(theme_list)
        self.story_words = self.initialize_story_words()
        self.story = Story(self.story_words, self.nlp, robot_theme, theme_1, theme_2)  # Pass themes to Story
        self.client_sockets = []
        self.agents = []
        self.initialize_agent_and_sockets()
        self.story.agents = self.agents
       
        print(f"Agents: {', '.join([agent.name for agent in self.agents])}")
        
        self.simulation = Simulation(self.story, self.agents, self.nlp, self.client_sockets, self.robot_client_socket)

    def create_word_weight_string(self):
        return ",".join([f"{word.text}:{word.difficulty}" for word in self.story_words])
    
    def initialize_agent_and_sockets(self):
        global theme_list, pid_1, pid_2, task_id
        task_id = "baseline"
        print("Waiting for robot client")

        server = ServerSocket(server_id, server_port)
        self.robot_client_socket = server.accept_connection()
        human_agents = []
        # word_weight_string = self.create_word_weight_string()
        print("Waiting for first client connection...")
        client_socket1 = server.accept_connection()
        data = client_socket1.recv(1024).decode().strip()
        name, pid, theme, in_narrative = data.split(':', 3)
        pid_1 = pid
        print(f'participant_1: {name}:{pid}:{theme}')
        print(f'word_weight_string: {self.create_word_weight_string()}')

        

        client_socket1.send(self.create_word_weight_string().encode())
        self.client_sockets.append(ClientSocket(client_socket1, name))
        human_agents.append(Agent(name,in_contribution=in_narrative,theme=theme ))  # Story will be set later

        print("Waiting for second client connection...")
        client_socket2 = server.accept_connection()
        data = client_socket2.recv(1024).decode().strip()
        name, pid,  theme, in_narrative = data.split(':', 3)
        pid_2 = pid
        print(f'participant_2: {name}:{pid}:{theme}')
        
        client_socket2.send(self.create_word_weight_string().encode())
        self.client_sockets.append(ClientSocket(client_socket2, name))
        human_agents.append(Agent(name, in_contribution=in_narrative,theme=theme ))  # Story will be set later

        # Create robot agents
        robot_agents = [
            Agent("sam", is_robot=True)
        ]

        predicted_themes[robot_agents[0].name] = robot_theme

        print(f'Selected theme for {robot_agents[0].name}: {predicted_themes[robot_agents[0].name]} ')
        
        # Combine all agents
                # Set the story for all agents
        all_agents = robot_agents + human_agents
        for agent in all_agents:
            agent.story = self.story
            self.agents.append(agent)
    
    def create_combined_word_list(self, theme_dict):
        """
        Creates a list of 9 unique words by uniting all words from the two human-selected themes (theme_1 and theme_2)
        and the robot's theme into a set of unique words, shuffling the set, and then randomly sampling 9 words from it.
        
        Args:
        - theme_dict (dict): Dictionary containing themes and their associated words.
        
        Returns:
        - list: A shuffled list of 9 unique words, 3 from each of the three themes (theme_1, theme_2, and robot_theme).
        """
        global theme_1, theme_2, robot_theme

        # Combine all words from the three themes into a set (to ensure uniqueness)
        combined_set = set(theme_dict[theme_1]) | set(theme_dict[theme_2]) | set(theme_dict[robot_theme])

        # Shuffle the combined set and convert to list
        combined_words_list = list(combined_set)
        random.shuffle(combined_words_list)

        # Sample 9 words from the shuffled combined list
        sampled_words = random.sample(combined_words_list, min(9, len(combined_words_list)))

        return sampled_words



    def load_story_theme_words(self, folder_path):
        """
        Reads a folder containing text files, each representing a story theme.
        Creates a dictionary where the key is the theme name (text file name without extension)
        and the value is a list of words for that theme.
        
        Args:
        - folder_path (str): Path to the folder containing theme text files.
        
        Returns:
        - dict: A dictionary of story theme words.
        """
        story_theme_dict = {}

        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                # Get the theme name (file name without extension)
                theme_name = os.path.splitext(file_name)[0]
                
                # Read the words from the file, line by line
                with open(os.path.join(folder_path, file_name), 'r') as file:
                    words = [line.strip() for line in file if line.strip()]
                
                # Add the theme and its words to the dictionary
                story_theme_dict[theme_name] = words

        # Print the dictionary after loading
        # print(story_theme_dict)

        return story_theme_dict  

    def initialize_story_words(self):

        folder_path = "story_themes"

        # Load the story theme words from the folder
        story_theme_dict = self.load_story_theme_words(folder_path)

        # # Example of using the loaded dictionary (printing the themes and words)
        # for theme, words in story_theme_dict.items():
        #     print(f"Theme: {theme}, Words: {words}")

        word_texts = self.create_combined_word_list(story_theme_dict)
        return [Word(text, self.nlp) for text in word_texts]


    def play_game(self):
        print("\nStarting the game:")
        story = self.simulation.simulate()
        print("The Story Ends here")
        
class ServerSocket:
    def __init__(self, host, port):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((host, port))
        self.server_socket.listen(3)

    def accept_connection(self):
        client_socket, addr = self.server_socket.accept()
        print(f"Connection from: {addr}")
        return client_socket


if __name__ == "__main__":

    game = StorytellingGame()
    game.play_game()
