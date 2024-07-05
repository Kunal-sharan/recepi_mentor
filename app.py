import streamlit as st
import json
from typing import List
import random
import time
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.memory import ConversationBufferWindowMemory #ConversationBufferMemory
from langchain.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
import spacy
from bs4 import BeautifulSoup
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
from typing import Literal
import streamlit as st
import json
from langchain_groq import ChatGroq
from langchain.callbacks import get_openai_callback
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import streamlit.components.v1 as components
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("Recipe Mentor")
st.write("Please read the instructions before starting")
with st.expander("Instructions to user"):
   st.markdown("""
    # Features
    1. **Recipes Generation Based on Personal Preference**: Get customized recipes tailored to your dietary preferences and available ingredients with their nutritional values.
    2. **Chat Support**: Access chat support if you encounter any problems with the instructions or any part of the recipe.
    3. **Video Demonstration**: Access a section for video demonstrations if you need further help with the recipes.

    # Recipe Helper Instruction Manual
    ### Step-by-Step Guide
    1. **Fill User Preferences**: Begin by filling in your dietary preferences, allergies, favorite cuisines, taste preferences, meal type, number of people, cooking time, and available ingredients.
       - After filling in all the necessary information, click on the **Submit** button. Wait until the running is complete; only click on the button, do not press Enter from the keyboard.
    2. **Wait for Processing**: Please wait for 3-5 seconds while the system processes your input. You will see a running indicator during this time.
    3. **Click on Generate**: Once the processing stops, click on the **Generate** button to proceed. Wait until the running is complete; only click on the button, do not press Enter from the keyboard.
    4. **Chatbot Assistance**: You will now have access to a chatbot specifically designed to assist you with recipes.
      - plz click on submit button in chatbot
    5. **Video Demonstration**: To get video demonstrations of the recipes, click on the **Generate Videos** button. Wait until the running is complete.

    ### Notes
    - Ensure you have a stable internet connection for the best experience.
    - If you encounter any issues, please refresh the page and try again, following the steps in the same chronological order.
    - only click on the **buttons** for any action, do not press **Enter** from the keyboard.
    - Don't click anything when the process is running.
    - Dealing with chatbot please wait for it to reply may face some latency issues
    """)


nlp = spacy.load("en_core_web_sm")
@st.cache_resource
def get_modules():
  model = SentenceTransformer('all-MiniLM-L6-v2')
  return model
# nlp = spacy.load("en_core_web_sm")
# model = SentenceTransformer('all-MiniLM-L6-v2')

@st.experimental_fragment
def video_recomendation(dish_data):
  def get_similarity_scores(dish_data):
      def get_video_ids(user_query):
          headers = {"User-Agent": "Guest"}
          video_res = requests.get(f'https://www.youtube.com/results?search_query={"+".join(user_query.split(" "))}', headers=headers)
          soup = BeautifulSoup(video_res.text, 'html.parser')
          arr_video = soup.find_all('script')

          arr_main = []
          for i in arr_video:
              if 'var ytInitialData' in str(i.get_text()):
                  arr_main.append(i)
                  break
          main_script = arr_main[0].get_text()[arr_main[0].get_text().find('{'):arr_main[0].get_text().rfind('}')+1]
          data = json.loads(main_script)
          video_data = data.get('contents').get('twoColumnSearchResultsRenderer').get('primaryContents').get('sectionListRenderer').get('contents')[0].get('itemSectionRenderer').get('contents')
          video_json = [i for i in video_data if 'videoRenderer' in str(i)]

          video_ids = [i.get('videoRenderer').get('videoId') for i in video_json if i.get('videoRenderer')]
          return video_ids

      def get_transcript(video_ids):
          yt_data = []
          for i in video_ids:
              txt = ""
              try:
                  transcript = YouTubeTranscriptApi.get_transcript(i, languages=['en'])
                  for j in transcript:
                      txt += j['text'] + " "
                  yt_data.append({"video_id": i, "transcript": txt})
              except:
                  continue
          return yt_data

      def compare_transcript(arr):
          clean_arr = []
          for i in arr:
              transcript = i['transcript']
              doc = nlp(transcript)
              filtered_words = [token.text for token in doc if not token.is_stop]
              clean_transcript = ' '.join(filtered_words)
              clean_arr.append({"video_id": i['video_id'], "transcript": clean_transcript})
          return clean_arr

      all_scores = {}

      for dish in dish_data['dishes']:
          recepi_name = dish['recipe_name']
          recepi_instructions = dish['recipe_instructions']

          # Extract steps from recipe
          steps = " ".join([instruction["description"] for instruction in recepi_instructions if instruction["description"]])

          # Fetch video IDs
          video_ids = get_video_ids(recepi_name)

          # Fetch transcripts
          yt_transcript_data = get_transcript(video_ids)

          # Clean text from recipe steps
          doc = nlp(steps)
          filtered_words = [token.text for token in doc if not token.is_stop]
          clean_text = ' '.join(filtered_words)

          # Compare transcript and calculate similarity scores
          model=get_modules()
          score_arr = []
          test_vec = model.encode([clean_text])[0]

          for sent in compare_transcript(yt_transcript_data):
              transcript = sent['transcript']
              transcript_vec = model.encode([transcript])[0]
              similarity_score = cosine_similarity([test_vec], [transcript_vec])[0][0]
              score_arr.append({"id": sent['video_id'], "score": similarity_score})

          # Store similarity scores for the current dish
          all_scores[recepi_name] = score_arr

      return all_scores
  def get_top_keys(data, top_n=2):
    top_results = {}

    for dish, videos in data.items():
        # Sort the list of dictionaries by the score in descending order
        sorted_videos = sorted(videos, key=lambda x: float(x['score']), reverse=True)

        # Get the top `top_n` video IDs
        top_videos = sorted_videos[:top_n]

        # Store the top video IDs in the result dictionary
        top_results[dish] = top_videos

    return top_results

  generate_video=st.button("See video demonstration")
  if generate_video:

    with st.spinner("....."):
      scores = get_similarity_scores(dish_data)
      # top=get_top_keys(scores)
      # st.write(scores)
      v_d=get_top_keys(scores)
      for i in v_d.keys():
        with st.expander(f'{i}'):
          for k in v_d[f"{i}"]:
            with st.container(border=True):
              st.video(f"https://www.youtube.com/watch?v={k['id']}")
              st.write(f"{int(float(k['score'])*100.0)}% similar to the recepi of {i} generated by bot")


class Ingredient(BaseModel):
  name: str =Field(description="name of the ingredient")
  quantity: str = Field(description="quantity of ingredient")
  alternatives: List[str] = Field(description="List of all the alternative ingredients that can be used instead of this")
  nutritional_value: str = Field(description="Nutritional value of the ingredient")

class Instructions(BaseModel):
  step: int = Field(description="step number")
  description: str = Field(description="description of the step")

class Recepi(BaseModel):
  recipe_name: str = Field(description="name of the dish")
  recipe_ingredients: List[Ingredient] = Field(description="List all the ingredients for the dish")
  recipe_instructions: List[Instructions] = Field(descriptions="List of all the instructions to make the dish")
  cal: str = Field(description="Calories of the dish")
  carbs: str = Field(description="Carbohydrates of the dish")
  fat: str = Field(description="Fat of the dish")
  protein: str = Field(description="Protein of the dish")

class Dishes(BaseModel):
  dishes: List[Recepi] = Field(description="List of 2 dishes based on the user_data")


parser= JsonOutputParser(pydantic_object=Dishes)

prompt = PromptTemplate(
    template="Provide 2 recepi for the user based on the that is delimited by single quotes \n user_data:'{user_data}'.\n{format_instructions}",
    input_variables=["user_data"],
    partial_variables={"format_instructions": parser.get_format_instructions()},

)

chat = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key="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"  # Ensure to provide your API key here if not set as an environment variable
        )

chain_new=prompt | chat | parser


def fetch_recipe(user_data):

    st.session_state.recipes = chain_new.invoke({"user_data":f"{user_data}"})

    return st.session_state.recipes if "recipes" in st.session_state else None

col1,col2=st.columns([1,1])
user_data=None
recipes=None
with col1:
  st.write("### Generates 1-2 recipies based on the user prefrences")
  st.warning("Click Generate only after submiting your preferences")
  generator=st.button("Generate")

  if generator:
    if "history" in st.session_state and "conversation" in st.session_state:
      del st.session_state.history
      del st.session_state.conversation
    if 'user_data' in st.session_state:
      user_data=st.session_state.user_data
      recipes = fetch_recipe(user_data)
      exp_user=st.expander("# User Dietary Information")
      exp_user.markdown(f"""

**Dietary Restrictions:** {user_data["dietary_restrictions"]}

**Allergies:** {user_data["allergies"]}

**Favorite Cuisines:** {user_data["favorite_cuisines"]}

**Taste Preference:** {user_data["taste_preference"]}

**Meal Type:** {user_data["meal_type"]}

**Number of People:** {user_data["number_of_people"]}

**Cooking Time:** {user_data["cooking_time"]}

**Available Ingredients:** {user_data["available_ingredients"]}

**Prefers Low Sugar Food:** {'Yes' if user_data["prefers_low_sugar_food"] else 'No'}

**Has High Blood Pressure:** {'Yes' if user_data["has_high_blood_pressure"] else 'No'}

**Other Dietary Preferences:** {user_data["other_dietary_preferences"]}
""")
      # st.write(recipes)
      if recipes:
        expander = st.expander("See explanation")
        i=0
        # st.write(recipes)
        for recipe in recipes["dishes"]:
          with st.expander(f'{recipe["recipe_name"]}'):
              st.markdown(f"## {recipe['recipe_name']}")
              st.markdown("### Ingredients")
              for ingredient in recipe["recipe_ingredients"]:
                  st.markdown(f"- **{ingredient['name']}**: {ingredient['quantity']} (Nutritional Value: {ingredient['nutritional_value']}) (Alternatives: {', '.join(ingredient['alternatives'])})")
              st.markdown("### Instructions")
              for instruction in recipe["recipe_instructions"]:
                  st.markdown(f"{instruction['step']}. {instruction['description']}")
                  # st.markdown("---")
              st.markdown("---")
              st.markdown(f"**Calories:** {recipe['cal']}")
              st.markdown(f"**Carbohydrates:** {recipe['carbs']}")
              st.markdown(f"**Fat:** {recipe['fat']}")
              st.markdown(f"**Protein:** {recipe['protein']}")
              st.markdown("---")
              i += 1
        video_recomendation(recipes)



# Define initial values for the user data
dietary_restrictions = ""
allergies = ""
favorite_cuisines = ""
taste_preference = ""
meal_type = ""
number_of_people = 1
cooking_methods = ""
cooking_time = 0
user_name = ""
user_city = ""
user_state = ""
user_country = ""
spicyness = False
sugar = False
bp = False
other = ""

# Define the form inside the sidebar
with st.sidebar:
    st.title("User Preferences")
    with st.form("preferences_form"):
        # Dietary Restrictions and Allergies
        dietary_restrictions = st.text_input("Do you have any dietary restrictions?",value="no")
        allergies = st.text_input("Are you allergic to any foods?",value="no")

        # Preferences
        favorite_cuisines = st.text_input("What are your favorite cuisines?",value="all")
        taste_preference = st.selectbox("Do you prefer spicy, sweet, savory, or mild foods?", ["Spicy", "Sweet", "Savory", "Mild"])

        # Meal Type and Occasion
        meal_type = st.selectbox("What type of meal are you looking for?", ["Breakfast", "Lunch", "Dinner", "Snack"])
        available_ingredients = st.text_area("Are there any ingredients you have on hand or would like to use?")
        number_of_people = st.number_input("How many people will be eating?", min_value=1, step=1)

        # Cooking Methods and Equipment
        # cooking_methods = st.text_input("Do you have any preferred cooking methods?")
        cooking_time = st.number_input("How much time do you have for cooking? (in minutes)", min_value=0, step=1)

        sugar = st.checkbox("Do you prefer low sugar food?", value=False)
        bp = st.checkbox("Do you have high blood pressure?", value=False)
        other = st.text_input("Other dietary preferences",value="no")

        # Submit button
        submitted = st.form_submit_button("Submit")

        if submitted and available_ingredients:
          user_data = {
        "dietary_restrictions": dietary_restrictions,
        "allergies": allergies,
        "favorite_cuisines": favorite_cuisines,
        "taste_preference": taste_preference,
        "meal_type": meal_type,
        "number_of_people": number_of_people,
        "cooking_time": cooking_time,
        "available_ingredients":available_ingredients,
        "prefers_low_sugar_food": sugar,
        "has_high_blood_pressure": bp,
        "other_dietary_preferences": other
    }
          st.session_state.user_data=user_data
          st.success("Preference saved")

# Retrieve and display user data from session state if it exists

# col1,col2=st.columns([1,1])

@st.experimental_fragment
def bot(user_data,recipes):
  @dataclass
  class Message:
    """Class for keeping track of a chat message."""
    origin: Literal["human", "ai"]
    message: str

  def load_css():
    # with open("styles.css", "r") as f:
          css = '''<style>
          .chat-row {
                display: flex;
                margin: 5px;
                width: 100%;
            }

            .row-reverse {
                flex-direction: row-reverse;
            }

            .chat-bubble {
                font-family: "Source Sans Pro", sans-serif, "Segoe UI", "Roboto", sans-serif;
                border: 1px solid transparent;
                padding: 5px 10px;
                margin: 0px 7px;
                max-width: 70%;
            }

            .ai-bubble {
                background: #83D475;
                border-radius: 10px;
                color:white
            }

            .human-bubble {
                background: linear-gradient(135deg, rgb(0, 178, 255) 0%, rgb(0, 106, 255) 100%);
                color: white;
                border-radius: 20px;
            }

            .chat-icon {
                border-radius: 5px;
            }
          </style>'''
          st.markdown(css, unsafe_allow_html=True)
  memory = ConversationBufferWindowMemory(k=2)# This creates a memory that stores two latest converstions as k=2
  template = """Act like a chef and You have these knowlege about user user_data: '{user_data}' about these dishes the user is making i.e dishes: {recipes} have a friendly conversation
  based on the knowlege you have and also act like dietician and provide user suggestions to promote healthy and diverse eating habits based on user_data and dishes he is eating if asked
  Current conversation:
  {history}
  Human: {input}
  AI:
  """
  prompt_template = PromptTemplate(input_variables=["history", "input", "user_data","recipes"], template=template)
  llm = ChatGroq(
            temperature=0,
            model="llama3-70b-8192",
            api_key="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"  # Ensure to provide your API key here if not set as an environment variable
        )


  def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    # if "token_count" not in st.session_state:
    #     st.session_state.token_count = 0
    if "conversation" not in st.session_state:

        st.session_state.conversation = ConversationChain(
      llm=llm,
      prompt=prompt_template.partial(user_data=user_data,recipes=recipes),
      verbose=False,
      memory=memory
  )

  def on_click_callback():
          human_prompt = st.session_state.human_prompt
          llm_response = st.session_state.conversation.run(
              human_prompt
          )
          st.session_state.history.append(
              Message("human", human_prompt)
          )
          st.session_state.history.append(
              Message("ai", llm_response)
          )
          # st.session_state.token_count += cb.total_tokens


  load_css()
  initialize_session_state()

  def reset():
    if "history" in st.session_state:
      st.session_state.history=[]

  with st.container(border=True):
    st.title("Chat with the recipes generated 🤖")
    chat_placeholder = st.container(border=True,height=500)
    prompt_placeholder = st.form("chat-form")

    with chat_placeholder:
        for chat in st.session_state.history:
            div = f"""
    <div class="chat-row
        {'' if chat.origin == 'ai' else 'row-reverse'}">
        <div class="chat-bubble
        {'ai-bubble' if chat.origin == 'ai' else 'human-bubble'}">
            &#8203;{chat.message}
        </div>
    </div>
            """
            st.markdown(div, unsafe_allow_html=True)

        for _ in range(3):
            st.markdown("")

        with prompt_placeholder:
          st.markdown("**Chat**")
          cols = st.columns((5, 1,1))
          cols[0].text_input(
              "Chat",
              value="tell me about the recipies",
              label_visibility="collapsed",
              key="human_prompt",
          )
          cols[1].form_submit_button(
              "Submit",
              type="primary",
              on_click=on_click_callback,
          )
          cols[2].form_submit_button("reset",on_click=reset,type="secondary")


    components.html("""
    <script>
    const streamlitDoc = window.parent.document;

    const buttons = Array.from(
        streamlitDoc.querySelectorAll('.stButton > button')
    );
    const submitButton = buttons.find(
        el => el.innerText === 'Submit'
    );

    streamlitDoc.addEventListener('keydown', function(e) {
        switch (e.key) {
            case 'Enter':
                submitButton.click();
                break;
        }
    });
    </script>
    """,
        height=0,
        width=0,
    )

with col2:
  if user_data and recipes:
    # st.write(recipes)
    bot(user_data,recipes)
