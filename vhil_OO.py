
import pyautogui
import pygetwindow
import time
import pywinctl as pwc
from PIL import ImageGrab

import re
import os
from together import Together
from meta_ai_api import MetaAI
import sys
import sounddevice as sd
# from TTS.api import TTS
import time
import whisper
import subprocess
import threading
import sounddevice as sd
import wave
import numpy as np
from pynput.keyboard import Key, Controller
import concurrent.futures
import requests
import abc

import logging
import functools

import csv
import sqlite3
from datetime import date

def assert_type(obj, expected_type):
    assert isinstance(obj, expected_type), f"Expected {expected_type}, but got {type(obj)}"
# ---------- KEYBOARD UTILS ---------
def input_keys(keys,duration=0.000001, delay=0.1):
  print("** Exec Input keys: " + keys.lower())
  pynput_keyboard = Controller()
  numb_inputs = 0
  for letter in keys.lower():
      # pyautogui.press(letter.lower())
      # time.sleep(1)
      pynput_keyboard.type(letter,duration=0,delay=delay)
      numb_inputs = numb_inputs + 1
    #   time.sleep(1.5)
      time.sleep(0.4)
      if numb_inputs == 2:
          break

def refresh_page():
#   pyautogui.hotkey('command', 'r')
  print("\n**REFRESHING PAGE")
  pyautogui.press('esc')
  pynput_keyboard = Controller()
  with pynput_keyboard.pressed(Key.cmd):
        pynput_keyboard.press('r')
        pynput_keyboard.release('r')
  time.sleep(4.4)
  pyautogui.press('esc')

def scroll_full_page_down():
  VoiceAssistant.voice("I'm scrolling")
  input_keys("dd")

def scroll_half_page_down():
  VoiceAssistant.voice("I'm scrolling")
  input_keys("d")

def scroll_to_top_of_page():
  time.sleep(0.5)
  VoiceAssistant.voice("I'm scrolling")
  input_keys("gg")

def clear_prefilled_txt_box():
  pyautogui.press('esc') # use input keys instead when refactoring
  input_keys("gi")
  time.sleep(0.3)
  pyautogui.hotkey('command', 'a')
  # Delete selected text
  time.sleep(0.3)
  pyautogui.press('delete')
  # exit out of the input txtbox  
  pyautogui.press('esc') # use input keys instead when refactoring

def fill_in_txtbox(input_text,duration=0.02,delay=0.02, clear_txtbox_first=True):
  # focus on txtbox
  pyautogui.press('esc') # use input keys instead when refactoring
  time.sleep(0.2)
  input_keys("gi")
  time.sleep(0.3)
  pynput_keyboard = Controller()
  # type actual text, need to do in this way cuz pyautogui is dumb sometimes and pynput cannot type "." for some reason
  for char in input_text:
    #  if char == ".":
    # All non-letter characters (including punctuation, digits and whitespace) will be typed using pyautogui.
    if not char.isalpha(): 
        pyautogui.typewrite(char)
    else:
        pynput_keyboard.type(char,duration,delay)   
  # exit out of the input txtbox  
  pyautogui.press('esc') # use input keys instead when refactoring

  # sentences = new_msg.split(".")
  # for i, sentence in enumerate(sentences):
  #       # for char in sentence:
  #       #     pynput_keyboard.press(char).release()
  #       pynput_keyboard.type(sentence,0.03,0.03)
        
  #       if i < len(sentences) - 1:  # Don't press '.' after last sentence
  #           pyautogui.typewrite(".")
  # pynput_keyboard.type(input,0.03,0.03)

# ---------- WINDOW UTILS ---------
def get_matching_windows(window_name,only_one_matching_window=False):
    while True:#attempts < max_attempts:
        windows = pwc.getWindowsWithTitle(
            title=window_name, 
            condition=pwc.Re.CONTAINS, 
            flags=pwc.Re.IGNORECASE
        )
        print(windows)
        if windows is not None: 
            break
    
    if(only_one_matching_window and len(windows) > 1):
        raise Exception("FOUND MORE THAN ONE MATCHING WINDOW, please be more specific")
    return windows
# PURE FUNC
def activate_window(window_name):
    print(f"** Activating {window_name} Window")
    # pyautogui.sleep(1)  # Wait 1000ms
    # max_attempts = 30000
    # attempts = 0
    # windows=None

    # # while True:#attempts < max_attempts:
    #     windows = pwc.getWindowsWithTitle(
    #         title=window_name, 
    #         condition=pwc.Re.CONTAINS, 
    #         flags=pwc.Re.IGNORECASE
    #     )
    #     print(windows)
    #     if windows is not None: 
    #         break
    
    # if(len(windows) > 1):
    #     raise Exception("FOUND MORE THAN ONE MATCHING WINDOW, please be more specific")

    windows = get_matching_windows(window_name, only_one_matching_window=True)

    if windows:
        windows[0].activate()
        print(f"{window_name} window activated")
        pyautogui.sleep(0.1)  # Wait 100ms
        pyautogui.click(
            windows[0].left + 80, 
            windows[0].top + 40
        )
        print("Clicked on " + window_name + " window")
        return windows # ret matching windows
    else:
            print(f"Failed to activate. Can't find {window_name} window.")
            #print(f"No {window_name} window found")

class ScreenshotManager:
    def __init__(self, entire_window=False, custom_use_defaults=False, screenshot_filename="screenshot.png", window_name="LinkedIn"):
        self.entire_window = entire_window
        self.custom_use_defaults = custom_use_defaults
        self.x1, self.y1 = 1527,200 #231 #457, 237
        self.x2, self.y2 = 2508,1100 # 1421, 827
        self.width, self.height = self.x2 - self.x1, self.y2 - self.y1
        self.screenshot_filename = screenshot_filename
        self.window_name = window_name
        self._windows = activate_window(self.window_name)
        self.running = True
        self.coords_lock = threading.Lock()
        
        # self.root= tk.Tk()
    @property
    def windows(self):
        return self._windows
    def config_ss_params(self):
        # if self.entire_window:
        #     # windows = activate_window(self.window_name)
        #     # self.windows = windows
        #     # self.x1, self.y1 = self.windows[0].left, self.windows[0].top
        #     # self.x2, self.y2 = self.windows[0].right, self.windows[0].bottom
        #     # self.width, self.height = self.windows[0].width, self.windows[0].height
        #     self.update_window_coords()
        #     self.start_coord_updates()
        # elif not self.custom_use_defaults:
        # manually obtain coords
        if not self.custom_use_defaults and not self.entire_window:
            print("** Configuring Screenshot params...")
            for i in range(5, 0, -1):
                print(i)
                time.sleep(1)
            self.x1, self.y1 = pyautogui.position()
            print(f"1st Mouse position: ({self.x1}, {self.y1})")
            for i in range(5, 0, -1):
                print(i)
                time.sleep(1)
            self.x2, self.y2 = pyautogui.position()
            print(f"2nd Mouse position: ({self.x2}, {self.y2})")
            self.width, self.height = self.x2 - self.x1, self.y2 - self.y1

            image = pyautogui.screenshot(region=(self.x1, self.y1, self.width, self.height))
            image.save(self.screenshot_filename)
            print("Took test screenshot with new params @ screenshot.png")
        elif self.entire_window:
            print("Using entire window for screenshots")
        else:
            print("Using custom default coordinates for screenshots")
            return
        
    def update_window_coords(self):
        try:
            with self.coords_lock: 
                if self.windows is not None:
                    new_x1, new_y1 = self.windows[0].left, self.windows[0].top
                    new_x2, new_y2 = self.windows[0].right, self.windows[0].bottom
                    # new_width, new_height = self.windows[0].width, self.windows[0].height
                    new_width = self.windows[0].right - self.windows[0].left
                    new_height = self.windows[0].bottom - self.windows[0].top
                    print("SELF.WINDOWS:",self.windows)
                    print("new_x1:", new_x1)
                    print("new_y1:", new_y1)
                    print("new_x2:", new_x2)
                    print("new_y2:", new_y2)
                    print("new_width:", new_width)
                    print("new_height:", new_height)
                                        
                    if(new_x1 == new_y1 == new_x2 == new_y2 == 0):
                        print("\nUH-OH...0 Coords found \n")
                        return
                    return
                    if (new_x1, new_y1, new_x2, new_y2, new_width, new_height) != (self.x1, self.y1, self.x2, self.y2, self.width, self.height):
                        self.x1, self.y1 = new_x1, new_y1
                        self.x2, self.y2 = new_x2, new_y2
                        self.width, self.height = new_width, new_height
                        print("Updated window coordinates")
        except Exception as e:
            print(f"An error occurred in updating window coords: {str(e)}")
        # self.root.after(500, self.update_window_coords)
            

    def start_coord_updates(self):
        threading.Thread(target=self.run_updates).start()

    def run_updates(self):
        while self.running:
            self.update_window_coords()
            time.sleep(1)  # 200ms

    def stop_updates(self):
        self.running = False

    def take_ss(self):
        with self.coords_lock: 
            # print("in TAKE SS func")
            # image = pyautogui.screenshot(region=(self.x1, self.y1, self.width, self.height))
            if self.entire_window:
                while True:
                    self.x1, self.y1 = self.windows[0].left, self.windows[0].top
                    self.width, self.height = self.windows[0].width, self.windows[0].height
                    if self.width > 0 and self.height > 0:
                        break
                    print(f"Current Screenshot Coordinates (will retry): {self.x1, self.y1, self.width, self.height}") 
                    time.sleep(2)
                    self._windows = get_matching_windows(self.window_name, only_one_matching_window=True)

            print(f"Screenshot Coordinates: {self.x1, self.y1, self.width, self.height}")
            image = ImageGrab.grab(bbox=(self.x1, self.y1, self.x1 + self.width, self.y1 + self.height))
            image.save(self.screenshot_filename)
            # print("hit keys & took ss")
        return self.upload_ss()

    def upload_ss(self):
        with open(self.screenshot_filename, 'rb') as f:
            data = f.read()
        response = requests.put('http://bashupload.com/screenshot.png', data=data, timeout=4)
        # print("printing response")
        # print(response.content)
        # time.sleep(0.5)
        return self.extract_link(response)

    def extract_link(self, response):
        match = re.search(r'wget\s+(http[s]?://\S+)', response.text)
        if match:
            matched_link = match.group(1)
            # print(matched_link)
            return matched_link
        else:
            print("No link found")
            return None 
class MicInputRecorder:
    def __init__(self, whisper_model_name='tiny',duration=3, sample_rate=44100, channels=1, input_audio_file_name="responding_to_vision.wav"):
        """
        Initialize MicInputRecorder.

        Args:
            whisper_model_name (str): Whisper model name. Defaults to 'tiny'.
            duration (int): Recording duration in seconds. Defaults to 3.
        """
        self.whisper_model_name = whisper_model_name
        self.model = whisper.load_model(whisper_model_name)
        self.duration = duration
        self.sample_rate = sample_rate  # Hz
        self.channels = channels  # channel=1 is Mono
        self.input_audio_file_name = input_audio_file_name

    def record_audio(self):
        """
        Record audio from microphone.

        Returns:
            np.ndarray: Recorded audio.
        """
        with VoiceAssistant.voice_lock:
            print("Recording...")
            audio = sd.rec(int(self.duration * self.sample_rate), 
                        samplerate=self.sample_rate, 
                        channels=self.channels)
            sd.wait()  # Wait for recording to finish
            print("Recording finished.")
        return audio

    def save_audio_to_wav(self, audio):
        """
        Save recorded audio to WAV file.

        Args:
            audio (np.ndarray): Recorded audio.
        """
        # Ensure highest value is in 16-bit range
        audio *= 32767 / np.max(np.abs(audio))
        audio = audio.astype(np.int16)

        wave_file = wave.open(self.input_audio_file_name, "wb")
        wave_file.setnchannels(self.channels)
        wave_file.setsampwidth(2)  # 16-bit audio
        wave_file.setframerate(self.sample_rate)
        wave_file.writeframes(audio.tobytes())
        wave_file.close()

    def transcribe_audio(self):
        """
        Transcribe recorded audio using Whisper.

        Returns:
            str: Transcribed text.
        """
       
        result = self.model.transcribe(self.input_audio_file_name)
        print("Transcription from mic: ")
        print(result["text"])
        print("Transcription done.")
        return result["text"]

    def get_mic_input(self):
        """
        Record and transcribe microphone input.

        Returns:
            str: Transcribed text.
        """
        audio = self.record_audio()
        self.save_audio_to_wav(audio)
        return self.transcribe_audio()
class VoiceAssistant:
    voice_lock = threading.Lock()
    enabled = True
    @classmethod
    def enable_voice(cls):
        cls.enabled = True

    @classmethod
    def disable_voice(cls):
        cls.enabled = False

    @staticmethod
    # block is useless cuz we now have locks
    def voice(input, block=False):
        """
        Play a VoiceAssistant.voice message.

        Args:
        input (str): The text to be spoken.
        block (bool, optional): Whether to block until the VoiceAssistant.voice message is finished. Defaults to False.
        """
        if VoiceAssistant.enabled:
            thread = threading.Thread(target=VoiceAssistant._voice_thread, args=(input, block))
            thread.start()
        # if block:
        #     thread.join()  # Wait for the thread to finish

    @staticmethod
    def _voice_thread(input, block=False):
        VoiceAssistant.voice_lock.acquire()
        try:
            subprocess.call(["say", input])
        finally:
            VoiceAssistant.voice_lock.release()
# ---------- LLM UTILS ---------
class LLMManager(abc.ABC):
    def __init__(self, new_llm_every_req=True):
        self._total_numb_requests_served = 0
        self._lock = threading.Lock()
        # self.MAX_REQS_B4_TIMEOUT = 50
        self.new_llm_every_req = new_llm_every_req
        self._internalLLM = None if new_llm_every_req else self._init_internal_llm()

    @property
    def internalLLM(self):
          """
          Gets the internal LLM instance.
          """
          return self._internalLLM
    
    @internalLLM.setter
    def internalLLM(self, value):
        """
        Sets the internal LLM instance.
        """
        self._internalLLM = value

    @abc.abstractmethod
    def _init_internal_llm(self):
        pass
    @abc.abstractmethod
    def prompt_llm(self,prompt_str):
        pass

    def _increment_total_numb_requests_served(self):
        with self._lock:
            self._total_numb_requests_served += 1

    @property
    def total_numb_requests_served(self):
        with self._lock:
            return self._total_numb_requests_served
    @total_numb_requests_served.setter
    def total_numb_requests_served(self, value):
        """
        Sets the internal LLM instance.
        """
        self._total_numb_requests_served = value
class MetaManager(LLMManager):
    """
    A manager class for handling MetaAI() requests.
    """

    MAX_REQS_B4_TIMEOUT = 40 #acc is 50
    def __init__(self, new_llm_every_req=True):
        """
        Initializes the MetaManager instance.

        Args:
        new_llm_every_req (bool): Whether to create a new LLM for every request. Defaults to True.
        """
        super().__init__(new_llm_every_req)
        #prime the metaAI
        thread = threading.Thread(target=self.prime_MetaAI)
        thread.start()
        # self.internal_meta = MetaAI()
        # self._internalLLM = None if new_llm_every_req else MetaAI()
    def _init_internal_llm(self):
        m = MetaAI()
        return m

    def prime_MetaAI(self):
        if self.internalLLM is not None:
            res = self.internalLLM.prompt("Hi")

    def prompt_llm(self, prompt_str):
      """
      Prompts the LLM with the given prompt string.

      Args:
      prompt_str (str): The prompt string to use.

      Returns:
      dict: The response from the LLM.
      """
      res = None
      self._increment_total_numb_requests_served() 

      if(self.internalLLM is None):
        # create new meta object for every req
        main_meta = MetaAI()
        res = main_meta.prompt(prompt_str)
        #return main_meta.prompt(prompt_str)
      else:
        # use same meta object as long as possible
        mai = self.internalLLM
        # assert_type(mai, MetaAI)
        res = mai.prompt(prompt_str)
        self._update_internalLLM() # update internal metaAI if necessary
 
      return res
    
    def _update_internalLLM(self):
      if(self.total_numb_requests_served % MetaManager.MAX_REQS_B4_TIMEOUT == 0):
            # Get new MetaAI() for internal meta to avoid time out
            self.internalLLM = MetaAI()
            # prime it so its now slow on the first req
            thread = threading.Thread(target=self.prime_MetaAI)
            thread.start()
class TogetherManager(LLMManager):
    MAX_RETRIES = 3
    """
    A manager class for handling Together API requests.
    """
    def __init__(self, ssManager, new_llm_every_req=True, 
                 TOGETHER_model_str="meta-llama/Llama-Vision-Free",
                 max_tokens= 512,
                 temperature=0.5,#0.1, #0.7,
                 top_p=0.4, #0.8, #0.7,
                 top_k=50): #30#50):
        """
        Initializes the TogetherManager instance.

        Args:
        new_llm_every_req (bool): Whether to create a new LLM for every request. Defaults to True.
        """
        super().__init__(new_llm_every_req)
        #  self.TOGETHER_model_str and self.internalLLM maybe the EXACT SAME FOR THIS CLASS
        self._TOGETHER_model_str = TOGETHER_model_str 
        self._ss = ssManager
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.top_p=top_p
        self.top_k=top_k

        # self.internal_meta = MetaAI()
        # self._internalLLM = None if new_llm_every_req else MetaAI()
    def enable_90B(self):
        self.TOGETHER_model_str = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    def disable_90B(self):
        self.TOGETHER_model_str = "meta-llama/Llama-Vision-Free"
        
    def _init_internal_llm(self):
        t = Together()
        return t
    @property
    def TOGETHER_model_str(self):
          """
          Gets the internal LLM instance.
          """
          return self._TOGETHER_model_str
    @TOGETHER_model_str.setter
    def TOGETHER_model_str(self, value):
          """
          Gets the internal LLM instance.
          """
          self._TOGETHER_model_str = value
    @property
    def ss(self):
          """
          Gets the internal LLM instance.
          """
          return self._ss
    
    # return response of LLAMA vision
    def meta_vision_wrapper(self, query, ss_link):
      client = self.internalLLM if self.internalLLM else Together() # use same LLM, else use new LLM each requests
      response = client.chat.completions.create(
          model=self.TOGETHER_model_str,
          messages=[
      #         {
      #                 "role": "user",
      #                 "content": [
      #                         {
      #                                 "type": "text",
      #                                 "text": "what is this "
      #                         },
      #                         {
      #                                 "type": "image_url",
      #                                 "image_url": {
      #                                         "url": "https://www.islandeguide.com/custom/domain_1/image_files/2_photo_11878.jpg" #s3://together-ai-uploaded-user-images-prod/c8606f29-76d7-4b52-a736-5e000b38b8ef.jpg"
      #                                 }
      #                         }
      #                 ]
      #         },
      #         {
      #                 "role": "assistant",
      #                 "content": "This is a QR code for WiFi."
      #         }
      # ],



          # {
          #   "role": "user",
          #   "content": [
          #     {"type": "text", 
          #      "text": "take ur time and solve this math problem correctly. Dont get distracted and only output the final correct answer." }, #"What sort of animal is in this picture? What is its usual diet? What area is the animal native to? And isn’t there some AI model that’s related to the image?"},
          #     {
          #       "type": "image_url",
          #       "image_url": {
          #         "url":  "https://drive.usercontent.google.com/u/0/uc?id=1nscxcJhIeyPfTds7uNNF4dSvhZLZwu6W&export=download"#"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/LLama.jpg/444px-LLama.jpg?20050123205659",
          #       },
          #     },  
          #   ],
          # }

          {
            "role": "user",
            "content": [
              {"type": "text", 
              "text": query }, #"What sort of animal is in this picture? What is its usual diet? What area is the animal native to? And isn’t there some AI model that’s related to the image?"},
              {
                "type": "image_url",
                "image_url": {
                  "url":  ss_link #"https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/LLama.jpg/444px-LLama.jpg?20050123205659",
                },
              },  
            ],
          }
          # ,{
          #   "role": "user",
          #   "content": [
          #     {"type": "text", 
          #      "text": "Nice. now only tell me the final answer value (only the number)." }
          #   ]
          # }
        ],

        max_tokens = self.max_tokens,
        temperature = self.temperature,
        top_p = self.top_p,
        top_k = self.top_k,
        #   max_tokens= 512,#50,#512,#50,#512,
        #   temperature=0.1,#0.3,#0.7,
        #   top_p= 0.8,#0.5,#0.90,#0.7,
        #   top_k=30,#10,#50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        # truncate=130560,
        stream=False
      )
    #   print(response.choices[0].message.content)
      return response.choices[0].message.content

    def ask_vision_meta(self,q):
      VoiceAssistant.voice("I'm reading")
      retries = 0
      vision_res = None
      while retries < TogetherManager.MAX_RETRIES:
          try:
              link_to_ss = self.ss.take_ss()
              time.sleep(0.5)
              vision_res = self.meta_vision_wrapper(q, link_to_ss)
              break  # Exit the loop if successful
          except Exception as e:
              retries += 1
              print(f"Attempt {retries} failed: {e}")
              if retries == TogetherManager.MAX_RETRIES:
                  print("Max retries exceeded. Giving up.")
                  return None
      return vision_res
        
    def prompt_llm(self, prompt_str):
      """
      Prompts the LLM with the given prompt string.

      Args:
      prompt_str (str): The prompt string to use.

      Returns:
      dict: The response from the LLM.
      """
      res = self.ask_vision_meta(prompt_str)
      self._increment_total_numb_requests_served()
      print("total reqs served: " + str(self.total_numb_requests_served))
      return res
      # if(self.internalLLM is None):
      #   # create new meta object for every req
      #   main_meta = MetaAI()
      #   return main_meta.prompt(prompt_str)
      # else:
      #   # use same meta object as long as possible
      #   res = self.internalLLM.prompt(prompt_str)
      #   self._update_internalLLM() # update internal metaAI if necessary
      #   return re

# ---------- Logging & State Tracker  ---------
class StateMeta(type):
    current_file_name = os.path.basename(__file__) # name of the Python file that is currently being executed.
    logging_filename = current_file_name.split('.')[0] + ".log"#"vision_hil_log_file.log"
    state_filename =  current_file_name.split('.')[0] + "_state.txt"

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(logging_filename, mode='w')
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    enabled = True  # New toggle variable

    def state_decorator(func):
        @functools.wraps(func)
        # IMPORTANT: "self" refers to the caller function!!!!
        def wrapper(self, *args, **kwargs):
            if StateMeta.enabled:  # Check the toggle
                # Log the function name and arguments
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                StateMeta.logger.info(f"Calling {func.__name__}({signature})")
                
                # Write the current state to the state file
                # with open(StateMeta.state_filename, 'w') as file:
                #     print(f"Calling {func.__name__}", file=file)
                #  Write the current state to the state file
                # llm = getattr(self, 'llm', None)
                # if llm:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Calling {func.__name__}, {llm.total_numb_requests_served}, 0", file=file)
                # else:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Calling {func.__name__}", file=file)
                # vm = getattr(self, 'vm', None)
                # # vm = getattr(self, 'vm', None)
                # # if hasattr(self, 'vm') and self.vm is not None:
                # #     print(f"DEBUGGGGG: {self.vm.total_numb_requests_served}")
                # # else:
                # #     print("DEBUGGGGG: self.vm is None or does not exist")
                # if vm:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Calling {func.__name__}, {self.llm.total_numb_requests_served}, {self.vm.total_numb_requests_served}", file=file)
                # else:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Calling {func.__name__}", file=file)
                # if not any(value is None for value in self.__dict__.values()):
                # if hasattr(self, 'vm') and hasattr(self, 'llm') and hasattr(self, 'total_connections_made'):
                
                # remember "self" is the caller func
                # only have to check if this attribute is set because llm and vm are in the parent class
                # print(self)

                # inits dont hve any attributes yet
                with open(StateMeta.state_filename, 'w') as file:
                    if func.__name__ == "__init__" or not hasattr(self, 'vm'):
                        print(f"Calling {func.__name__}", file=file)
                    elif hasattr(self, 'total_connections_made'):
                        print(f"Calling {func.__name__}, {self.llm.total_numb_requests_served},{self.vm.total_numb_requests_served},{self.total_connections_made}", file=file)
                    else:
                        print(f"Calling {func.__name__}, {self.llm.total_numb_requests_served},{self.vm.total_numb_requests_served}", file=file)
                            # print(f"Calling {func.__name__}", file=file)
                


            # Call the function
            result = func(self, *args, **kwargs)

            if StateMeta.enabled:  # Check the toggle
                # Log the function completion
                StateMeta.logger.info(f"Completed {func.__name__}({signature})")

                # # Write the completed state to the state file
                # with open(StateMeta.state_filename, 'w') as file:
                #     print(f"Completed {func.__name__}", file=file)
                # if hasattr(self, 'vm') and self.vm is not None:
                #     print(f"DEBUGGGGG: {self.vm.total_numb_requests_served}")
                # else:
                #     print("DEBUGGGGG: self.vm is None or does not exist")
                # with open(StateMeta.state_filename, 'w') as file:
                #     print(f"Completed {func.__name__}, {self.llm.total_numb_requests_served}, {self.vm.total_numb_requests_served}", file=file)
                
                # if hasattr(self, 'vm') and hasattr(self, 'llm') and hasattr(self, 'total_connections_made'):
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Completed {func.__name__}, {self.llm.total_numb_requests_served},{self.vm.total_numb_requests_served},{self.total_connections_made}", file=file)
                # else:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Completed {func.__name__}", file=file)

                # if hasattr(self, 'total_connections_made'):
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Completed {func.__name__}, {self.llm.total_numb_requests_served},{self.vm.total_numb_requests_served},{self.total_connections_made}", file=file)
                # else:
                #     with open(StateMeta.state_filename, 'w') as file:
                #         print(f"Completed {func.__name__}, {self.llm.total_numb_requests_served},{self.vm.total_numb_requests_served}", file=file)
    

            return result
        return wrapper

    def __new__(cls, name, bases, dct):
        for key, value in dct.items():
            if callable(value):
                dct[key] = cls.state_decorator(value)
        return super().__new__(cls, name, bases, dct)

    @classmethod
    def enable_logging(cls):
        cls.enabled = True

    @classmethod
    def disable_logging(cls):
        cls.enabled = False

# ---------- Automator Objects  ---------
# NVM... they are now instance variables
# purposely just class variables for now so that I don't have code duplication for now
# but changing these class variables affects ALL instances of the class -> not good for if different subclassses want diff configs of this parent class (in which case I have to change these to instance variables)
class Automator_LLMs(metaclass=StateMeta):
    # llm = None
    # vm = None
    def __init__(self, llm, vm, enable_voice=True):
        self.llm = llm
        self.vm = vm
        self.enable_voice = enable_voice
    def voice(self, input):
        if self.enable_voice and sys.platform == 'darwin':
            VoiceAssistant.voice(input)
        else:
            print(f" ** voice not available yet for this system: {sys.platform}")
    def ask_vision_meta(self, input):
            answer = self.vm.prompt_llm(input) 
            return answer
class ButtonClicker(Automator_LLMs):
    def __init__(self, llm, vm, descp, max_retries=2, pre_click_action=None, alt_click_action=None,safety_check=False, redun=2):
        self.descp = descp
        super().__init__(llm,vm)
        self.max_retries = max_retries
        self.pre_click_action = pre_click_action if pre_click_action else None 
        self.alt_click_action = alt_click_action if alt_click_action else None
        self.safety_check = safety_check
        self.redundancy = redun
        self.boost_90B = False
        if self.boost_90B:
            self.vm.enable_90B()
    # Deleting (Calling destructor)
    def __del__(self):
        if self.boost_90B:
            self.vm.disable_90B()

    def find_and_click(self):
        """
        Finds and clicks the button described by self.descp.
        """
        print(f"** INSIDE find_and_click_general func for {self.descp}")

        # if self.pre_click_action:
        #     self.pre_click_action()

        time.sleep(0.5)
        # retry = 0
        # while retry < self.max_retries:
        #     self._attempt_click(retry)
        #     retry += 1

        # if self.post_click_action:
        #     return self.post_click_action(res)
        
        try:
            self._attempt_click()
        except Exception as e:
            print(f"Raised Exception: {e}")
            raise Exception(f"Failed to click on {self.descp}")


        return 1  # bad

    def _attempt_click(self):
        """
        Attempts to click the button.
        """
        status = f"Let's click on {self.descp}"
        print(status)
        self.voice(status)

        retry = 0
        while retry < self.max_retries:
          if self.pre_click_action:
              self.pre_click_action()

          if self.safety_check:  
            check_for_button = f'''Given the image, can you identify {self.descp}? Only output YES or NO'''
            
            vision_res = None
            vision_res2 = None
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future1 = executor.submit(self.ask_vision_meta, check_for_button)
                if self.redundancy is not None:
                    future2 = executor.submit(self.ask_vision_meta, check_for_button)
                vision_res = future1.result()
                vision_res2 = future2.result()
                print(vision_res)
                print(vision_res2)
                # vision_res = self.ask_vision_meta(check_for_button)
                # print(vision_res)
            if self.redundancy is not None:
                if "yes" not in vision_res.lower() or "yes" not in vision_res2.lower():
                    status = f"CAN'T FIND {self.descp}! Retrying..."
                    print(status)
                    self.voice(status)
                    retry += 1
                    continue
            else:      
                if "yes" not in vision_res.lower():
                    status = f"CAN'T FIND {self.descp}! Retrying..."
                    print(status)
                    self.voice(status)
                    retry += 1
                    continue
          
          self._click_button()
          return 0  # good


          # double check
          if self.redundancy:
              vision_res2 = self.ask_vision_meta(check_for_button)
              print(vision_res2)
              if "yes" not in vision_res.lower() and "yes" not in vision_res2.lower():
                status = f"CAN'T FIND {self.descp}! Retrying..."
                print(status)
                self.voice(status)
                retry += 1
                continue
              else:
                self._click_button()
                return 0  # good
          else:      
            if "yes" not in vision_res.lower():
                status = f"CAN'T FIND {self.descp}! Retrying..."
                print(status)
                self.voice(status)
                retry += 1
                continue
            else:
                self._click_button()
                return 0  # good
            
        # OLD CODE        
        #   check_for_button = f'''Given the image, can you identify {self.descp}? Only output YES or NO'''
        #   vision_res = self.ask_vision_meta(check_for_button)
        #   print(vision_res)
        #     if "yes" not in vision_res.lower():
        #         status = f"CAN'T FIND {self.descp}! Retrying..."
        #         print(status)
        #         self.voice(status)
        #         retry += 1
        #         continue
        #     else:
        #         self._click_button()
        #         return 0  # good

        # # if self.alt_click_action():
        # #     self.alt_click_action()
        # else:
          
        raise Exception(f"Attempt click for {self.descp} failed")  

    def _click_button(self):
        """
        Clicks the button.
        """
        input_keys("f")
        followup = f'''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the LEFT or slightly UPPER LEFT side of the feature.
        Which lettered encoding is associated with {self.descp}? Start your answer with "The letter encoding is"'''
        #followup = f'''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box directly adjacent to the left of the feature. Which lettered encoding is associated with {self.descp}? Be brief and start your answer with "The letter encoding is "'''
        # followup= f'''Analyze the provided webpage image and identify the 1 or 2-letter encoding in the yellow box adjacent to {self.descp}. Respond with: "The letter encoding is [insert encoding]"'''
        followup= f'''Analyze the provided webpage image and identify the best 1-letter or 2-letter encoding in the yellow box adjacent left of the feature. Which lettered encoding is associated with {self.descp}? Respond with: "The letter encoding is "'''
        # followup= f'''Analyze the webpage image and identify the single best 1-2 letter encoding in the yellow box adjacent to {self.descp}. Respond with: "The best letter encoding is [insert encoding]"'''
        vision_res = self.ask_vision_meta(followup)
        print(vision_res)
        # res = meta.prompt(f"Given this input, only just output the one or two letter encoding: {vision_res}")
        res = self.llm.prompt_llm(f"Given this input, only just output the 1 or 2-letter encoding: {vision_res}")
        print(res["message"])
        input_keys(res["message"])
        status = f"Clicked on {self.descp}"
        print(status)
        self.voice(status)
class ConnectionExecutor(Automator_LLMs):
    def __init__(self, llm, vm, big_blob_info,max_retries=3,autosend=False,autoskip_connection=False):
        self.big_blob_info = big_blob_info
        super().__init__(llm,vm)
        self.local_meta = MetaAI()
        self.max_retries = max_retries
        self.autoskip_connection = autoskip_connection
        self.autosend=autosend
        self.hyperpersonalized_message = None

    def execute_connection(self):
        """
        Executes the connection process.
        """
        future = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(ConnectionExecutor.generate_personalized_msgs, self.big_blob_info, self.local_meta)
            self._attempt_connection()
        input_msg = future.result()
        self._handle_message_input(input_msg)

    def _attempt_connection(self):
        """
        Attempts to connect with the person.
        """
        # retries = 0
        # while retries < self.max_retries:
        #     result1 = find_and_click_connect_button()
        #     if result1 == 1:
        #         retries += 1
        #         refresh_page()
        #         print(f"Retrying buttons {retries}...")
        #         continue
        #     #result2 = find_and_click_add_note_button()
        #     add_note_button_clicker = ButtonClicker("grey 'Add a Note' button", self.llm, self.vm, max_retries=5)
        #     result2 = add_note_button_clicker.find_and_click()
        #     if result2 == 0:
        #         break
        #     else:
        #         retries += 1
        #         refresh_page()
        #         print(f"Retrying buttons {retries}...")

        # if self.max_retries == retries:
        #     status = "WAS NOT ABLE TO CONNECT"
        #     self.voice(status)
        #     raise Exception(status)
        

        attempt = 0
        def custom_pre_click_action():
            pyautogui.press('esc')
            scroll_to_top_of_page()
            scroll_half_page_down()
        while attempt < self.max_retries:
            try:
                # find_and_click_connect_button()
                # find_and_click_add_note_button()
                # def custom_alt_click_action(self, res):
                #       if find_and_click_hidden_connect_button() == 0:
                    

                # more_button_descp = 'grey "More" button'       
                # more_button_clicker = ButtonClicker(self.llm, self.vm, more_button_descp, max_retries=3, 
                #                                        pre_click_action=custom_pre_click_action,
                #                                        alt_click_action=find_and_click_hidden_connect_button)

                connect_button_descp = 'a button exactly labeled as "Connect" in black text with a blue background'#'the text "Connect" inside a blue button'#'a blue button labeled "Connect"'#'the blue Connect button'
                connect_button_clicker = ButtonClicker(self.llm, self.vm, connect_button_descp, max_retries=1, 
                                                       pre_click_action=custom_pre_click_action, safety_check=True)
                connect_button_clicker.find_and_click()
                add_note_descp = 'the grey "Add a Note" button near the bottom of the dialogue box' #'the grey "Add a Note" button'
                add_note_button_clicker = ButtonClicker(self.llm, self.vm, add_note_descp, max_retries=2)
                add_note_button_clicker.find_and_click()
                return
            except Exception as ex:
                print(f"\nGOT EXCEPTION: {str(ex)}\n")
                if "connect" in str(ex).lower():
                  try:
                      # Check if the Connect button is hidden by clicking on the grey More button
                      grey_more_button_descp = 'the grey "More" button'
                      more_button_clicker = ButtonClicker(self.llm, self.vm, grey_more_button_descp, max_retries=2,
                                                          pre_click_action=custom_pre_click_action)
                      more_button_clicker.find_and_click()
                      
                      # Click the grey Connect button
                      grey_connect_button_descp = 'the grey button exactly labeled as "Connect" in the dropdown menu'
                      grey_connect_button_clicker = ButtonClicker(self.llm, self.vm, grey_connect_button_descp, max_retries=2)
                      grey_connect_button_clicker.find_and_click()
                      
                      # intentional code duplication
                      add_note_descp = 'the grey "Add a Note" button near the bottom of the dialogue box' #"the grey 'Add a Note' button"
                      add_note_button_clicker = ButtonClicker(self.llm, self.vm, add_note_descp, max_retries=2)
                      add_note_button_clicker.find_and_click()
                      return
                      
                  except Exception as e:
                      print(f"\nError clicking grey More button, grey Connect button, or Add a Note button: {e}")

                refresh_page()
                print(f"Attempt {attempt+1} failed: {str(ex)}")
                attempt += 1

        raise Exception("Could not Attempt connection")        

    @staticmethod
    def generate_personalized_msgs(big_blob_info, local_meta):
      VoiceAssistant.voice("Generating the best personalized connection request message")
      aq1 = """ Craft three distinct, informal, and personalized LinkedIn connection request messages based on the provided profile information.
      Requirements:
      - Each message is under 300 characters
      - Each message is specific and detailed
      - Sounds naturally human but also be unique and have personality (don't use the word impressed or impressive and other cliches)
      - One message focuses on expressing interest in their experiences

      Avoid using templates or generic sign-offs.
      """
      ai_res = local_meta.prompt(aq1 + big_blob_info)
      final_answer_att1 = ai_res["message"]
      print(final_answer_att1)

      res = local_meta.prompt("pick the best one")
      print(res["message"])
      res = local_meta.prompt("only ouput the message")
      # print(res["message"])
      current_best_message = res["message"]
      while(len(current_best_message) > 298):
          #res = local_meta.prompt("pick the next best message and ensure it is under 300 characters")
          res = local_meta.prompt("Pick the next best one or generate a new message that is better")
          print(res["message"])
          res = local_meta.prompt("only ouput the message")
          # print(res["message"])
          current_best_message = res["message"]

      print(current_best_message)
      return current_best_message
    
    def go_back_to_profile_pg_from_adding_connection_note(self):
      """
      Goes back to the profile page from adding a connection note.
      """
      prompt_phrase = 'the grey "X" button near the top right of the dialogue box'
      button_clicker = ButtonClicker(self.llm, self.vm, prompt_phrase, max_retries=3)
      button_clicker.find_and_click()
      # OLD CODE/METHOD
    #   pyautogui.press('esc')  # Need to do this to get back to profile page
    #   prompt_phrase = 'the grey "Cancel" button in the dialogue box'
    #   button_clicker = ButtonClicker(self.llm, self.vm, prompt_phrase, max_retries=3)
    #   button_clicker.find_and_click()
    #   pyautogui.press('esc')  # Need to do this to get back to profile page

  
    def _get_user_confirmation(self):
        """
        Gets user confirmation to send the connection request.
        """ 
        self.voice("Should I sent it or no?")
        retry = True
        while retry:
            human_resp = MicInputRecorder().get_mic_input()
            if 'yes' in human_resp.lower():
                self._send_connection_request()
                retry = False
            elif 'no' in human_resp.lower():
                self._generate_new_message()
            elif 'skip' in human_resp.lower():
                self.voice("Alright! I will not send connection request. Skipping...")
                self.go_back_to_profile_pg_from_adding_connection_note()
                retry = False
            else:
                # self.voice("I didn't get that. Should i sent it or no?", block=True)
                self.voice("I didn't get that. Should i sent it or no?")

    def _send_connection_request(self):
        """
        Sends the connection request.
        """
        # self.voice("Okay, I will send it!", block=True)
        self.voice("Sending Message!")
        send_button_descp = 'the blue "Send" button near the bottom right of the dialogue box'
        send_button_clicker = ButtonClicker(self.llm, self.vm, send_button_descp, max_retries=2)
        send_button_clicker.find_and_click()
        # OLD CODE
        # pyautogui.press('esc')
        # check_for_connect_button = '''Given the image, can you identify a blue "Send" button near the bottom? Only output YES or NO'''
        # vision_res = self.ask_vision_meta(check_for_connect_button)
        # print(vision_res)
        # if "n" in vision_res.lower():
        #     print("BAD BUTTON!... going back")
        #     input_keys("z")
        # input_keys("f")
        # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
        # Which encoding is associated with the blue "Send" button? Start your answer with "The letter encoding is"
        # '''
        # vision_res = self.ask_vision_meta(followup)
        # print(vision_res)
        # # res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
        # res = self.llm.prompt_llm("Given this input, only just output the one or two encoding: " + vision_res)
        # print(res["message"])
        # input_keys(res["message"])
        # self.voice("Sent Connection Request")

    def _handle_message_input(self, input_msg):
        """
        Handles the message input from the user.
        """
        
        # self.voice("Here is the best message I came up with", block=True)
        self.voice("Here is the best message I came up with")
        self.voice(input_msg)
        fill_in_txtbox(input_msg)
        self.hyperpersonalized_message = input_msg
        if self.autoskip_connection:
            self.voice("Alright! I will not send connection request. Skipping...")
            self.go_back_to_profile_pg_from_adding_connection_note()
        elif self.autosend:
            self._send_connection_request()
        else:
            self._get_user_confirmation()

    def _generate_new_message(self):
        """
        Generates a new message.
        """
        self.voice("No worries! I will generate a new message")
        res = self.local_meta.prompt("Pick the next best one or generate a new message that is better")
        print(res["message"])
        res = self.local_meta.prompt("Only output the message")
        print(res["message"])
        new_msg = res["message"]
        clear_prefilled_txt_box()
        # self.voice("Here is the new message I came up with", block=True)
        self.voice("Here is the new message I came up with")
        self.voice(new_msg)
        fill_in_txtbox(new_msg)
        self.hyperpersonalized_message = new_msg
        # self.voice("Should I sent it or no?", block=True)
        self.voice("Should I sent it or no?")
class PersonFinder(Automator_LLMs):
    def __init__(self, llm, vm, max_retries=5):
        self.max_retries = max_retries
        super().__init__(llm,vm) 

    def find_next_person(self):
        """
        Finds the next person to connect with.
        """
        self.voice("Let's Find The Next Person To Connect With!")
        pyautogui.press('esc')
        retry = 0
        while retry < self.max_retries:
            self._scroll_to_connect_with_others()
            vision_res = self.ask_vision_meta(self._check_list_of_sim_ppl())
            print(vision_res)
            if not self._is_vision_res_positive(vision_res):
                print("ERR..CANT FIND OTHER PPL! Retrying...")
                self.voice("ERR..CANT FIND OTHER PEOPLE! Retrying...")
                retry += 1
            else:
                self._select_next_person()
                break

    def _scroll_to_connect_with_others(self):
        """
        Scrolls to the section where similar people to connect with are listed.
        """
        scroll_to_top_of_page()
        scroll_half_page_down()

    def _check_list_of_sim_ppl(self):
        """
        Returns the question to ask the vision model to check if the list of similar people is visible.
        """
        return "Given the image, can you identify names of other similar people to connect with? Only output YES or NO"

    def _is_vision_res_positive(self, vision_res):
        """
        Checks if the vision model response indicates that the list of similar people is visible.
        """
        # meta_res = meta.prompt("Given this input, only tell me if the input is a yes or no: " + vision_res)
        meta_res = self.llm.prompt_llm("Given this input, only tell me if the input is a yes or no: " + vision_res)
        return "yes" in meta_res["message"].lower()

    def _select_next_person(self):
        """
        Selects the next person to connect with.
        """
        input_keys("f")
        # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
        # Which encoding is associated with first picture of a face of one of the similar people near the bottom? Start your answer with "The letter encoding is"
        # '''

        # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the LEFT side of the feature.
        # Which lettered encoding is associated with the first picture of a face of one of the similar people? Start your answer with "The letter encoding is"'''
        # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the LEFT side of the feature. Which lettered encoding is associated with the first picture of a face of one of the similar people? Start your answer with "The letter encoding is"'''
        # Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the LEFT or slightly UPPER LEFT side of the feature.
        #     Which lettered encoding is associated with {self.descp}?
        descp = "the first picture of a face from the list of similar people"
        followup=f'''Analyze the provided webpage image and identify the best 1-letter or 2-letter encoding in the yellow box adjacent left of the feature. Which lettered encoding is associated with {descp}? Respond with: "The letter encoding is'''
        vision_res = self.ask_vision_meta(followup)
        print(vision_res)
        # res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
        res = self.llm.prompt_llm("Given this input, only just output the one or two letter encoding: " + vision_res)
        print(res["message"])
        input_keys(res["message"])
class LIAutomator(Automator_LLMs):
    METRIC_FIELDNAMES = ['date','name', 'profile_recon_info', 'custom_msg', 'execution_time', 'autoskip_connection', 'speed_recon']

    def __init__(self,llm, vm, autosend=False, autoskip_connection=False, speed_recon=False):
        super().__init__(llm,vm)
        self.autoskip_connection = autoskip_connection
        self.autosend= autosend
        self.speed_recon = speed_recon
        self._total_connections_made = 0
        self.metrics = []

    def _increment_total_connections_made(self):
        self._total_connections_made += 1

    @property
    def total_connections_made(self):
        return self._total_connections_made   

    def execute_full_connection_process(self):
        start_time = time.perf_counter()

        # activate_window("Linkedin")
        print("in start_full_conneciotn func... sleeping for 1 secs")
        time.sleep(1)
        scroll_counter = 0
        one_time_meta = MetaAI()
        big_blob = ""

        scroll_to_top_of_page()
        self.voice("Starting to Connect with new person")

        def update_big_blob(future):
            try:
                result = future.result()
                print(result)
                nonlocal big_blob
                big_blob = big_blob + (result)
            except Exception as e:
                print(f"Error updating big_blob: {e}")

            # try:
            #     result = future.result()
            #     result_queue.put(result)
            # except Exception as e:
            #     print(f"Error updating big_blob: {e}")

        if self.speed_recon:
            # orig_vm = self.vm
            # self.vm = TogetherManager(self.vm.ss, new_llm_every_req=True)
            while True:
                #req_check_interests = "Given the image, does the image contain, in whole or in part, an Education or Skills section? Only output YES or NO"
                #req_check_interests = "Based on the provided LinkedIn profile image, have we reached near the end of the profile based on the scroll bar to the right? Only respond with YES or NO"
                # req_check_interests = "Based on the provided LinkedIn profile image, have we reached near the end of their EXPERIENCES section? Only respond with YES or NO"
                req_check_interests = "Based on the provided LinkedIn profile image, does the image contain, in whole or in part, the Education section and specfic info on their education? Only respond with YES or NO"
                followup = '''I am trying to learn more about an person from their Linkedin profile.
                Only output relevant or unqiue information about this person from this LinkedIn screenshot. Use bullet points. 
                '''
                print("scroll counter = " + str(scroll_counter))
                # Create a list to store the futures
                futures = []
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    # Submit tasks to the thread pool
                    future1 = executor.submit(self.ask_vision_meta, req_check_interests)
                    future2 = executor.submit(self.ask_vision_meta, followup)

                    # Schedule a callback to update big_blob when future2 is done
                    future2.add_done_callback(update_big_blob)
                    # Wait for the first task to complete and get the result
                    vision_res = future1.result()
                    print(vision_res)
                                       
                    if "yes" in vision_res.lower():
                        print("END OF PROFILE!")
                        # self.voice("Done Gathering profile info...Branch 1.. Good")
                        self.voice("Done Gathering profile info!......Fantastic!")
                        break
                    elif scroll_counter >= 20:
                        res = one_time_meta.prompt("Given the below text, does it contain information about someone's education? Just answer YES or NO.  " + vision_res)
                        if "yes" not in res["message"].lower():
                            self.voice("Done Gathering profile info...Branch 2...Kinda Sus")
                            break  # FORCED BREAKING
                        break # GOOD BREAKING
                    else:
                        # # Process any results that are already in the queue
                        # while not result_queue.empty():
                        #     result = result_queue.get()
                        #     big_blob += result
                        #     print(result)
                        # Wait for the second task to complete and get the result
                        # followup_res = future2.result()
                        # print(followup_res)
                        # big_blob = big_blob + (followup_res)
                        
                        scroll_full_page_down()
                        scroll_counter = scroll_counter + 1
            # self.vm = orig_vm
        else:
            while True:
                req_check_interests = "Given the image, does the image contain, in whole or in part, an Education or Skills section? Only output YES or NO"
                vision_res = self.ask_vision_meta(req_check_interests)
                print(vision_res)
                print("scroll coutner = " + str(scroll_counter))

                if "yes" in vision_res.lower():
                    print("END OF PROFILE!")
                    self.voice("Done Gathering profile info...Branch 1.. Good")
                    break
                elif scroll_counter >= 4:
                    res = one_time_meta.prompt("Given the below text, does it contain information about someone's education? Just answer YES or NO.  " + vision_res)
                    if "yes" not in res["message"].lower():
                        self.voice("Done Gathering profile info...Branch 2...Kinda Sus")
                        break
                    break
                else:
                    followup = '''I am trying to learn more about an person from their Linkedin profile.
                    Only output relevant or unqiue information about this person from this LinkedIn screenshot. Use bullet points. 
                    '''

                    vision_res = self.ask_vision_meta(followup)
                    print(vision_res)
                    big_blob = big_blob + (vision_res)

                    scroll_full_page_down()
                    scroll_counter = scroll_counter + 1

        # execute_connection(big_blob_info=self.big_blob)
        ec = ConnectionExecutor(self.llm, self.vm, big_blob_info=big_blob, max_retries=3, autosend=self.autosend, autoskip_connection=self.autoskip_connection)
        ec.execute_connection()
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        self._increment_total_connections_made()
        self.save_metrics(big_blob, execution_time, ec)
        print("Start full Connection process func done!!!!")

    def save_metrics(self, big_blob, execution_time, ec):
        """
        Saves the metrics to a CSV file and a SQLite database.
        """
        # cur_date=str(date.today())
        cur_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        metric = {
            'date': cur_timestamp,
            'name': None,
            'profile_recon_info': big_blob,
            'custom_msg': ec.hyperpersonalized_message ,
            'execution_time': execution_time,
            'autoskip_connection': self.autoskip_connection,
            'speed_recon': self.speed_recon
        }
        self.metrics.append(metric)
        with open('metrics.csv', 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.METRIC_FIELDNAMES)
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(metric)

        # conn = sqlite3.connect('metrics.db')
        # c = conn.cursor()
        # c.execute('''CREATE TABLE IF NOT EXISTS metrics
        #              ({})'''.format(', '.join([f'{field} text' if field != 'execution_time' else f'{field} real' for field in self.METRIC_FIELDNAMES])))
        # c.execute("INSERT INTO metrics VALUES (:{})".format(', '.join(self.METRIC_FIELDNAMES)), metric)
        # conn.commit()
        # conn.close()

    def find_next_person(self):
        pf = PersonFinder(self.llm, self.vm)
        pf.find_next_person()
def main():
  ### OLD MAIN ###
  # start_time = time.perf_counter()
  # print("Starting OBJECT ORIENTED VHIL.py")
  # ss = ScreenshotManager()
  # ss.config_ss_params()
  # activate_window("Linkedin")
  # for i in range (10):
  #   execute_full_connection_process()
  #   find_next_person()

  # end_time = time.perf_counter()
  # execution_time = end_time - start_time
  # print(f"Execution time: {execution_time:.2f} seconds")
  try:   
    start_time = time.perf_counter()
    print("Starting OBJECT ORIENTED VHIL.py")

    StateMeta.enable_logging()  # Enable logging
    # StateMeta.disable_logging()  # Disable logging

    #   activate_window("Linkedin")
    ss = ScreenshotManager(entire_window=True,custom_use_defaults=False, window_name="LinkedIn")
    ss.config_ss_params() # also does activate_window

    mm = MetaManager(new_llm_every_req=False) # False might be faster
    # big_llama_vm = "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"
    # tm = TogetherManager(ss, new_llm_every_req=True, TOGETHER_model_str=big_llama_vm) # new_llm_every_req=True necessary at this stage for TOGETHER 
    tm = TogetherManager(ss, new_llm_every_req=True)
    # la = LIAutomator(mm, tm, autoskip_connection=True, speed_recon=True)
    la = LIAutomator(mm, tm,autosend=True, autoskip_connection=False, speed_recon=True)

    # testing screenshot coords
    # while True:
    #     tm.ask_vision_meta("boi")
    #     time.sleep(0.5)

    # ConnectionExecutor Test
    # c = ConnectionExecutor(mm,tm, None, 3, True)
    # c.execute_connection()
    # grey_connect_button_descp = 'the grey button exactly labeled as "Connect" in the dropdown menu'
    # grey_connect_button_clicker = ButtonClicker(mm, tm, grey_connect_button_descp, max_retries=2)
    # grey_connect_button_clicker.find_and_click()
    #   c.go_back_to_profile_pg_from_adding_connection_note()
    
    # Testing refreshes
    # for i in range(10):
    #     # refresh_page()
    #     keyboard = Controller()
    #     with keyboard.pressed(Key.cmd):
    #         keyboard.press('r')
    #         keyboard.release('r')
    #     time.sleep(8)
    # time.sleep(10)


    # la.find_next_person()
    VoiceAssistant.enable_voice()
    for i in range (100):
        la.execute_full_connection_process()
        la.find_next_person()

    


    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")
  
  except KeyboardInterrupt:
    ss.stop_updates()
    print("Stopping update for SS coords...")
    print("vhil OO execution stopped")
if __name__ == "__main__":
  main()





# # Debug function to engineer better prompts for LLMs
# # Prompts LLM with user inputted prompt + context_info
# def debug_prompt_engin_meta(context_info, new_metas=False):
   
#   while True:
#       user_input = input("[DEBUG] Please enter an prompt for Meta (type 'q' to exit): ")
#       if user_input.lower() == "q":
#           break
#       # Perform your desired action with the user_input
#       print(f"You entered: {user_input}")
#       #meta_op = meta.prompt(user_input) 
#       # vision_res = ask_vision_meta(user_input)
#       # print(vision_res)
#       same_meta = MetaAI()
#       if new_metas:
#         nm = MetaAI()
#         print(f"Meta object id = {id(nm)}")
#         meta_res = nm.prompt(user_input + context_info)
#         print(meta_res["message"])
#       else:
#         print(f"Meta object id = {id(same_meta)}")
#         meta_res = same_meta.prompt(user_input + context_info)
#         print(meta_res["message"])




# i think making a keyboard class is dumb, just to hide the pynput object.. most of these functions are pure
# class KeyboardAssistant:
#     pynput_keyboard = Controller()
#     @staticmethod
#     def input_keys(keys):
#         """
#         Simulate keyboard input.

#         Args:
#         keys (str): The keys to be pressed.
#         """
#         print(f"** Exec Input keys: {keys.lower()}")
#         for letter in keys.lower():
#             KeyboardAssistant.pynput_keyboard.type(letter, duration=0.000001, delay=0.1)
#             time.sleep(1.5)


#     def scroll_full_page_down():
#         """
#         Scroll down a full page.
#         """
#         VoiceAssistant.self.voice("I'm scrolling")
#         KeyboardAssistant.input_keys("dd")

#     @staticmethod
#     def scroll_half_page_down():
#         """
#         Scroll down half a page.
#         """
#         VoiceAssistant.self.voice("I'm scrolling")
#         KeyboardAssistant.input_keys("d")

#     @staticmethod
#     def scroll_to_top_of_page():
#         """
#         Scroll to the top of the page.
#         """
#         time.sleep(0.5)
#         VoiceAssistant.self.voice("I'm scrolling")
#         KeyboardAssistant.input_keys("gg")

#     @staticmethod
#     def clear_prefilled_txt_box():
#         """
#         Clear a pre-filled text box.
#         """
#         pyautogui.press('esc')
#         KeyboardAssistant.input_keys("gi")
#         time.sleep(0.5)
#         pyautogui.hotkey('command', 'a')
#         time.sleep(0.5)
#         pyautogui.press('delete')

#     @staticmethod
#     def fill_in_txtbox(input_text):
#         """
#         Fill in a text box with the given input.

#         Args:
#         input_text (str): The text to be filled in.
#         """
#         pyautogui.press('esc')
#         time.sleep(0.2)
#         KeyboardAssistant.input_keys("gi")
#         time.sleep(0.3)
#         for char in input_text:
#             if not char.isalpha():
#                 pyautogui.typewrite(char)
#             else:
#                 pynput_keyboard.type(char, 0.02, 0.02)







# class MicInputRecorder:
#     def __init__(self, whisper_model_name='tiny',duration=3, sample_rate=44100, channels=1, input_audio_file_name="responding_to_vision.wav"):
#         """
#         Initialize MicInputRecorder.

#         Args:
#             whisper_model_name (str): Whisper model name. Defaults to 'tiny'.
#             duration (int): Recording duration in seconds. Defaults to 3.
#         """
#         self.whisper_model_name = whisper_model_name
#         self.model = whisper.load_model(whisper_model_name)
#         self.duration = duration
#         self.sample_rate = sample_rate  # Hz
#         self.channels = channels  # channel=1 is Mono
#         self.input_audio_file_name = input_audio_file_name

#     def record_audio(self):
#         """
#         Record audio from microphone.

#         Returns:
#             np.ndarray: Recorded audio.
#         """
#         print("Recording...")
#         audio = sd.rec(int(self.duration * self.sample_rate), 
#                        samplerate=self.sample_rate, 
#                        channels=self.channels)
#         sd.wait()  # Wait for recording to finish
#         print("Recording finished.")
#         return audio

#     def save_audio_to_wav(self, audio):
#         """
#         Save recorded audio to WAV file.

#         Args:
#             audio (np.ndarray): Recorded audio.
#         """
#         # Ensure highest value is in 16-bit range
#         audio *= 32767 / np.max(np.abs(audio))
#         audio = audio.astype(np.int16)

#         wave_file = wave.open(self.input_audio_file_name, "wb")
#         wave_file.setnchannels(self.channels)
#         wave_file.setsampwidth(2)  # 16-bit audio
#         wave_file.setframerate(self.sample_rate)
#         wave_file.writeframes(audio.tobytes())
#         wave_file.close()

#     def transcribe_audio(self):
#         """
#         Transcribe recorded audio using Whisper.

#         Returns:
#             str: Transcribed text.
#         """
       
#         result = self.model.transcribe(self.input_audio_file_name)
#         print("Transcription from mic: ")
#         print(result["text"])
#         print("Transcription done.")
#         return result["text"]

#     def get_mic_input(self):
#         """
#         Record and transcribe microphone input.

#         Returns:
#             str: Transcribed text.
#         """
#         audio = self.record_audio()
#         self.save_audio_to_wav(audio)
#         return self.transcribe_audio()


# def generate_personalized_msgs(big_blob_info, local_meta):
#   VoiceAssistant.self.voice("Generating the best personalized connection request message")
#   aq1 = """ Craft three distinct, informal, and personalized LinkedIn connection request messages based on the provided profile information.
#   Requirements:
#   - Each message under 500 characters
#   - Sounds naturally human
#   - One message focuses on expressing interest in their experiences

#   Avoid using templates or generic sign-offs.
#   """
  
#   # aq1='''
#   # Given the below information about an person based on their Linkedin Profile, please craft 3 different informal personalized connection request messages for this person.  
#   # Ensure each message is under 500 characters and sounds exactly like a human. Also, consider making one of the messages about how you are interested in their experiences.
#   # Don't use any templates or valedictions!
#   # '''
  
#   # aq1='''
#   # Given the below information about an person based on their Linkedin Profile, please craft 3 different personalized connection request messages for this person.  
#   # Ensure each message is under 500 characters, is informal, sounds exactly like a human, and is excited to chat. Also, consider making one of the messages about how you are interested in their experiences.
  
#   # '''
#   ai_res = local_meta.prompt(aq1 + big_blob_info)
#   final_answer_att1 = ai_res["message"]
#   print(final_answer_att1)

#   # debug_prompt_engin_meta(context_info=big_blob_info,new_metas=True)

#   #Voice: thinking of best personalized connection request message
#   # while True:
#   #     user_input = input("Please enter something (type 'q' to exit): ")
#   #     if user_input.lower() == "q":
#   #         break
#   #     # Perform your desired action with the user_input
#   #     print(f"You entered: {user_input}")
#   #     #meta_op = meta.prompt(user_input) 
#   #     res = meta.prompt(user_input)
#   #     print(res["message"])
#   #    # check state where state is "Did we reach the interests section of this person's Linkedin Profile?"
#   #    # if yes: output STOP
#   #    # if no: ouput important information presented in the image
#   #     # press d

#   res = local_meta.prompt("pick the best one")
#   print(res["message"])
#   res = local_meta.prompt("only ouput the messagee")
#   # print(res["message"])
#   current_best_message = res["message"]
#   print(current_best_message)

#   return current_best_message


# def find_and_click_cancel_button():
#   pyautogui.press('esc')
#   time.sleep(0.5)
#   check_for_cancel_button = '''Given the image, can you identify the word "Cancel" in gray color near the bottom right? Only output YES or NO'''
#   vision_res = ask_vision_meta(check_for_cancel_button)
#   print(vision_res)
#   if "n" in vision_res.lower():
#       print("BAD BUTTON!... goign back")
#       # input_keys("z")
#   input_keys("f") 
#   # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to or hovering in front of the feature.
#   # Which encoding is hovering over the grey Cancel button near the bottom? Start your answer with "The letter encoding is"
#   # '''

#   #  This new follow up is designed by Vision
#   # followup = '''Identify the yellow box that corresponds to the Cancel button in the image. 
#   # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#   # The Cancel button is located near the bottom of the image and has the word 'Cancel' in it. 
#   # Please output the letter or letters of the yellow box that corresponds to the Cancel button'''

#   followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the left side of the feature.
#                 Which encoding is associated with the word "Cancel" in gray color? Start your answer with "The letter encoding is'''
#   # followup = '''Tell me the 1 lettered encoding that appears next to and to the left of the word "Cancel" in gray color near the bottom right in the image.'''

#   vision_res = ask_vision_meta(followup)
#   print(vision_res)
#   res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
#   print(res["message"])
#   input_keys(res["message"])
# # cancels adding connection note and goes back
# def go_back_to_profile_pg_from_adding_connection_note():
#   find_and_click_cancel_button()
#   pyautogui.press('esc') # need to do this to get back to profile page

#   # find_and_click_X_button()
#   # ** Below method works perfectly but we have to increase the screenshot parameters to be bigger*
#   # check_for_X_button = '''Given the image, can you identify a grey cancel X symbol near the top right? Only output YES or NO'''
#   # vision_res = ask_vision_meta(check_for_X_button)
#   # print(vision_res)
#   # if "n" in vision_res.lower():
#   #     print("BAD BUTTON!... goign back")
#   #     input_keys("z")
#   # input_keys("f") 
#   # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#   # Which encoding is associated with the grey cancel X symbol near the top right? Start your answer with "The letter encoding is"
#   # '''
#   # vision_res = ask_vision_meta(followup)
#   # print(vision_res)
#   # res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
#   # print(res["message"])
#   # input_keys(res["message"])
#   # VoiceAssistant.self.voice("Clicked X button")

# def find_and_click_connect_button():
#   pyautogui.press('esc')
#   retry = 0
#   while retry < 3:
#     scroll_to_top_of_page() # input_keys("gg")# go to top of page
#     scroll_half_page_down() # input_keys("d")# ensure we can see the connection page 
#     #Voice: Let's click the button to connect
#     VoiceAssistant.self.voice("Let's click the button to connect")
#     check_for_connect_button = '''Given the image, can you identify a blue "Connect" button? Only output YES or NO'''
#     vision_res = ask_vision_meta(check_for_connect_button)
#     print(vision_res)
#     if "n" in vision_res.lower():
#         # print("CAN'T FIND CONNECT BUTTON! Retrying...")
#         # VOICE: "CAN'T FIND CONNECT BUTTON"
#         # VoiceAssistant.self.voice("CAN'T FIND CONNECT BUTTON. Retrying...")
#         print("Connect button seems to be hidden")
#         VoiceAssistant.self.voice("Connect button seems to be hidden")
#         if (find_and_click_hidden_connect_button() == 0):
#            return 0 # good
#         # retry = retry + 1
#     else:
#         input_keys("f")
#         followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#         # Which encoding is associated with the blue Connect button near the bottom? Start your answer with "The letter encoding is"
#         # '''
#         # followup = '''Identify the yellow box that corresponds to the blue Connect button in the image. 
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button'''
#         # followup = '''Identify the 1 or 2 lettered encoding in a yellow box that corresponds to the blue Connect button in the image.
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button 
#         # in the image and please estimate the level of accuracy of the identification.'''          
        
#         # followup = '''Identify the 1 or 2 lettered encoding inside of a yellow box that corresponds to the blue Connect button in the image.
#         # The 1 or 2 lettered encoding in a yellow box is often next to and to the left of the feature/button it corresponds to.
#         # Please output the encoding that corresponds to the blue Connect button
#         # in the image and please estimate the level of accuracy of the identification.
#         # '''

#         # followup = '''Tell me the 1 or 2 lettered encoding that appears next to and to the left of the blue Connect button in the image.'''
#         vision_res = ask_vision_meta(followup)
#         print(vision_res)
#         # sys.exit()
#         res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
#         print(res["message"])
#         # res2 = meta.prompt("Given this input, only just output the level of accuracy of the identification as a percentage. " + vision_res)
#         # print(res2["message"])

#         # time.sleep(2)
#         input_keys(res["message"])
#         print("Clicked on Connect Button")
#         VoiceAssistant.self.voice("Clicked on Connect Button") 
#         return 0
#   return 1 # bad

# def find_and_click_hidden_connect_button():
#   retry = 0
#   while retry < 3:
#     scroll_to_top_of_page() # input_keys("gg")# go to top of page
#     scroll_half_page_down() # input_keys("d")# ensure we can see the connection page 
#     #Voice: Let's click the button to connect
#     VoiceAssistant.self.voice("Let's click the button to connect")
#     check_for_connect_button = '''Given the image, can you identify a grey "More" button? Only output YES or NO'''
#     vision_res = ask_vision_meta(check_for_connect_button)
#     print(vision_res)
#     if "n" in vision_res.lower():
#         print("CAN'T FIND MORE BUTTON! Retrying...")
#         # VOICE: "CAN'T FIND CONNECT BUTTON"
#         VoiceAssistant.self.voice("CAN'T FIND MORE BUTTON. Retrying...")
#         retry = retry + 1
#     else:
#         input_keys("f")
#         followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#         # Which encoding is associated with the grey "More" button? Start your answer with "The letter encoding is"
#         # '''
#         # followup = '''Identify the yellow box that corresponds to the blue Connect button in the image. 
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button'''
#         # followup = '''Identify the 1 or 2 lettered encoding in a yellow box that corresponds to the blue Connect button in the image.
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button 
#         # in the image and please estimate the level of accuracy of the identification.'''          
        
#         # followup = '''Identify the 1 or 2 lettered encoding inside of a yellow box that corresponds to the blue Connect button in the image.
#         # The 1 or 2 lettered encoding in a yellow box is often next to and to the left of the feature/button it corresponds to.
#         # Please output the encoding that corresponds to the blue Connect button
#         # in the image and please estimate the level of accuracy of the identification.
#         # '''

#         # followup = '''Tell me the 1 or 2 lettered encoding that appears next to and to the left of the blue Connect button in the image.'''
#         vision_res = ask_vision_meta(followup)
#         print(vision_res)
#         # sys.exit()
#         res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
#         print(res["message"])
#         # res2 = meta.prompt("Given this input, only just output the level of accuracy of the identification as a percentage. " + vision_res)
#         # print(res2["message"])

#         # time.sleep(2)
#         input_keys(res["message"])
#         if (find_and_click_general("grey Connect button") == 0):
#            return 0 # good

#   return 1; # bad

#         # print("Clicked on Connect Button")
#         # VoiceAssistant.self.voice("Clicked on Connect Button") 
       


# def find_and_click_add_note_button():
#   retry = 0
#   while retry < 3:
#     check_for_connect_button = '''Given the image, can you identify a "Add a Note" button near the bottom? Only output YES or NO'''
#     vision_res = ask_vision_meta(check_for_connect_button)
#     print(vision_res)
#     if "n" in vision_res.lower():
#         print("CAN'T FIND Add a note BUTTON. Retrying...")
#         VoiceAssistant.self.voice("CAN'T FIND Add a note BUTTON. Retrying...")
#         retry = retry + 1
        
#     else:
#         input_keys("f")
#         followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#         Which encoding is associated with the grey "Add a Note" button? Start your answer with "The letter encoding is"
#         '''
#         vision_res = ask_vision_meta(followup)
#         print(vision_res)
#         res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
#         print(res["message"])
        
#         input_keys(res["message"])
#         # VOICE: Adding Connection Note
#         print("Clicked on Add Connection Note")
#         VoiceAssistant.self.voice("Clicked on Add Connection Note")
#         return 0 # good
#   return 1 # bad
# # descp="grey Connect button"
# def find_and_click_general(descp):
#   print("** INSIDE find_and_click_general func")
#   retry = 0
#   while retry < 3:
#     # scroll_to_top_of_page() # input_keys("gg")# go to top of page
#     # scroll_half_page_down() # input_keys("d")# ensure we can see the connection page 
#     #Voice: Let's click the button to connect
#     VoiceAssistant.self.voice("Let's click the button to connect")
#     # check_for_connect_button = '''Given the image, can you identify a grey "More" button? Only output YES or NO'''
#     check_for_connect_button = '''Given the image, can you identify a ''' + descp + '''? Only output YES or NO'''
#     print(check_for_connect_button)
#     vision_res = ask_vision_meta(check_for_connect_button)
#     print(vision_res)
#     # if "n" in vision_res.lower():
#     if "yes" not in vision_res.lower():
#         status = "CAN'T FIND " + descp + "! Retrying..."
#         print(status)
#         # VOICE: "CAN'T FIND CONNECT BUTTON"
#         VoiceAssistant.self.voice(status)
#         retry = retry + 1
#     else:
#         input_keys("f")
#         followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#         Which encoding is associated with the ''' + descp + '''? Start your answer with "The letter encoding is"'''
#         print(followup)
#         # '''
#         # followup = '''Identify the yellow box that corresponds to the blue Connect button in the image. 
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button'''
#         # followup = '''Identify the 1 or 2 lettered encoding in a yellow box that corresponds to the blue Connect button in the image.
#         # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#         # The blue Connect button is most likely located near the center. 
#         # Please output the letter(s) of the yellow box that corresponds to the blue Connect button 
#         # in the image and please estimate the level of accuracy of the identification.'''          
        
#         # followup = '''Identify the 1 or 2 lettered encoding inside of a yellow box that corresponds to the blue Connect button in the image.
#         # The 1 or 2 lettered encoding in a yellow box is often next to and to the left of the feature/button it corresponds to.
#         # Please output the encoding that corresponds to the blue Connect button
#         # in the image and please estimate the level of accuracy of the identification.
#         # '''

#         # followup = '''Tell me the 1 or 2 lettered encoding that appears next to and to the left of the blue Connect button in the image.'''
#         vision_res = ask_vision_meta(followup)
#         print(vision_res)
#         # sys.exit()
#         res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
#         print(res["message"])
#         # res2 = meta.prompt("Given this input, only just output the level of accuracy of the identification as a percentage. " + vision_res)
#         # print(res2["message"])

#         # time.sleep(2)
#         input_keys(res["message"])


#         # print("Clicked on Connect Button")
#         # VoiceAssistant.self.voice("Clicked on Connect Button") 
#         status = "Clicked on " + descp
#         print(status)
#         # VOICE: "CAN'T FIND CONNECT BUTTON"
#         VoiceAssistant.self.voice(status)
#         return 0 # good
#   return 1 # bad







# def execute_connection(big_blob_info):
#      ################################
#     # find_and_click_connect_button()

#      ################################
#     # retry = 0
#     # while retry < 3:
#     #   scroll_to_top_of_page() # input_keys("gg")# go to top of page
#     #   scroll_half_page_down() # input_keys("d")# ensure we can see the connection page 
#     #   #Voice: Let's click the button to connect
#     #   VoiceAssistant.self.voice("Let's click the button to connect")
#     #   check_for_connect_button = "Given the image, can you identify a blue Connect button? Only output YES or NO"
#     #   vision_res = ask_vision_meta(check_for_connect_button)
#     #   print(vision_res)
#     #   if "n" in vision_res.lower():
#     #       print("CAN'T FIND CONNECT BUTTON! Retrying...")
#     #       # VOICE: "CAN'T FIND CONNECT BUTTON"
#     #       VoiceAssistant.self.voice("CAN'T FIND CONNECT BUTTON. Retrying...")
#     #       retry = retry + 1
#     #   else:
#     #       input_keys("f")
#     #       # followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#     #       # Which encoding is associated with the blue Connect button near the bottom? Start your answer with "The letter encoding is"
#     #       # '''
#     #       # followup = '''Identify the yellow box that corresponds to the blue Connect button in the image. 
#     #       # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#     #       # The blue Connect button is most likely located near the center. 
#     #       # Please output the letter(s) of the yellow box that corresponds to the blue Connect button'''
#     #       # followup = '''Identify the 1 or 2 lettered encoding in a yellow box that corresponds to the blue Connect button in the image.
#     #       # The yellow box will often be toward the left side of the feature/button it corresponds to. 
#     #       # The blue Connect button is most likely located near the center. 
#     #       # Please output the letter(s) of the yellow box that corresponds to the blue Connect button 
#     #       # in the image and please estimate the level of accuracy of the identification.'''          
          
#     #       # followup = '''Identify the 1 or 2 lettered encoding inside of a yellow box that corresponds to the blue Connect button in the image.
#     #       # The 1 or 2 lettered encoding in a yellow box is often next to and to the left of the feature/button it corresponds to.
#     #       # Please output the encoding that corresponds to the blue Connect button
#     #       # in the image and please estimate the level of accuracy of the identification.
#     #       # '''

#     #       followup = '''Tell me the 1 or 2 lettered encoding that appears next to and to the left of the blue Connect button in the image.'''


#     #       vision_res = ask_vision_meta(followup)
#     #       print(vision_res)
#     #       res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
#     #       print(res["message"])
#     #       # res2 = meta.prompt("Given this input, only just output the level of accuracy of the identification as a percentage. " + vision_res)
#     #       # print(res2["message"])

#     #       # time.sleep(2)
#     #       input_keys(res["message"])
#     #       print("Clicked on Connect Button")
#     #       VoiceAssistant.self.voice("Clicked on Connect Button") 
#     #       break
        
#    ################################
#     # find_and_click_add_note_button()
#      ################################

#     local_meta = MetaAI()
#     future = None
#     # Create a ThreadPoolExecutor
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#     # Submit the function to the executor
#         future = executor.submit(generate_personalized_msgs, big_blob_info, local_meta)


#     max_retries = 3
#     retries = 0

#     while retries < max_retries:
#         result1 = find_and_click_connect_button()
#         result2 = find_and_click_add_note_button()
#         if result2 == 0: # good, found the add a note button
#             break  # Exit the loop if func2 returns True
#         else:
#             retries += 1
#             print(f"Retrying buttons {retries}...")

#     # check_for_connect_button = "Given the image, can you identify the large text box near the bottom? Only output YES or NO"
#     # vision_res = ask_vision_meta(check_for_connect_button)
#     # print(vision_res)
#     # if "n" in vision_res.lower():
#     #     print("ENO CONNECT BUTTON!")
        
#     # else:
#     #     input_keys("f")
#     #     followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#     #     Which encoding is associated with the large text box? Start your answer with "The letter encoding is"
#     #     '''
#     #     vision_res = ask_vision_meta(followup)
#     #     print(vision_res)

#     #     res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
#     #     print(res["message"])
#     #     # time.sleep(2)
#     #     input_keys(res["message"])


    
  
#     #  Bad spot 
#     # pyautogui.write(input,interval=0.2)
#     # pynput_keyboard.type(input, 0.5, 0.5)


#     # VOICE: Sent Connection Request
#     # local_meta = MetaAI()
#     # input = generate_personalized_msgs(big_blob_info, local_meta)
#     input = future.result() 
#     VoiceAssistant.self.voice("Here is the best message I came up with", block=True) # must block
#     VoiceAssistant.self.voice(input)
#     fill_in_txtbox(input)
#     # pynput_keyboard.type(input,0.03,0.03)
#     # pyautogui.typewrite(input,interval=0.05)
#     VoiceAssistant.self.voice("Should I sent it or no?", block=True)
#     retry=True
#     while retry:
#       #human_resp = get_mic_input()
#       mic = MicInputRecorder()
#       human_resp = mic.get_mic_input()
#       if 'yes' in human_resp.lower():
#         VoiceAssistant.self.voice("Okay, I will send it!", block=True)
#         pyautogui.press('esc') # use input keys instead when refactoring
        
#         check_for_connect_button = '''Given the image, can you identify a blue "Send" button near the bottom? Only output YES or NO'''
#         vision_res = ask_vision_meta(check_for_connect_button)
#         print(vision_res)
#         if "n" in vision_res.lower():
#             print("BAD BUTTON!... goign back")
#             input_keys("z")
#         input_keys("f") 
#         followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#         Which encoding is associated with the blue "Send" button? Start your answer with "The letter encoding is"
#         '''
#         vision_res = ask_vision_meta(followup)
#         print(vision_res)
#         res = meta.prompt("Given this input, only just output the one or two encoding: " + vision_res)
#         print(res["message"])
        
        
#         input_keys(res["message"])
#         VoiceAssistant.self.voice("Sent Connection Request")
#         retry = False
        
#       # Dont send it..come up with a new one
#       elif 'no' in human_resp.lower():
#         VoiceAssistant.self.voice("No worries! I will generate a new message")
#         # new_msg = generate_personalized_msgs(big_blob_info)
#         res = local_meta.prompt("Pick the next best one or generate a new message that is better")
#         print(res["message"])
#         res = local_meta.prompt("Only ouput the message")
#         print(res["message"])
#         new_msg = res["message"]
        
#         clear_prefilled_txt_box()
#         # pyautogui.write(new_msg, interval=0.2)
#         VoiceAssistant.self.voice("Here is the new message I came up with", block=True)
#         VoiceAssistant.self.voice(new_msg)
#         # pynput_keyboard.type(new_msg,0.03,0.03)
#         # pyautogui.typewrite(new_msg,interval=0.05)
#         fill_in_txtbox(new_msg)
     
#         VoiceAssistant.self.voice("Should I sent it or no?", block=True)
#       elif 'skip' in human_resp.lower():
#          VoiceAssistant.self.voice("Alright! I will not send connection request. Skipping...")
#          go_back_to_profile_pg_from_adding_connection_note();
#          retry = False
#       else:
#         VoiceAssistant.self.voice("I didn't get that. Should i sent it or no?", block=True)   
        
#     # print("ALL DONE!!!!")
#     print("Execute Connection function done!!!!")





# # pyautogui.press('esc') # use input keys instead when refactoring
# # input_keys("gi")
# # time.sleep(1)
# # pyautogui.hotkey('command', 'a')
# # # Delete selected text
# # time.sleep(1)
# # pyautogui.press('delete')
# # time.sleep(1)
# # pyautogui.write("kk")



# def execute_full_connection_process():
  
#   #client = Together(api_key=os.environ.get('TOGETHER_API_KEY'))
#   print("in start_full_conneciotn func...sleeping for 2 secs")
#   time.sleep(2)
#   big_blob= ""
#   one_time_meta = MetaAI()


#   # tts = TTS("tts_models/en/ljspeech/fast_pitch")
#   # with open("tts_instance.pkl", "wb") as f:
#   #     pickle.dump(tts, f)

#   # TESTING TTS
#   # tts = TTS(model_name=tts_models[1]).to(device)
#   # wav = tts.tts(text="Hello world!")
#   # sd.play(wav, 22050)  # Play audio at 22.05 kHz sample rate
#   # # status = sd.wait()  # Wait until audio finishes playing

#   # Text to speech saved to file
#   # tts.tts_to_file(text="Hello world!", file_path="output.wav")
 

#   #find_next_person()
#   # execute_connection("testing 123")
#   scroll_to_top_of_page() # input_keys("gg")
#   scroll_counter = 0
#   # VOICE: "Starting to Connect with new person"
#   VoiceAssistant.self.voice("Starting to Connect with new person")
#   while True:
#       # req_check_interests = "Based on this screenshot of a portion of an person's Linkedin Profile, did we reach the EDUCATION or INTERESTS section of this person's Linkedin Profile? Only output YES or NO"
#       req_check_interests = "Given the image, does the image contain, in whole or in part, an Education or Skills section? Only output YES or NO"

#       vision_res = ask_vision_meta(req_check_interests)
#       print(vision_res)   
#       print("scroll coutner = " + str(scroll_counter))
#       if "yes" in vision_res.lower():
#           print("END OF PROFILE!")
#           #VOICE: Done Gathering profile info
#           VoiceAssistant.self.voice("Done Gathering profile info...Branch 1.. Good")
#           break
#       elif scroll_counter >= 4:
#           # TODO: needs some work  help check state if the vision model is too dumb
#           res = one_time_meta.prompt("Given the below text, does it contain information about someone's education? Just answer YES or NO.  " + vision_res)
#           if "y" in res["message"].lower():
#             #VOICE: Done Gathering profile info
#             VoiceAssistant.self.voice("Done Gathering profile info...Branch 2...Kinda Sus")
#             break
#           break # TODO: CONTROL FLOW IS WEIRD HERE need to fix this whole WHILE LOOP FLOW
#       else:
#           # ** Begin Read **
#           followup = '''I am trying to learn more about an person from their Linkedin profile.
#           Only output relevant or unqiue information about this person from this LinkedIn screenshot. Use bullet points. 
#           '''
#           vision_res = ask_vision_meta(followup)
#           print(vision_res)
#           big_blob = big_blob + (vision_res)
#           # ** End Read **
#           # #VOICE: Scrolling
#           # VoiceAssistant.self.voice("I'm scrolling")
#           # input_keys("dd")
#           scroll_full_page_down()
#           scroll_counter = scroll_counter + 1
  
  

#   # personalized_msg = generate_personalized_msgs(big_blob_info=big_blob)
          
#   # aq1='''
#   # Given the below information about an person based on their Linkedin Profile, please craft 3 different personalized connection request messages for this person.  
#   # Ensure each message is under 400 characters and is informal and sounds exactly like a human. Also, consider making one of the messages about how you are interested in their experiences.'''
#   # ai_res = meta.prompt(aq1 + big_blob)
#   # final_answer_att1 = ai_res["message"]
#   # print(final_answer_att1)
#   # #Voice: thinking of best personalized connection request message
#   # VoiceAssistant.self.voice("thinking of best personalized connection request message")
#   # # while True:
#   # #     user_input = input("Please enter something (type 'q' to exit): ")
#   # #     if user_input.lower() == "q":
#   # #         break
#   # #     # Perform your desired action with the user_input
#   # #     print(f"You entered: {user_input}")
#   # #     #meta_op = meta.prompt(user_input) 
#   # #     res = meta.prompt(user_input)
#   # #     print(res["message"])
#   # #    # check state where state is "Did we reach the interests section of this person's Linkedin Profile?"
#   # #    # if yes: output STOP
#   # #    # if no: ouput important information presented in the image
#   # #     # press d

#   # res = meta.prompt("pick the best one")
#   # print(res["message"])
#   # res = meta.prompt("only ouput the messagee")
#   # print(res["message"])

#   # execute_connection(res["message"])
#   # execute_connection(personalized_msg, big_blob_info=big_blob)
#   execute_connection(big_blob_info=big_blob)
#   print("Start full Connection process func done!!!!") 



# def find_next_person():
#     # VOICE: Let's Find The Next Person To Connect With!
#     VoiceAssistant.self.voice("Let's Find The Next Person To Connect With!")
#     retry = 0
#     while retry < 3:
#       scroll_to_top_of_page() # go to top of page
#       # scroll down enough to see otehr ppl to connect with
#       scroll_half_page_down() #input_keys("d")
#       check_list_of_sim_ppl = "Given the image, can you identify names of other similar people to connect with near the bottom? Only output YES or NO"
#       vision_res = ask_vision_meta(check_list_of_sim_ppl)
#       print(vision_res)
#       meta_res = meta.prompt("Given this input, only tell me if the input is a yes or no: " + vision_res)
#       print(meta_res["message"])
#       if "no" in meta_res["message"].lower():
#           print("ERR..CANT FIND OTHER PPL! Retrying...")
#           # VOICE: "ERR..CANT FIND OTHER PPL!"
#           VoiceAssistant.self.voice("ERR..CANT FIND OTHER PPL! Retrying...")
#           retry = retry + 1 
#       else: 
#           input_keys("f")
#           followup = '''Next to each selectable feature in this image of webpage, there is a 1 or 2 lettered encoding in a yellow box next to the feature.
#           Which encoding is associated with first picture of a face of one of the similar people near the bottom? Start your answer with "The letter encoding is"
#           '''
#           vision_res = ask_vision_meta(followup)
#           print(vision_res)
#           res = meta.prompt("Given this input, only just output the one or two letter encoding: " + vision_res)
#           print(res["message"])

#           # time.sleep(2)
#           input_keys(res["message"])
#           break

