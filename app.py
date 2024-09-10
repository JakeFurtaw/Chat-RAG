# This file is used to launch the program if the user wants to launch it using python argument versus gradio
from chatrag import demo

demo.launch(inbrowser=True, share=True)