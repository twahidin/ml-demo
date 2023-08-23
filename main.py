import streamlit as st
from streamlit_webrtc import webrtc_streamer
from twilio.rest import Client
import numpy as np 
import mediapipe as mp
import streamlit_antd_components as sac
from move import callback, train_callback, train_data_model, initialise_file, analysis_callback, photo_callback, transfer_model, load_existing_models, delete_models, photo_upload_input, photo_capture_input, photo_analyse
from functools import partial
from PIL import Image
import av
import threading
import time

#variable declaration
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic # Mediapipe Solutions
# file_initialise = False
# capture_landmark = False
# upload_img = False
# train_model = False
# capture_complete = False
# photo_capture = False
#model_name = './data/finalise_model.sav' #sports name can be changed with s_option
coords = './data/coords.csv'
# img_list = []
# prob_list = []
# class_list = []
# analysis_complete = False
# a_model = ''
# X = ''
# img_counter = 5
# init = False


#Functions to switched on camera
#Function to change captured camera values/resolution
cx = 1280
cy = 720

lock = threading.Lock()
img_container = {"img": None}

def video_frame_callback(frame):
	image = frame.to_ndarray(format="rgb24")
	#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	with lock:
		img_container["img"] = image
	return frame

def main():
	st.title(":blue[ITD Showcase 2023 - MOVE App V2]")
	sac.divider(label='ML/LLM showcase', icon='app', align='start', direction='horizontal', dashed=False, bold=False)

	if "sport" not in st.session_state:
		st.session_state.sport = ""

	if "pose" not in st.session_state:
		st.session_state.pose = ""
	
	if "name" not in st.session_state:
		st.session_state.name = ""
	
	if "form_submit" not in st.session_state:
		st.session_state.form_submit = False

	if "upload_option" not in st.session_state:
		st.session_state.upload_option = ""
	
	if "pose_counter" not in st.session_state:
		st.session_state.pose_counter = 0

	if "pose_list" not in st.session_state:
		st.session_state.pose_list = []
	
	if "model_list" not in st.session_state:
		st.session_state.model_list = []
		load_existing_models()
	
	if "start_train" not in st.session_state:
		st.session_state.start_train = False

	if "num_coords" not in st.session_state:
		st.session_state.num_coords = 0

	with st.sidebar:
		option = sac.menu([sac.MenuItem('home', icon='house', tag='ITD'),
			sac.MenuItem('MOVE App', icon='app', children=[
			sac.MenuItem('Demo', icon='file-person', tag='Pre-train'),
			sac.MenuItem('Training', icon='person-bounding-box', tag='SckitLearn'),
			sac.MenuItem('Analysis', icon='person-check-fill', children=[
				sac.MenuItem('Photo Analysis', icon='person-video2', tag='Image'),
				sac.MenuItem('Live Analysis', icon='person-video', tag='Live'),
			],
				tag='MediaPipe'),
		]),
		sac.MenuItem(type='divider'),
		sac.MenuItem('Settings', icon='gear'),
		sac.MenuItem('Refresh', icon='box-arrow-right')
		], index=0, format_func='title', size='middle', indent=24, open_index=None, open_all=True, return_index=False)

	if option == 'home':

		st.subheader(":red[MOVE app prototype 2021]")
		st.write("""
		This application aims to analyze and assess body movements, making it an excellent tool for Physical Education in schools. It uses machine learning and computer vision techniques to detect and track body landmarks, providing real-time insights into your movements.

		Developed as a prototype for Physical Education classes, this tool can offer a valuable way to improve physical exercises and performance.

		**Features:**

		1. **Real-time Pose Detection:** Utilizes the MediaPipe library to identify body landmarks and classify different poses in real-time.

		2. **Machine Learning Analysis:** Uses various ML models such as Random Forest Classifier to analyze data and predict poses.

		3. **Training Option:** You can train new poses and movements by providing samples in real-time, and the app will learn to recognize those poses.

		4. **Interactive Dashboard:** Provides a user-friendly dashboard for users to interact with the app and see real-time analysis.

		To use this application, open the real-time camera stream, start performing body movements, and the app will analyze and provide insights into your movements.

		To get started:

		1. Train your data under the Training sub menu
		2. Perform the movements you want to catpure with a minimum of 2 poses and save the model
		3. Use _Photo Analysis_ to capture still images of each pose for image analysis up to 5 images
		4. Use _Live Analysis_ for real time analysis

		Note: Make sure to grant permission for the app to access your webcam.
		""")

		# Include the rest of your Streamlit code below this line.

	elif option == 'Demo':
		num = 30
		st.subheader(":red[Demonstration of ML code by Nicholas Renotte]")
		account_sid = st.secrets['TWILIO_ACCOUNT_SID']
		auth_token = st.secrets['TWILIO_AUTH_TOKEN']
		client = Client(account_sid, auth_token)
		token = client.tokens.create()
		webrtc_streamer(key="example", video_frame_callback=callback, rtc_configuration={"iceServers": token.ice_servers})
	
	elif option == 'Training':
		st.subheader(":red[Training of data using RandomForest Classifier]")
		if st.session_state.form_submit == False:
			placeholder1 = st.empty()
			with placeholder1:
				with st.form("my_form"):
					st.write("You are going to create a new model when you start this training")
					name = st.text_input("Enter your name:", max_chars=15)
					sport = st.text_input("Enter your sport:", max_chars=15)
					option = st.selectbox('Select the method of training',
											('Photo uploads', 'Photo Capture','Video Capture'))
					
					# Every form must have a submit button.
					submitted = st.form_submit_button("Submit")
					if submitted:
						if sport and name and option:
							st.session_state.sport = sport
							st.session_state.name = name
							st.session_state.upload_option = option
							st.session_state.form_submit = True
							initialise_file()
							placeholder1.empty()
						else:
							st.write("Please enter a name, sport, training options, you would like to train")
		
		if st.session_state.form_submit == True:
			if st.session_state.upload_option == 'Photo uploads':
				photo_upload_input()
				pass
			elif st.session_state.upload_option == 'Photo Capture':
				photo_capture_input()
				pass
			else: #Live Capture
				account_sid = st.secrets['TWILIO_ACCOUNT_SID']
				auth_token = st.secrets['TWILIO_AUTH_TOKEN']
				client = Client(account_sid, auth_token)
				token = client.tokens.create()
				#timer and pose is correct
				st.write("**You have 10 seconds to record each pose only**")
				st.session_state.pose = st.text_input("Enter a pose name:", max_chars=15)
				pose_name = st.session_state.pose
				st.write("Training pose: ", pose_name)
				if pose_name:
					st.session_state.pose_counter += 1
					callback_with_name = partial(train_callback, pose_name=pose_name)
					webrtc_streamer(key="example", video_frame_callback=callback_with_name, rtc_configuration={"iceServers": token.ice_servers})
					complete  = sac.switch(label="Completed my training", value=False, checked=None, unchecked=None, align='start', position='top', size='default', disabled=False)
					if complete:
						if st.session_state.pose_counter < 2:
							st.write("Please record a minimum of 2 poses")
							st.write("Training of data cannot be completed")
						else:
							st.session_state.form_submit = False
							st.session_state.pose_counter = 0
							train_data_model()
				else:
					st.write("Please enter a pose name")
				pass
	elif option == 'Live Analysis':
		st.subheader(":red[Live Analysis]")
		model = transfer_model()
		if model is not False:
			account_sid = st.secrets['TWILIO_ACCOUNT_SID']
			auth_token = st.secrets['TWILIO_AUTH_TOKEN']
			client = Client(account_sid, auth_token)
			token = client.tokens.create()
			callback_model = partial(analysis_callback, model=model)
			webrtc_streamer(key="example", video_frame_callback=callback_model, rtc_configuration={"iceServers": token.ice_servers})
		else:
			st.write("Model not loaded")
		pass

	elif option == 'Photo Analysis':
		st.subheader(":red[Photo Analysis]")
		model = transfer_model()
		option = st.selectbox('Please select input method to analyse your photos:', ['Photo upload', 'Live 5 photos capture'])
		if option == 'Live 5 photos capture':
			if model is not False:
				counter = 0
				account_sid = st.secrets['TWILIO_ACCOUNT_SID']
				auth_token = st.secrets['TWILIO_AUTH_TOKEN']
				client = Client(account_sid, auth_token)
				token = client.tokens.create()
				ctx = webrtc_streamer(key="example", video_frame_callback=video_frame_callback, rtc_configuration={"iceServers": token.ice_servers})
				while ctx.state.playing and counter < 5 and len(st.session_state.pose_list) < 6:
					time.sleep(0.5)
					with lock:
						img = img_container["img"]
						if img is not None:
							st.session_state.pose_list.append(img)
							counter += 1
							if counter == 5:
								continue
			
				st.write(f"Captured {len(st.session_state.pose_list)} images so far.")
				if st.button("Show Images"):
					# Determine how many images to display
					num_images = len(st.session_state.pose_list)
					if num_images == 0:
						return
					
					# Create columns for each image
					cols = st.columns(num_images)
					
					for i, img in enumerate(st.session_state.pose_list):
						result_img, prob, class_pose = photo_callback(img, model)
						
						with cols[i]:
							st.image(result_img, use_column_width=True)
							st.write("Probability: ", prob, "%")
							st.write("Pose: ", class_pose)

					if st.button("Reset Images"):
						st.session_state.pose_list = []
				pass
			else:
				st.write("Model not loaded")
		else:
			if model is not False:
				uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
				if uploaded_file:
					img = Image.open(uploaded_file)
					result_img = photo_analyse(img, model)
					st.image(result_img)
				pass
			else:
				st.write("Model not loaded")
				pass
		
	elif option == 'Settings':
		delete_models()
		pass

	elif option == 'Refresh':
		for key in st.session_state.keys():
			del st.session_state[key]
		st.experimental_rerun()

if __name__ == "__main__":
	main()
