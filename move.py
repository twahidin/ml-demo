import streamlit as st
import av
import csv
import os
import re
import pandas as pd
import numpy as np 
import pickle 
import mediapipe as mp
from landmarks import landmarks
import cv2
import streamlit_antd_components as sac
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import pickle

current_stage = ''
counter = 0 
bodylang_prob = np.array([0,0]) 
bodylang_class = '' 
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
init = False
model_name = './data/finalise_model.sav' #sports name can be changed with s_option
coords = './data/coords.csv'
num_coords = 33


with open('deadlift.pkl', 'rb') as f: 
	model = pickle.load(f)

def callback(frame):
	global current_stage
	global counter
	global bodylang_class
	global bodylang_prob

	image = frame.to_ndarray(format="bgr24")
	results = pose.process(image)
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
			mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2), 
			mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2)) 

		try: 
			row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
			# pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
			X = pd.DataFrame([row], columns = landmarks) 
			bodylang_prob = model.predict_proba(X)[0]
			bodylang_class = model.predict(X)[0]
			if bodylang_class =="down" and bodylang_prob[bodylang_prob.argmax()] > 0.7: 
				current_stage = "down" 
			elif current_stage == "down" and bodylang_class == "up" and bodylang_prob[bodylang_prob.argmax()] > 0.7:
				current_stage = "up" 
				counter += 1 
			image = cv2.putText(image,bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()]), (00,20), #the webcam resolution
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

		except Exception as e: 
			print(e)
	else:
		print("No landmarks found") 

	img = image[:, :500, :] 
	
	return av.VideoFrame.from_ndarray(img, format="bgr24")


def analysis_callback(frame, model):
	image = frame.to_ndarray(format="bgr24")
	results = pose.process(image)
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
			mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2), 
			mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2)) 

		try: 
			row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
			# pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
			X = pd.DataFrame([row], columns = landmarks) 
			bodylang_prob = model.predict_proba(X)[0]
			bodylang_class = model.predict(X)[0]
			image = cv2.putText(image,bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()]), (00,20), #the webcam resolution
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

		except Exception as e: 
			print(e)
	else:
		print("No landmarks found") 

	img = image[:, :500, :] 
	
	return av.VideoFrame.from_ndarray(img, format="bgr24")

def photo_callback(image, model):
	results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, 
			mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2), 
			mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2)) 

		try: 
			row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
			X = pd.DataFrame([row], columns=landmarks) 
			bodylang_prob = model.predict_proba(X)[0]
			bodylang_class = model.predict(X)[0]
			image = cv2.putText(image, bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()]), (0,20),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
		except Exception as e: 
			print(e)
	else:
		print("No landmarks found")

	img = image[:, :500, :]
	return img, round(bodylang_prob[bodylang_prob.argmax()]*100), bodylang_class


def train_callback(frame, pose_name):
	image = frame.to_ndarray(format="bgr24")
	results = pose.process(image)
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
								  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
								  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))

		try:
			row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
			with open('./data/coords.csv', mode='a', newline='') as f:
				csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
				row.insert(0, pose_name)
				csv_writer.writerow(row)

		except Exception as e:
			print(e)
	else:
		print("No landmarks found")
	img = image[:, :500, :]
	return av.VideoFrame.from_ndarray(img, format="bgr24")

def initialise_file():
	num_coords = 33
	landmarks = ['class']
	for val in range(1, num_coords+1):
		landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
	with open('./data/coords.csv', mode='w', newline='') as f:
		csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csv_writer.writerow(landmarks)


# Main function for processing the image
def photo_train(img, pose_name):
	image_array = np.array(img)
	image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
	results = pose.process(image_rgb)
	
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
								  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
								  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))
	try:
		
		row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
		with open('./data/coords.csv', mode='a', newline='') as f:
			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			row.insert(0, pose_name)
			csv_writer.writerow(row)

	except Exception as e:
		print(e)

	return Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def photo_analyse(img, model):
	image_array = np.array(img)
	image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
	results = pose.process(image_rgb)
	
	if results.pose_landmarks:
		mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
								  mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius = 2),
								  mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius = 2))
		try: 
			row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
			X = pd.DataFrame([row], columns=landmarks) 
			bodylang_prob = model.predict_proba(X)[0]
			bodylang_class = model.predict(X)[0]
			image_rgb = cv2.putText(image_rgb, bodylang_class + ": " + str(bodylang_prob[bodylang_prob.argmax()]), (0,20),
					   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
			st.write(":blue[Probability] : ", bodylang_prob[bodylang_prob.argmax()]*100, ':red[Pose] :', bodylang_class)
			pass
		except Exception as e: 
			print(e)
	else:
		print("No landmarks found")
	return Image.fromarray(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))

# # Callback for storing data
# def photo_show(img, pose_name):
# 	image_array = np.array(img)
# 	results = pose.process(image_array)
# 	#results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# 	if results.pose_landmarks:
# 		row = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten().tolist()
# 		with open('./data/coords.csv', mode='a', newline='') as f:
# 			csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# 			row.insert(0, pose_name)
# 			csv_writer.writerow(row)


def train_data_model():
	sport = st.session_state.sport
	name = st.session_state.name

	# Create a new model_name based on the name and sports
	model_name = f'./data/{name}_{sport}_finalise_model.sav'

	dft = pd.read_csv('./data/coords.csv')
	X = dft.drop('class', axis=1)
	y = dft['class']
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
	pipelines = {
		#'lr':make_pipeline(StandardScaler(), LogisticRegression()), #Prediction Probability
		#'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
		'rf':make_pipeline(StandardScaler(), RandomForestClassifier()), #Prediction Probability
		#'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
	} #Isotonic Regression is another training model with prediction probability

	fit_models = {}
	for algo, pipeline in pipelines.items():
		model = pipeline.fit(X_train, y_train)
		fit_models[algo] = model
	st.write("Training Completed")

	# Create the directory if it doesn't exist
	dir_path = os.path.dirname(model_name)
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)

	# Save the model
	pickle.dump(fit_models['rf'], open(model_name, 'wb'))
	st.write(f'Model saved as {model_name}')
	
	# Add the trained model to the model_list
	st.session_state.model_list.append({"name_sport": f"{name}_{sport}", "model_name": model_name})

def load_existing_models():
	data_folder = './data'
	model_pattern = re.compile(r'^(.+)_(.+)_finalise_model\.sav$')

	# Check if st.session_state.model_list exists, if not, initialize it
	if not hasattr(st.session_state, 'model_list'):
		st.session_state.model_list = []

	# Check if the data folder exists
	if os.path.exists(data_folder):
		# List all files in the data folder
		files = os.listdir(data_folder)

		# Iterate over the files
		for file in files:
			# Match the file name with the model pattern
			match = model_pattern.match(file)
			if match:
				# Extract name and sport from the file name
				name, sport = match.groups()

				# Construct model_name based on the file name
				model_name = os.path.join(data_folder, file)

				# Add the model to st.session_state.model_list
				st.session_state.model_list.append({"name_sport": f"{name}_{sport}", "model_name": model_name})

# Function to transfer the model
def transfer_model():
	# List the names of the models in the st.session_state.model_list
	
	model_names = [model["name_sport"] for model in st.session_state.model_list]
	# Use the sac.transfer component to display the model names for selection
	transfer_result = sac.transfer(items=model_names, label="Select one model only", index=None, titles=['Models', 'Selected'],
										  format_func='upper', search=True, pagination=False, oneway=True,
										  disabled=False, width='100%', height=None, return_index=False)
	# Check if a single model is selected
	if transfer_result == None or transfer_result == []:
		return False
	elif transfer_result[0] != None and len(transfer_result) < 2:
		# Load the selected model
		selected_model_name = transfer_result[0]
		selected_model_path = next(model["model_name"] for model in st.session_state.model_list if model["name_sport"] == selected_model_name)
		loaded_model = pickle.load(open(selected_model_path, 'rb'))
		st.write(f'Model {selected_model_name} loaded')
		return loaded_model
	else:
		return False

def delete_models():
	# Check if st.session_state.model_list exists, if not, just return
	user_choice = 'Cancel'
	if not hasattr(st.session_state, 'model_list'):
		st.write("No models found in the session state to delete.")
		return

	# List the names of the models in the st.session_state.model_list
	model_names = [model["name_sport"] for model in st.session_state.model_list]

	# If no models are in the list, inform the user
	if not model_names:
		st.write("No models found in the data directory to delete.")
		return

	# Use the sac.transfer component to display the model names for selection
	transfer_result = sac.transfer(items=model_names, label="Select models to delete", index=None, titles=['Available Models', 'Selected to Delete'],
									format_func='upper', search=True, pagination=False, oneway=True,
									disabled=False, width='100%', height=None, return_index=False)
	
	if st.button('Delete model'):
		user_choice = 'Delete'
	# If user chooses 'Delete', delete the selected models, else just return
	if user_choice == 'Delete':
		if transfer_result:
			for selected_model_name in transfer_result:
				selected_model_path = next(model["model_name"] for model in st.session_state.model_list if model["name_sport"] == selected_model_name)
				# Check if the file exists and delete
				if os.path.exists(selected_model_path):
					os.remove(selected_model_path)
					st.write(f'Model {selected_model_name} deleted.')
					# Also remove from st.session_state.model_list
					st.session_state.model_list = [model for model in st.session_state.model_list if model["model_name"] != selected_model_path]
					return
		else:
			st.write("No models selected for deletion.")
	else:
		return


def photo_upload_input():
	pose_name = st.text_input("Enter pose name: ")
	if pose_name:
		uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
		if uploaded_file:
			img = Image.open(uploaded_file)
			result_img = photo_train(img, pose_name)
			st.image(result_img)
			st.success("Analyse Success!")
			st.write("Upload a new photo for another pose or the same pose")
			if result_img:
				st.session_state.pose_counter += 1
				uploaded_file
	else:
		st.warning("Please enter a pose name.")
	
	if st.session_state.pose_counter > 2:
		if st.button("Train Data"):
			train_data_model()
			st.session_state.pose_counter = 0
			st.session_state.form_submit = False

def photo_capture_input():
	pose_name = st.text_input("Enter pose name: ")
	if pose_name:
		img_file_buffer = st.camera_input('Please capture your pose')
		if img_file_buffer is not None:
			img = Image.open(img_file_buffer )
			result_img = photo_train(img, pose_name)
			st.image(result_img)
			st.success("Analyse Success!")
			st.write("Take another photo for another pose or the same pose")
			if result_img:
				st.session_state.pose_counter += 1
				img_file_buffer = None
	else:
		st.warning("Please enter a pose name.")
	
	if st.session_state.pose_counter > 2:
		if st.button("Train Data"):
			train_data_model()
			st.session_state.form_submit = False
			st.session_state.pose_counter = 0