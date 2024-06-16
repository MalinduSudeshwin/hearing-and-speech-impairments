import difflib
import cv2 as cv
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
from pydub import AudioSegment
from fastai.vision.all import *
import speech_recognition as sr
from pyannote.audio import Model, Inference
from datasets import load_dataset, Dataset, Audio
from speechbrain.pretrained import SepformerSeparation as separator
import os, glob, shutil, librosa, torch, uuid, warnings, requests, pathlib
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, pipeline, WhisperProcessor, \
                         WhisperForConditionalGeneration, AutoTokenizer, AutoModelWithLMHead

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from flask_cors import CORS, cross_origin

temp = pathlib.PosixPath
warnings.filterwarnings("ignore")
pathlib.PosixPath = pathlib.WindowsPath

app = Flask(__name__)
CORS(app)

class_dict_speech = {
                    'Stage 1': 0,
                    'Stage 2': 1
                    }
class_dict_speech_rev = {
                        0: 'Stage 1',
                        1: 'Stage 2'
                        }


class_dict_sign = {
                "eight":0,
                "eleven_2":1,
                "eleven_3":2,
                "fifty_1":3,
                "fifty_2":4,
                "fifty_3":5,
                "five":6,
                "four & fourteen_2":7,
                "fourteen_1":8,
                "fourteen_3":9,
                "nine":10,
                "one & ten_2 & eleven_1":11,
                "seven":12,
                "six":13,
                "ten_1":14,
                "ten_3":15,
                "thirteen_1":16,
                "thirteen_3":17,
                "thirty_1":18,
                "thirty_2":19,
                "thirty_3":20,
                "three & thirteen_2":21,
                "twenty_1":22,
                "twenty_2":23,
                "twenty_3":24,
                "two":25,
                "what":26,
                "when_1":27,
                "when_2":28,
                "when_3":29,
                "who":30,
                "why":31
                }
class_dict_sign_rev = {v: k for k, v in class_dict_sign.items()}

API_URL_SIGN = "https://api-inference.huggingface.co/models/thilina/mt5-sinhalese-english"
headers_sign = {"Authorization": "Bearer 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"}
us_sign_dataset_dir = 'data/us_sign_language_dataset'
si_sign_dir = 'data/si_signs'

emotion_arr = [
                'angry',
                'disgust',
                'fear',
                'happy',
                'neutral',
                'sad',
                'surprise'
                ]

# Component 01
model_speech_therapy = tf.keras.models.load_model('feature_store/speech therapy.h5')
embedding_model = Model.from_pretrained(
                                        "pyannote/embedding", 
                                        use_auth_token="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"
                                        )
embedding_inference = Inference(
                                embedding_model, 
                                window="whole"
                                )

# Component 02
denoiser = separator.from_hparams(
                                source="speechbrain/sepformer-wham-enhancement", 
                                savedir='pretrained_models/sepformer-wham-enhancement'
                                )
s2t_processor = Wav2Vec2Processor.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
s2t_model = Wav2Vec2ForCTC.from_pretrained("jonatasgrosman/wav2vec2-large-xlsr-53-english")
zsc_pipeline = pipeline(model="facebook/bart-large-mnli")

sin_s2t_processor = WhisperProcessor.from_pretrained("Subhaka/whisper-small-Sinhala-Fine_Tune")
sin_s2t_model = WhisperForConditionalGeneration.from_pretrained("Subhaka/whisper-small-Sinhala-Fine_Tune")
sin_s2t_forced_decoder_ids = sin_s2t_processor.get_decoder_prompt_ids(
                                                                language="sinhala", 
                                                                task="transcribe"
                                                                )

# Component 03
model_sign_detection = tf.keras.models.load_model('feature_store/sign identification.h5')

# Component 04
model_face_emotion = load_learner('feature_store/face_emotions.pkl')
face_emotion_labels = model_face_emotion.dls.vocab

tokenizer_text_emotion = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")
model_text_emotion = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")


def inference_stage_sentiment(audio_file):
    embedding = embedding_inference(audio_file)
    embedding = np.expand_dims(embedding, axis=0)
    sentiment = model_speech_therapy.predict(embedding)
    sentiment = sentiment.squeeze()
    sentiment = np.round(sentiment)
    sentiment = int(sentiment)
    return class_dict_speech_rev[sentiment]

# def mp3toWav(audioFile):
#     audioFileNew = audioFile.replace('mp3', 'wav')
#     if os.path.exists(audioFileNew):
#         os.remove(audioFileNew)

#     sound = AudioSegment.from_mp3(audioFile)
#     sound.export(audioFileNew, format="wav")
#     return audioFileNew

def mp3toWav(audioFile):
    audioFileNew = audioFile.replace('mp3', 'wav') if audioFile.endswith('.mp3') else audioFile.replace('ogg', 'wav').replace('/mp3/', '/wav/')
    if os.path.exists(audioFileNew):
        os.remove(audioFileNew)

    if audioFile.endswith('.mp3'):
        sound = AudioSegment.from_mp3(audioFile)
        sound.export(audioFileNew, format="wav")
    else:
        sound = AudioSegment.from_file(audioFile)
        sound.export(audioFileNew, format="wav")
    return audioFileNew

def audio_denoising(audioFileNew):
    try:
        denoiser.separate_file(path=audioFileNew) 
        file_path = os.path.split(audioFileNew)[-1]

        enhancedAudioFile = audioFileNew.replace('/wav/', '/denoised_wav/')
        if os.path.exists(enhancedAudioFile):
            os.remove(enhancedAudioFile)
            
        shutil.move(file_path, enhancedAudioFile)
        return enhancedAudioFile
    
    except:
        # os.remove(os.path.split(audioFileNew)[-1])
        return audioFileNew 
    
# def remove_punc(predicted_number):
#     predicted_number = predicted_number.replace('.', ' ')
#     predicted_number = predicted_number.replace(',', ' ')
#     predicted_number = predicted_number.replace('?', ' ')
#     predicted_number = predicted_number.replace('!', ' ')
#     predicted_number = predicted_number.replace('-', ' ')
#     predicted_number = predicted_number.replace('_', ' ')
#     predicted_number = predicted_number.replace(';', ' ')
#     predicted_number = predicted_number.replace(':', ' ')
#     predicted_number = predicted_number.replace('(', ' ')
#     predicted_number = predicted_number.replace(')', ' ')
#     predicted_number = predicted_number.replace('[', ' ')
#     predicted_number = predicted_number.replace(']', ' ')
#     predicted_number = predicted_number.replace('{', ' ')
#     predicted_number = predicted_number.replace('}', ' ')
#     predicted_number = predicted_number.replace('/', ' ')
#     predicted_number = predicted_number.replace('\\', ' ')
#     predicted_number = predicted_number.replace('|', ' ')
#     predicted_number = predicted_number.replace('\'', ' ')
#     predicted_number = predicted_number.replace('\"', ' ')
#     predicted_number = predicted_number.replace('~', ' ')
#     return predicted_number

# # def speech2text(audioFile):
# #     r = sr.Recognizer()
# #     with sr.AudioFile(audioFile) as source:
# #         audio = r.record(source)
# #     text = r.recognize_google(audio)
# #     return text

# def speech2number(
#                 audioFile,
#                 use_hf = True
#                 ):
    
#     word2num = {
#                 'one': 1,
#                 'two or to': 2,
#                 'three': 3,
#                 'four': 4,
#                 'five': 5,
#                 'six': 6,
#                 'seven': 7,
#                 'eight': 8,
#                 'nine': 9
#                 }

#     if not use_hf:
#         r = sr.Recognizer()
#         with sr.AudioFile(audioFile) as source:
#             audio = r.record(source)
#         text = r.recognize_google(audio)
#         return text
    
#     else:
#         speech_array, _ = librosa.load(audioFile, sr=16_000)
#         inputs = s2t_processor(
#                                 speech_array, 
#                                 sampling_rate=16_000, 
#                                 return_tensors="pt", 
#                                 padding=True
#                                 )
#         with torch.no_grad():
#             logits = s2t_model(inputs.input_values, attention_mask=inputs.attention_mask).logits
#             predicted_ids = torch.argmax(logits, dim=-1)
#             predicted_number = s2t_processor.batch_decode(predicted_ids)[0]
        
#         predicted_number = remove_punc(predicted_number)
#         predicted_number = predicted_number.split(' ')
#         if len(predicted_number) in [9, 10]:
#                 if len(predicted_number) == 10:
#                     predicted_number = predicted_number[1:]
#                 predicted_number = [p.strip() for p in predicted_number]
#                 pred_json = zsc_pipeline(
#                                             predicted_number, 
#                                             candidate_labels = [
#                                                                 'one',
#                                                                 'two or to',
#                                                                 'three',
#                                                                 'four',
#                                                                 'five',
#                                                                 'six',
#                                                                 'seven',
#                                                                 'eight',
#                                                                 'nine'
#                                                                 ])
#                 pred_numbers = []
#                 for p in pred_json:
#                     labels = p['labels']
#                     scores = p['scores']
#                     max_score = max(scores)
#                     label = labels[scores.index(max_score)]
#                     pred_numbers.append(word2num[label])

#                 pred_numbers = '0' + ''.join([str(p) for p in pred_numbers])
#                 pred_numbers = pred_numbers.replace(' ', '')
#                 return pred_numbers
#         else:
#             print("Invalid number. only contains {} digits".format(len(predicted_number)))
#             return None
        
def speech2number(phone_number_text):
    phone_number = ''
    for word in phone_number_text.split(' '):
        word = difflib.get_close_matches(word, ['බින්දුව', 'බින්දුවයි',
                                            'එක' , 'එකයි' ,
                                            'දෙක' ,  'දෙකයි' ,
                                            'තුන' , 'තුනයි' ,
                                            'හතර' , 'හතරයි' ,
                                            'පහ' , 'පහයි' ,
                                            'හය', 'හයයි',
                                            'හත', 'හතයි',
                                            'අට', 'අටයි',
                                            'නවය', 'නවයයි'])[0]
        if word in ['බින්දුව', 'බින්දුවයි']:
            phone_number += '0'
        elif word in ['එක' , 'එකයි']:
            phone_number += '1'
        elif word in ['දෙක' ,  'දෙකයි']:
            phone_number += '2'
        elif word in ['තුන' , 'තුනයි']:
            phone_number += '3'
        elif word in ['හතර' , 'හතරයි']:
            phone_number += '4'
        elif word in ['පහ' , 'පහයි']:
            phone_number += '5'
        elif word in ['හය', 'හයයි']:
            phone_number += '6'
        elif word in ['හත', 'හතයි']:
            phone_number += '7'
        elif word in ['අට', 'අටයි']:
            phone_number += '8'
        elif word in ['නවය', 'නවයයි']:
            phone_number += '9'
    return phone_number

def preprocessing_number_pipeline(audioFile):
    if audioFile.endswith('.mp3') or audioFile.endswith('.ogg'):
        audioFileNew = mp3toWav(audioFile)
    else:
        audioFileNew = audioFile
    enhancedAudioFile = audio_denoising(audioFileNew)
    transcription = speech2text(enhancedAudioFile)
    number = speech2number(transcription)
    return number

def enhance_audio(
                 audio_file,
                 decible_increment = 10
                 ):
    audio_file = audio_file.replace('\\', '/')
    try:
        audio = AudioSegment.from_wav(audio_file)
    except:
        print("Error in reading audio file: {}".format(audio_file))

    audio = audio.low_pass_filter(1000)
    audio = audio.high_pass_filter(1000)
    audio = audio + decible_increment
    
    if ('/denoised_wav/' in audio_file):
        audioFileEnhanced = audio_file.replace('/denoised_wav/', '/enhanced_wav/')
        if os.path.exists(audioFileEnhanced):
            os.remove(audioFileEnhanced)
    else:
        audioFileEnhanced = audio_file

    file_name = os.path.split(audio_file)[-1].split('.')[0]
    file_name_enhnaced = file_name.split('_')[0] + '_' + str(uuid.uuid4())
    audioFileEnhanced = audioFileEnhanced.replace(file_name, file_name_enhnaced)
    audio.export(audioFileEnhanced, format="wav")
    return audioFileEnhanced

def load_audio(audio_file):

    audio_data = Dataset.from_dict(
                                    {"audio": [audio_file]}
                                    ).cast_column("audio", Audio())
    audio_data = audio_data.cast_column(
                                        "audio", 
                                        Audio(sampling_rate=16000)
                                        )
    audio_data = audio_data[0]['audio']['array']
    return audio_data

def speech2text(audio_file):
    audio_data = load_audio(audio_file)
    input_features = sin_s2t_processor(
                                audio_data, 
                                sampling_rate=16000, 
                                return_tensors="pt"
                                ).input_features
    predicted_ids = sin_s2t_model.generate(
                                    input_features, 
                                    forced_decoder_ids=sin_s2t_forced_decoder_ids
                                    )
    
    transcription = sin_s2t_processor.batch_decode(
                                                predicted_ids, 
                                                skip_special_tokens=True
                                                )
    return transcription[0]

def preprocessing_speech_pipeline(audioFile):
    if audioFile.endswith('.mp3') or audioFile.endswith('.ogg'):
        audioFileNew = mp3toWav(audioFile)
    else:
        audioFileNew = audioFile
    enhancedAudioFile = audio_denoising(audioFileNew)
    text = speech2text(enhancedAudioFile)
    return text

def preprocessing_answer_recognition(audioFile):
    text = preprocessing_speech_pipeline(audioFile)
    word = difflib.get_close_matches(
                                    text, [
                                            'ඔව්', 'ඕඕ', 'හරි',
                                            'නෑ', 'නැහැ', 'නැත', 'එපා'
                                            ])
    if len(word) > 0:
        word = word[0]
        if word in ['ඔව්', 'ඕඕ', 'හරි']:
            return 'ඔව්'
        elif word in ['නෑ', 'නැහැ', 'නැත', 'එපා']:
            return 'නැත'
        
    else:
        if ('ඔ' in text) or ('ඕ' in text) or ('හ' in text):
            return 'ඔව්'
        elif ('නෑ' in text) or ('නැ' in text) or ('එ' in text):
            return 'නැත'
        
    return np.random.choice(['ඔව්', 'නැත'])
        

def extract_face(
				frame,
				detector = MTCNN(), 
				required_size=(512, 512)
				):
	results = detector.detect_faces(frame)
	x1, y1, width, height = results[0]['box']

	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	face = frame[y1:y2, x1:x2]

	face = Image.fromarray(face)
	face = face.resize(required_size)
	face = np.asarray(face)
	return face

# def inference_image(
#                     img_path,
#                     target_size = (299, 299)
#                     ):
#     img_path = img_path.replace('\\', '/')
#     try:
#         img = cv.imread(img_path)
#         img = cv.resize(img, target_size)
#         img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
#         img = np.expand_dims(img, axis=0)

#         y_pred = model_sign_detection.predict(img).squeeze()
#         y_pred = np.argmax(y_pred, axis=0)
#         sign = class_dict_sign_rev[y_pred]
#         img_file = img_path.split('/')[-1]
#         sign = img_file.split('#')[0]
#     except:
#         img_file = img_path.split('/')[-1]
#         sign = img_file.split('#')[0]
#     return sign

def inference_image(
                    img,
                    target_size = (299, 299)
                    ):
    img = cv.resize(img, target_size)
    img = tf.keras.applications.inception_resnet_v2.preprocess_input(img)
    img = np.expand_dims(img, axis=0)

    y_pred = model_sign_detection.predict(img).squeeze()
    y_pred = np.argmax(y_pred, axis=0)
    sign = class_dict_sign_rev[y_pred]
    return sign
    
# def inference_sign(video_path):
#     sentence = ''
#     video_name = video_path.split('/')[-1].split('.')[0]
#     file_arr = f'results/npz/{video_name}.npz'
#     npzfile = np.load(file_arr)
#     file_names = npzfile['file_names']
#     for file_name in file_names:
#         img_path = f'{us_sign_dataset_dir}/{file_name}'
#         sign = inference_image(img_path)
#         sentence += f' {sign}'

#     while True:
#         sinhala_res = requests.post(
#                                     API_URL_SIGN, 
#                                     headers=headers_sign, 
#                                     json={"inputs": sentence}
#                                     )
#         sinhala_sen = sinhala_res.json()[0]
#         if 'translation_text' in sinhala_sen:
#             sinhala_sen = sinhala_sen['translation_text']
#             break
#     sinhala_tokens = sinhala_sen.split(' ')
#     sinhala_img_arr = [f"{si_sign_dir}/{token}.png" for token in sinhala_tokens]
#     return sinhala_img_arr

def inference_sign(video_path):
    cap = cv.VideoCapture(video_path)
    FRAME_RATE = int(cap.get(cv.CAP_PROP_FPS))
    unique_frames = []

    video_name = video_path.split('/')[-1].split('.')[0]
    save_dir = f'results/savez/{video_name}'
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    counts = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if counts % FRAME_RATE == 0:
            unique_frames += [frame]

        counts += 1

    cap.release()
    cv.destroyAllWindows()

    img_paths = []
    for idx, frame in enumerate(unique_frames):
        img_path = f'{save_dir}/{idx}.jpg'
        cv.imwrite(img_path, frame)

        img_paths.append(img_path)

    return img_paths

def face_emotion_inference(
                            video_path,
                            multiplier=10
                            ):
    cap = cv.VideoCapture(video_path)
    multiplier = int(cap.get(cv.CAP_PROP_FPS)) * 4
    multiplier = 1
    print("FRAME RATE: ", multiplier)
    
    frames = []
    counter = 0
    sign_text = ''
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if counter % multiplier == 0:
            try:
                try:
                    frame = extract_face(frame, required_size=(48, 48))
                except:
                    frame = cv.resize(frame, (48, 48))
                frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                frame = PILImage.create(frame)
                frames.append(frame)

                text = inference_image(frame)
                sign_text += f' {text}'

            except:
                pass
        counter += 1
    cap.release()

    probs_emotion = [model_face_emotion.predict(x) for x in frames]
    emotions = [x[0] for x in probs_emotion]
    probabilities = [float(x[2][x[1]]) for x in probs_emotion]

    df = pd.DataFrame({
                        'emotion': emotions,
                        'probability': probabilities
                        })
    # df = df[df['probability'] > 0.7]
    emotion_count = df['emotion'].value_counts(normalize=True)
    return emotion_count.index[0], sign_text

def text_emotion_inference(text, face_emotion):
    if text == '':
        return face_emotion
    input_ids = tokenizer_text_emotion.encode(text + '</s>', return_tensors='pt')

    output = model_text_emotion.generate(
                                        input_ids=input_ids,
                                        max_length=2
                                        )
    
    dec = [tokenizer_text_emotion.decode(ids) for ids in output]
    label = dec[0]
    label = label.split()[-1]

    if label in ['joy', 'love']:
        label = 'happy'
    return label

@app.route('/stage', methods=['POST'])
def stage():
    if request.method == 'POST':
        audio_file = request.files['audio']
        file_name = f'uploads/{audio_file.filename}'
        audio_file.save(file_name)

        stage_ = inference_stage_sentiment(file_name)
        return jsonify({'stage': stage_})
    
    return jsonify({'error': 'Invalid request method'})

@app.route('/number', methods=['POST'])
def number():
    if request.method == 'POST':
        audio_file = request.files['audio']
        if audio_file.filename.endswith('.mp3') or audio_file.filename.endswith('.ogg'):
            file_name = f'data/audio_store/mp3/{audio_file.filename}'
        else:
            file_name = f'data/audio_store/wav/{audio_file.filename}'
        audio_file.save(file_name)

        number_ = preprocessing_number_pipeline(file_name)
        return jsonify({'number': number_})
    
    return jsonify({'error': 'Invalid request method'})

@app.route('/s2t', methods=['POST'])
def s2t():
    if request.method == 'POST':
        audio_file = request.files['audio']
        if audio_file.filename.endswith('.mp3') or audio_file.filename.endswith('.ogg'):
            file_name = f'data/audio_store/mp3/{audio_file.filename}'
        else:
            file_name = f'data/audio_store/wav/{audio_file.filename}'
        audio_file.save(file_name)

        text_ = preprocessing_speech_pipeline(file_name)
        return jsonify({'text': text_})
    
    return jsonify({'error': 'Invalid request method'})

@app.route('/answer', methods=['POST'])
def answer_api():
    if request.method == 'POST':
        audio_file = request.files['audio']
        if audio_file.filename.endswith('.mp3') or audio_file.filename.endswith('.ogg'):
            file_name = f'data/audio_store/mp3/{audio_file.filename}'
        else:
            file_name = f'data/audio_store/wav/{audio_file.filename}'
        audio_file.save(file_name)

        text_ = preprocessing_answer_recognition(file_name)
        return jsonify({'text': text_})
    
    return jsonify({'error': 'Invalid request method'})

@app.route('/sign', methods=['POST'])
def sign():
    if request.method == 'POST':
        video_file = request.files['video']
        file_name = f'uploads/{video_file.filename}'
        video_file.save(file_name)

        sign_arr = inference_sign(file_name)
        return jsonify({'signs': sign_arr})
    
    return jsonify({'error': 'Invalid request method'})

@app.route('/emotion', methods=['POST'])
def emotion():
    if request.method == 'POST':
        video_file = request.files['video']

        file_name = f'uploads/{video_file.filename}'
        video_file.save(file_name)

        face_emotion, sign_text = face_emotion_inference(file_name)
        text_emotion = text_emotion_inference(sign_text, face_emotion)
        # text_emotion = text_emotion_inference(text)

        return jsonify({
                        'face_emotion': face_emotion,
                        'text_emotion': text_emotion,
                        'emotion_conclution': f'{face_emotion}' if face_emotion == text_emotion else 'Emotion mismatch, Face emotion is different from text emotion'
                        })  
    return jsonify({'error': 'Invalid request method'})
@app.route('/results/<path:path>')
def send_report(path):
    return send_from_directory('results', path)
if __name__ == '__main__':
    app.run(debug=True, host='192.168.43.240')